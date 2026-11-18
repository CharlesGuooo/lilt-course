"""
冻结 LiLT 上的轻量 head · 公共工具。每一章的脚本都 import 这里。

- load_funsd()        下载并读取 FUNSD；词按几何阅读顺序排好，标签是扁平的 4 类（没有 BIO）
- extract_features()  冻结 LiLT 跑一遍，把每个词的特征缓存到磁盘（只跑一次）
- word_feats()        从缓存里取某种“输出契约”的特征：text 768 / layout 192 / both 960 / last4 3264
- train_head()        统一的训练循环：同样的优化器、轮数上限和早停规则
- evaluate()          逐词 P/R/F1、宏平均、ECE（置信度校准）、实体全对率
- latency_ms()        单线程 CPU 计时（p50 / p95）
"""
import io
import json
import math
import os
import pickle
import random
import statistics
import time
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")     # 不打印“序列超过 512”之类的提示：我们自己切窗口
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

CKPT = "SCUT-DLVCLab/lilt-roberta-en-base"
TOKENIZER_CKPT = "roberta-base"          # LiLT-roberta-en-base 的文本流就是 roberta-base，词表相同
FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("HEADS_CACHE", os.path.join(HERE, ".cache"))

LABELS = ["other", "question", "answer", "header"]   # 扁平标签：other = 不属于目标字段（Other）
FIELD_IDS = [1, 2, 3]                                 # 宏平均只算三个字段类
VIEWS = {"text": 768, "layout": 192, "both": 960, "last4": 4 * 768 + 192}
SEEDS = [0, 1, 2]


# ----------------------------------------------------------------------------
# 打印
# ----------------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def pct(x):
    return f"{100 * x:.1f}%"


# ----------------------------------------------------------------------------
# FUNSD：下载、读取、阅读顺序
# ----------------------------------------------------------------------------
def _png_size(fp):
    head = fp.read(24)                    # PNG 的 IHDR：第 16-24 字节是宽和高
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def reading_order(boxes):
    """
    几何阅读顺序（模拟 OCR 的输出顺序）：先按纵向中心分行，行内从左到右。
    同一行的判定：和这一行第一个词的纵向中心相差不超过 0.5 个中位词高。
    这是启发式，不是真 OCR 引擎的顺序；多栏表单会把左右两栏交错在同一行里。
    """
    yc = [(b[1] + b[3]) / 2 for b in boxes]
    med_h = statistics.median([max(1, b[3] - b[1]) for b in boxes])
    lines, cur, cur_y = [], [], None
    for i in sorted(range(len(boxes)), key=lambda i: (yc[i], boxes[i][0])):
        if cur and abs(yc[i] - cur_y) > 0.5 * med_h:
            lines.append(cur)
            cur = []
        if not cur:
            cur_y = yc[i]
        cur.append(i)
    if cur:
        lines.append(cur)
    return [i for line in lines for i in sorted(line, key=lambda i: boxes[i][0])]


def load_funsd():
    """
    返回 {"train": [...], "dev": [...], "test": [...]}，每份文档是 dict：
      id, words, boxes_px, boxes（0..1000 归一化）, labels（LABELS 下标）, ent（词所属实体编号）, size
    官方训练集 149 份里每隔 5 份取 1 份做 dev（30 份），其余 119 份训练；官方测试集 50 份只用来报告结果。
    """
    os.makedirs(CACHE, exist_ok=True)
    zpath = os.path.join(CACHE, "funsd_dataset.zip")
    if not os.path.exists(zpath):
        print(f"[下载] {FUNSD_URL}")
        urllib.request.urlretrieve(FUNSD_URL, zpath)
    out = {"train": [], "dev": [], "test": []}
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        for split, folder in [("train", "training_data"), ("test", "testing_data")]:
            anns = sorted(n for n in names if n.startswith(f"dataset/{folder}/annotations/") and n.endswith(".json"))
            for k, n in enumerate(anns):
                doc_id = os.path.basename(n)[:-5]
                with z.open(f"dataset/{folder}/images/{doc_id}.png") as fp:
                    w_img, h_img = _png_size(fp)
                form = json.loads(z.read(n))["form"]
                words, boxes, labels, ent = [], [], [], []
                lab_of = {e["id"]: e["label"] for e in form}
                links = sorted({(a, b) for e in form for a, b in e["linking"]
                                if lab_of.get(a) == "question" and lab_of.get(b) == "answer"})
                for e in form:
                    for w in e["words"]:
                        if not w["text"].strip():
                            continue
                        words.append(w["text"])
                        boxes.append(w["box"])
                        labels.append(LABELS.index(e["label"]))
                        ent.append(e["id"])
                order = reading_order(boxes)
                doc = {
                    "id": doc_id,
                    "words": [words[i] for i in order],
                    "boxes_px": [boxes[i] for i in order],
                    "labels": np.array([labels[i] for i in order]),
                    "ent": np.array([ent[i] for i in order]),
                    "links": links,                      # FUNSD 标注的 (question 实体, answer 实体) 配对
                    "size": (w_img, h_img),
                }
                doc["boxes"] = np.array([normalize_box(b, w_img, h_img) for b in doc["boxes_px"]])
                target = "test" if split == "test" else ("dev" if k % 5 == 0 else "train")
                out[target].append(doc)
    return out


def normalize_box(box, width, height):
    """像素坐标 -> 0..1000 整数坐标（和官方 LiLT 代码的 normalize_bbox 一样）"""
    x0, y0, x1, y1 = box
    c = lambda v, s: min(1000, max(0, int(1000 * v / s)))
    return [c(x0, width), c(y0, height), c(x1, width), c(y1, height)]


# ----------------------------------------------------------------------------
# 冻结 LiLT：每个词一个特征（缓存到磁盘）
# ----------------------------------------------------------------------------
def load_backbone():
    from transformers import LiltModel, RobertaTokenizerFast
    model = LiltModel.from_pretrained(CKPT).eval()
    for p in model.parameters():
        p.requires_grad_(False)                       # 冻结：一个参数都不训练
    tok = RobertaTokenizerFast.from_pretrained(TOKENIZER_CKPT, add_prefix_space=True)
    return model, tok


def windows_by_tokens(n_tok_per_word, max_tok=510):
    """把词按顺序切成若干窗口，每个窗口的 subword 数不超过 max_tok（再加 <s> </s> 正好 512）"""
    wins, start, used = [], 0, 0
    for i, n in enumerate(n_tok_per_word):
        if used + n > max_tok:
            wins.append((start, i))
            start, used = i, 0
        used += n
    wins.append((start, len(n_tok_per_word)))
    return wins


@torch.no_grad()
def encode_doc(model, tok, words, boxes):
    """
    一份文档 -> 每个词取第一个 subword 位置上的特征：
      text4  [W, 4, 768]  文本流最后 4 层（第 9..12 层）
      layout [W, 192]     layout 流最后一层
    超过 512 token 的文档切成不重叠的窗口，各自前向（窗口边界处的词看不到另一窗口的上下文）。
    """
    enc = tok(words, is_split_into_words=True, add_special_tokens=False)
    ids, wid = enc["input_ids"], enc.word_ids()
    per_word = [[] for _ in words]
    for t, w in enumerate(wid):
        per_word[w].append(t)
    text4 = np.zeros((len(words), 4, 768), np.float16)
    layout = np.zeros((len(words), 192), np.float16)
    base = model
    taps = {}
    hooks = [base.encoder.layer[i].output.register_forward_hook(
        lambda _m, _i, out, i=i: taps.__setitem__(("t", i), out)) for i in range(8, 12)]
    hooks.append(base.encoder.layer[11].layout_output.register_forward_hook(
        lambda _m, _i, out: taps.__setitem__(("l", 11), out)))
    wins = windows_by_tokens([len(p) for p in per_word])
    try:
        for a, b in wins:
            tok_idx = [t for w in range(a, b) for t in per_word[w]]
            input_ids = [tok.bos_token_id] + [ids[t] for t in tok_idx] + [tok.eos_token_id]
            bbox = [[0, 0, 0, 0]] + [list(boxes[wid[t]]) for t in tok_idx] + [[0, 0, 0, 0]]
            base(input_ids=torch.tensor([input_ids]), bbox=torch.tensor([bbox]),
                 attention_mask=torch.ones(1, len(input_ids), dtype=torch.long))
            pos = 1                                     # 跳过 <s>
            for w in range(a, b):
                text4[w] = torch.stack([taps[("t", i)][0, pos] for i in range(8, 12)]).numpy()
                layout[w] = taps[("l", 11)][0, pos].numpy()
                pos += len(per_word[w])
    finally:
        for h in hooks:
            h.remove()
    return text4, layout, len(ids), len(wins)


def extract_features(data):
    """所有文档跑一遍冻结 LiLT，结果存到 CACHE/features.pkl（每份文档的前向耗时也存下来）。已有缓存就直接读。"""
    path = os.path.join(CACHE, "features.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            feats = pickle.load(f)
        for split, docs in data.items():
            for d in docs:
                d.update(feats[d["id"]])
        return
    model, tok = load_backbone()
    feats = {}
    for split, docs in data.items():
        for d in docs:
            t1 = time.perf_counter()
            text4, layout, n_tok, n_win = encode_doc(model, tok, d["words"], d["boxes"])
            feats[d["id"]] = {"text4": text4, "layout": layout, "n_tok": n_tok, "n_win": n_win,
                              "enc_s": time.perf_counter() - t1, "enc_threads": torch.get_num_threads()}
            d.update(feats[d["id"]])
    with open(path, "wb") as f:
        pickle.dump(feats, f)


def word_feats(doc, view="both"):
    """按输出契约取特征。返回 float32 [W, D]"""
    t = doc["text4"].astype(np.float32)
    lay = doc["layout"].astype(np.float32)
    if view == "text":
        return t[:, 3]                                  # 文本流最后一层 = HF 的 last_hidden_state
    if view == "layout":
        return lay
    if view == "both":
        return np.concatenate([t[:, 3], lay], 1)        # 960：官方 LiLT 任务头的做法
    if view == "last4":
        return np.concatenate([t.reshape(len(t), -1), lay], 1)
    raise ValueError(view)


# ----------------------------------------------------------------------------
# 批处理、训练、评测
# ----------------------------------------------------------------------------
def collate(docs, view):
    n = max(len(d["words"]) for d in docs)
    D = VIEWS[view]
    x = torch.zeros(len(docs), n, D)
    y = torch.full((len(docs), n), -100, dtype=torch.long)
    m = torch.zeros(len(docs), n, dtype=torch.bool)
    bx = torch.zeros(len(docs), n, 4)
    for i, d in enumerate(docs):
        k = len(d["words"])
        x[i, :k] = torch.from_numpy(word_feats(d, view))
        y[i, :k] = torch.from_numpy(d["labels"])
        m[i, :k] = True
        bx[i, :k] = torch.from_numpy(d["boxes"]).float()
    return x, m, y, bx


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def file_bytes(module):
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.tell()


@torch.no_grad()
def predict_docs(head, docs, view, bs=8):
    """返回每份文档的 (pred [W], conf [W])"""
    head.eval()
    out = []
    for i in range(0, len(docs), bs):
        batch = docs[i:i + bs]
        x, m, _, bx = collate(batch, view)
        pred, conf = head.predict(x, m, bx)
        for j, d in enumerate(batch):
            k = len(d["words"])
            out.append((pred[j, :k].numpy(), conf[j, :k].numpy()))
    return out


def evaluate(docs, preds):
    """
    逐词指标（FUNSD 官方报的是实体级 F1，这里不是同一个口径）：
      f1[c]      每类的 P/R/F1
      macro_f1   三个字段类（question/answer/header）F1 的平均
      micro_f1   把三个字段类的词合在一起算的 F1（other 不算）
      ece        置信度校准误差（10 个等宽区间）
      ent_full   金标实体里“所有词都标对”的比例（不判断实体边界，不是 exact-field match）
    """
    y = np.concatenate([d["labels"] for d in docs])
    p = np.concatenate([pr for pr, _ in preds])
    c = np.concatenate([cf for _, cf in preds])
    res = {"per_class": {}}
    tp_all = fp_all = fn_all = 0
    for k, name in enumerate(LABELS):
        tp = int(((p == k) & (y == k)).sum())
        fp = int(((p == k) & (y != k)).sum())
        fn = int(((p != k) & (y == k)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        res["per_class"][name] = {"p": prec, "r": rec, "f1": f1, "n": int((y == k).sum())}
        if k in FIELD_IDS:
            tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
    res["macro_f1"] = float(np.mean([res["per_class"][LABELS[k]]["f1"] for k in FIELD_IDS]))
    res["micro_f1"] = 2 * tp_all / max(1, 2 * tp_all + fp_all + fn_all)
    res["acc"] = float((p == y).mean())
    res["ece"] = ece(c, p == y)
    full = tot = 0
    for d, (pr, _) in zip(docs, preds):
        for e in np.unique(d["ent"]):
            idx = d["ent"] == e
            lab = d["labels"][idx][0]
            if lab == 0:
                continue
            tot += 1
            full += int((pr[idx] == lab).all())
    res["ent_full"] = full / max(1, tot)
    return res


def reliability(conf, correct, edges=(0, 0.5, 0.7, 0.9, 0.97, 1.0)):
    """按置信度分段：每段有多少词、平均置信度、实际准确率"""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        idx = (conf > a) & (conf <= b)
        if idx.any():
            rows.append({"lo": a, "hi": b, "n": int(idx.sum()), "conf": float(conf[idx].mean()),
                         "acc": float(correct[idx].mean())})
    return rows


def flat_preds(docs, preds):
    y = np.concatenate([d["labels"] for d in docs])
    p = np.concatenate([pr for pr, _ in preds])
    c = np.concatenate([cf for _, cf in preds])
    return y, p, c


def print_latency(name, lat, width=26):
    cells = "  ".join(f"n={n}: {v[0]:6.2f}" for n, v in lat.items())
    print(f"  {name:<{width}s} {cells}   (p50, ms)")


def ece(conf, correct, bins=10):
    """Expected Calibration Error：按置信度分 10 个区间，|平均置信度 - 准确率| 按样本数加权"""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        idx = (conf > a) & (conf <= b)
        if idx.any():
            total += idx.mean() * abs(conf[idx].mean() - correct[idx].mean())
    return float(total)


def train_head(make_head, data, view="both", seed=0, lr=2e-3, wd=0.01, bs=8, max_epochs=60, patience=10):
    """
    统一训练预算：AdamW(lr=2e-3, wd=0.01)，每批 8 份文档，最多 60 轮；
    每轮在 dev 上算宏平均 F1，连续 10 轮不涨就停，取 dev 最好的那一轮。
    """
    set_seed(seed)
    head = make_head(VIEWS[view], len(LABELS))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    best, best_state, best_ep, t0 = -1.0, None, 0, time.perf_counter()
    train = list(data["train"])
    for ep in range(max_epochs):
        head.train()
        random.shuffle(train)
        for i in range(0, len(train), bs):
            x, m, y, bx = collate(train[i:i + bs], view)
            loss = head.loss(x, m, y, bx)
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = evaluate(data["dev"], predict_docs(head, data["dev"], view))["macro_f1"]
        if f1 > best:
            best, best_ep = f1, ep
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_epoch": best_ep + 1, "dev_macro_f1": best, "train_s": time.perf_counter() - t0}


def run_seeds(name, make_head, data, view="both", **kw):
    """3 个随机种子各训练一次，在测试集上评测；返回汇总和最后一个 head"""
    runs = []
    for seed in SEEDS:
        head, info = train_head(make_head, data, view, seed, **kw)
        r = evaluate(data["test"], predict_docs(head, data["test"], view))
        r.update(info)
        runs.append(r)
    summ = summarize(name, runs)
    summ["params"] = count_params(head)
    summ["file_bytes"] = file_bytes(head)
    summ["view"] = view
    return summ, head


def summarize(name, runs):
    def ms(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return {
        "name": name,
        "macro_f1": ms([r["macro_f1"] for r in runs]),
        "micro_f1": ms([r["micro_f1"] for r in runs]),
        "ent_full": ms([r["ent_full"] for r in runs]),
        "ece": ms([r["ece"] for r in runs]),
        "per_class_f1": {c: ms([r["per_class"][c]["f1"] for r in runs]) for c in LABELS},
        "epochs": [r["best_epoch"] for r in runs],
        "train_s": ms([r["train_s"] for r in runs]),
        "n_runs": len(runs),
    }


def print_summary(s, width=26):
    f = s["per_class_f1"]
    print(f"  {s['name']:<{width}s} 宏F1 {s['macro_f1']['mean']:.3f}±{s['macro_f1']['std']:.3f}  "
          f"Q {f['question']['mean']:.3f}  A {f['answer']['mean']:.3f}  H {f['header']['mean']:.3f}  "
          f"实体全对 {s['ent_full']['mean']:.3f}  ECE {s['ece']['mean']:.3f}")


# ----------------------------------------------------------------------------
# 计时与结果
# ----------------------------------------------------------------------------
def latency_ms(fn, repeat=60, warmup=8, threads=1):
    """单线程 CPU 计时，返回 (p50, p95) 毫秒。只测 head 本身，不含特征传输和前后处理。"""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                fn()
            ts = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(old)
    ts.sort()
    return ts[len(ts) // 2], ts[int(len(ts) * 0.95) - 1]


def fake_doc(n, D, seed=0):
    """计时用：n 个词的随机特征和随机框（只决定 shape，数值无关）"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, n, D, generator=g)
    m = torch.ones(1, n, dtype=torch.bool)
    xy = torch.rand(1, n, 2, generator=g) * 900
    bx = torch.cat([xy, xy + 40], -1)
    return x, m, bx


def head_latency(head, D, ns=(64, 128, 256, 512)):
    head.eval()
    out = {}
    for n in ns:
        x, m, bx = fake_doc(n, D)
        out[n] = latency_ms(lambda: head.predict(x, m, bx))
    return out


def save_result(name, obj):
    os.makedirs(os.path.join(CACHE, "results"), exist_ok=True)
    with open(os.path.join(CACHE, "results", f"{name}.json"), "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_result(name):
    with open(os.path.join(CACHE, "results", f"{name}.json")) as f:
        return json.load(f)


def cpu_name():
    import platform
    import subprocess
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or platform.machine()
