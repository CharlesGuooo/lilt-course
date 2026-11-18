"""
第 7 章：落地 —— 冻结 LiLT 当共享 backbone，每个任务只训练一个轻量 head
1) 一个“特征抽取器”：words + boxes -> 每个词一个向量（可选：哪条流、哪几层、怎么池化）
2) 冻结 backbone + 线性 head 的训练流程（玩具数据，只看流程）
3) CPU 上 backbone 和 head 各花多少时间；60 个 head 共享一次 backbone 前向
4) 缓存特征要多少存储
"""
import time
from common import *

model, real = load_model()
tok, tok_real = load_tokenizer()
for p in model.parameters():
    p.requires_grad_(False)          # 冻结：backbone 一个参数都不训练


class FrozenLiLTFeatures:
    """
    streams: "text" (768) / "layout" (192) / "both" (960)
    layers:  用哪几层的输出，取平均。默认只用最后一层 [11]；例如 [8, 9, 10, 11] 取后四层平均
    pooling: "first" 取每个词第一个 subword；"mean" 对一个词的所有 subword 取平均
    返回: word_feats [W, D]，以及 token 级特征和 word_ids（给需要 token 级输入的 head 用）
    """

    def __init__(self, model, tok, streams="both", layers=(11,), pooling="first"):
        self.model, self.tok = model, tok
        self.streams, self.layers, self.pooling = streams, list(layers), pooling

    def __call__(self, words, boxes_px, page_w=PAGE_W, page_h=PAGE_H):
        boxes = [normalize_box(b, page_w, page_h) for b in boxes_px]
        enc = encode(self.tok, words, boxes)
        taps = run_with_taps(self.model, enc)
        parts = []
        if self.streams in ("text", "both"):
            parts.append(torch.stack([taps["text"][k] for k in self.layers]).mean(0))
        if self.streams in ("layout", "both"):
            parts.append(torch.stack([taps["layout"][k] for k in self.layers]).mean(0))
        tok_feats = torch.cat(parts, -1)[0]                       # [L, D]
        wid = enc["word_ids"]
        W = len(words)
        out = torch.zeros(W, tok_feats.shape[-1])
        if self.pooling == "first":
            seen = set()
            for i, w in enumerate(wid):
                if w is not None and w not in seen:
                    seen.add(w)
                    out[w] = tok_feats[i]
        else:
            cnt = torch.zeros(W, 1)
            for i, w in enumerate(wid):
                if w is not None:
                    out[w] += tok_feats[i]
                    cnt[w] += 1
            out = out / cnt.clamp(min=1)
        # 被截断（超过 512 token）的词在 out 里会是全零：生产环境要分窗口处理
        n_covered = len({w for w in wid if w is not None})
        return {"word_feats": out, "token_feats": tok_feats, "word_ids": wid, "n_covered": n_covered}


# ---------------------------------------------------------------------------
banner("1. 特征抽取器的几种配置 → head 拿到的输入维度")
doc = get_words()
for cfg_ in [dict(streams="text"), dict(streams="layout"), dict(streams="both"),
             dict(streams="both", layers=(8, 9, 10, 11)), dict(streams="both", pooling="mean")]:
    fx = FrozenLiLTFeatures(model, tok, **cfg_)
    r = fx(doc["words"], doc["boxes_px"])
    print(f"  {str(cfg_):<52s} → word_feats {shape(r['word_feats'])}")
print("  只用 HF 的 last_hidden_state = streams='text'，丢掉了 192 维 layout 输出。")
print("  这些是“输出契约”要定下来的东西：backbone 服务到底吐出哪种。")

# ---------------------------------------------------------------------------
banner("2. 冻结 backbone + 线性 head：训练流程（玩具：一张文档，只演示步骤）")
FIELDS = ["Other", "invoice_no", "date", "name", "address", "total"]
fx = FrozenLiLTFeatures(model, tok, streams="both")
X = fx(doc["words"], doc["boxes_px"])["word_feats"]           # [W, 960]，算一次就可以反复用
y = torch.tensor([FIELDS.index(f) for f in doc["field"]])
head = torch.nn.Linear(X.shape[1], len(FIELDS))
opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
for step in range(200):
    loss = torch.nn.functional.cross_entropy(head(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    prob = head(X).softmax(-1)
    conf, pred = prob.max(-1)
print(f"  head 参数量：{count_params(head):,}（960 x {len(FIELDS)} + {len(FIELDS)}）")
for i in [5, 8, 9, 12, len(doc['words']) - 1]:
    print(f"     {doc['words'][i]:<14s} 预测 {FIELDS[pred[i]]:<10s} prob {conf[i]:.2f}   真值 {doc['field'][i]}")
print("  训练集就一张文档，这里一定会过拟合。只看流程：特征算一次，head 在特征上训练。")
print("  输出的 prob 是模型对字段的置信度，不是 OCR 的 conf，两者不能混。")

# ---------------------------------------------------------------------------
banner("3. CPU 计时：一次 backbone 前向 vs 60 个线性 head")
torch.set_num_threads(4)
long_words = (doc["words"] * 20)[:300]
long_boxes = (doc["boxes_px"] * 20)[:300]
enc = encode(tok, long_words, [normalize_box(b) for b in long_boxes])
L = enc["input_ids"].shape[1]
with torch.no_grad():
    model(**model_inputs(enc))                                   # 预热
    t0 = time.perf_counter()
    for _ in range(3):
        model(**model_inputs(enc))
    t_backbone = (time.perf_counter() - t0) / 3
    feats = torch.randn(len(long_words), 960)
    heads = [torch.nn.Linear(960, 8) for _ in range(60)]
    t0 = time.perf_counter()
    for h in heads:
        h(feats).softmax(-1)
    t_heads = time.perf_counter() - t0
print(f"  序列长度 {L}，4 线程")
print(f"  backbone 一次前向：{t_backbone * 1000:8.1f} ms")
print(f"  60 个线性 head 合计：{t_heads * 1000:8.2f} ms")
print("  → 成本几乎全在 backbone。共享一次 backbone 前向、按任务换 head，是这个方案省钱的根本原因。")

# ---------------------------------------------------------------------------
banner("4. 如果要缓存特征：每份文档多大")
for d, name in [(768, "text"), (960, "both")]:
    for dtype, nbytes in [("fp32", 4), ("fp16", 2)]:
        print(f"  512 token x {d} 维（{name}）{dtype}: {512 * d * nbytes / 1e6:.2f} MB")
