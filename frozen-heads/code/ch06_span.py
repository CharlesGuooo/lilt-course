"""
第 6 章：片段 head —— biaffine 片段打分
1) 从扁平 tag 派生片段目标：片段有多长、正样本有多稀
2) 直接照搬（负样本权重 1、最长 32 词）的结果
3) 在 dev 上选负样本权重和最长片段，再用 3 个种子测
4) 片段级指标：对“派生片段”和对“金标实体”分别算 exact match；扁平 tag 的理论上限
5) 成本：参数量、候选片段数、单线程延迟
"""
from common import *
from heads import BiaffineSpanHead, LinearHead, flat_runs

data = load_funsd()
extract_features(data)


def make(L, w):
    return lambda D, K: BiaffineSpanHead(D, K, max_len=L, none_weight=w)


def n_cands(n, L):
    return sum(min(L, n - i) for i in range(n))          # 每个起点最多 L 个终点，不能越过文档末尾


# ---------------------------------------------------------------------------
banner("1. 从扁平 tag 派生片段：片段多长、正样本多稀")
for L in [32, 64]:
    for split in ["train", "test"]:
        n_field = n_long = 0
        for d in data[split]:
            for i, j, _ in flat_runs(d["labels"].tolist()):
                n_field += j - i + 1
                n_long += (j - i + 1) * (j - i + 1 > L)
        print(f"  最长 {L:2d} 词 · {split:<5s} 字段词 {n_field:6,d} 个，落在更长片段里的 {n_long:5,d} 个（{pct(n_long / n_field)}）")
n_pos = sum(len(flat_runs(d["labels"].tolist())) for d in data["train"])
n_cand = sum(n_cands(len(d["words"]), 32) for d in data["train"])
print(f"  训练集：最长 32 词时候选片段 {n_cand:,} 个，真片段 {n_pos:,} 个，正样本只占 {pct(n_pos / n_cand)}")

# ---------------------------------------------------------------------------
banner("2. 直接照搬：负样本权重 1、最长 32 词（3 个种子）")
try:
    base = load_result("linear_both")
except FileNotFoundError:
    base, _ = run_seeds("linear · both 960", LinearHead, data, "both")
base["name"] = "linear（基线）"
print_summary(base)
naive_s, _ = run_seeds("biaffine · w=1 L=32", make(32, 1.0), data, "both")
print_summary(naive_s)

# ---------------------------------------------------------------------------
banner("3. 在 dev 上选负样本权重 w 和最长片段 L（种子 0），再用 3 个种子测")
grid = {}
for L in [32, 64]:
    for w in [1.0, 0.3, 0.1, 0.03]:
        _, info = train_head(make(L, w), data, "both", seed=0)
        grid[(L, w)] = info["dev_macro_f1"]
        print(f"  L={L:2d}  w={w:<5}  dev 宏F1 {info['dev_macro_f1']:.3f}  （第 {info['best_epoch']} 轮最好）")
L_best, w_best = max(grid, key=grid.get)
print(f"  dev 上最好：L={L_best}, w={w_best}")
span_s, span = run_seeds(f"biaffine · w={w_best} L={L_best}", make(L_best, w_best), data, "both")
print_summary(base)
print_summary(naive_s)
print_summary(span_s)
print(f"  训练轮数：{span_s['epochs']}，每个种子训练 {span_s['train_s']['mean']:.0f} s")


# ---------------------------------------------------------------------------
def prf(pred, gold):
    tp = len(pred & gold)
    p, r = tp / max(1, len(pred)), tp / max(1, len(gold))
    return p, r, 2 * p * r / max(1e-9, p + r)


def entities(d):
    out = set()
    for e in np.unique(d["ent"]):
        idx = np.where(d["ent"] == e)[0]
        if d["labels"][idx[0]] != 0:
            out.add((tuple(idx.tolist()), int(d["labels"][idx[0]])))
    return out


def as_spans(runs):
    return {(tuple(range(i, j + 1)), c) for i, j, c, *_ in runs}


banner("4. 片段级 exact match（测试集，种子 2）")
lin, _ = train_head(LinearHead, data, "both", seed=2)
gold_run, gold_ent, pred_span, pred_lin = set(), set(), set(), set()
for k, d in enumerate(data["test"]):
    x, m, _, bx = collate([d], "both")
    tag = lambda s: {(k,) + t for t in s}
    gold_run |= tag(as_spans(flat_runs(d["labels"].tolist())))
    gold_ent |= tag(entities(d))
    pred_span |= tag(as_spans(span.decode(x, m)[0][0]))
    pred_lin |= tag(as_spans(flat_runs(lin.predict(x, m, bx)[0][0].tolist())))
rows = {}
for name, pred in [("biaffine 片段", pred_span), ("linear → 连续同标签合并", pred_lin), ("金标扁平 tag → 片段（上限）", gold_run)]:
    a = prf(pred, gold_run)
    b = prf(pred, gold_ent)
    rows[name] = {"vs_runs": a, "vs_entities": b}
    print(f"  {name:<24s} 对派生片段 P {a[0]:.3f} R {a[1]:.3f} F1 {a[2]:.3f}   对金标实体 P {b[0]:.3f} R {b[1]:.3f} F1 {b[2]:.3f}")
print(f"  金标实体 {len(gold_ent)} 个，派生片段 {len(gold_run)} 个。最后一行：标签全对，实体级 F1 也只有这么多——")
print("  扁平 tag 本身不带实例边界，这个上限对所有只学扁平 tag 的 head 都成立。")

# ---------------------------------------------------------------------------
banner("5. 成本：参数量、候选片段数、单线程延迟（batch=1）")
print(f"  参数 {span_s['params']:,}   文件 {span_s['file_bytes'] / 1024:.1f} KB   每个词作起点最多 {L_best} 个候选")
lat = head_latency(span, 960)
print_latency("biaffine 片段", lat, 22)
print_latency("linear", head_latency(LinearHead(960, 4), 960), 22)
for n in [64, 512]:
    print(f"  n={n}：候选片段 {n_cands(n, L_best):,} 个，每个都要算 4 类分数")

span_s["latency"] = lat
span_s["span_eval"] = rows
span_s["naive"] = {"macro_f1": naive_s["macro_f1"], "ece": naive_s["ece"]}
span_s["grid"] = [[L, w, f] for (L, w), f in grid.items()]
save_result("span", span_s)
