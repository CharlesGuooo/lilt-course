"""
第 3 章：序列上下文 —— 一维卷积、BiLSTM、小 Transformer
1) 三种序列 head 和线性基线放在同一张表里（同特征、同切分、同训练预算）
2) 一维“相邻”和二维“相邻”差多少：卷积的感受野够不够到配对的答案
3) 成本：参数量、文件大小、序列长度 n 变长时的单线程延迟
"""
from common import *
from heads import Conv1DHead, BiLSTMHead, TransformerHead, LinearHead

data = load_funsd()
extract_features(data)

# ---------------------------------------------------------------------------
banner("1. 序列 head vs 线性基线（both 960，3 个种子，测试集 50 份）")
try:
    base = load_result("linear_both")
except FileNotFoundError:
    base, _ = run_seeds("linear · both 960", LinearHead, data, "both")
base["name"] = "linear（基线）"
print_summary(base)
runs = {}
for key, name, make in [("conv1d", "Conv1D k3×2 d(1,2)", Conv1DHead),
                        ("bilstm", "BiLSTM 1 层 2×128", BiLSTMHead),
                        ("transformer", "Transformer 1 层 d256", TransformerHead)]:
    s, head = run_seeds(name, make, data, "both")
    runs[key] = (s, head)
    print_summary(s)
print("  训练轮数（dev 最好的那一轮）：" + "；".join(f"{k} {s['epochs']}" for k, (s, _) in runs.items()))

# ---------------------------------------------------------------------------
banner("2. 一维相邻 vs 二维相邻（测试集）")
rf = runs["conv1d"][1].receptive_field
half = rf // 2
hit2d = tot2d = 0
dist = []
for d in data["test"]:
    b = np.array(d["boxes_px"], float)
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    dd = np.linalg.norm(c[:, None] - c[None], axis=-1)
    np.fill_diagonal(dd, np.inf)
    nn6 = np.argsort(dd, 1)[:, :6]                       # 每个词在页面上最近的 6 个词
    pos = np.arange(len(c))
    hit2d += (np.abs(nn6 - pos[:, None]) <= half).sum()
    tot2d += nn6.size
    for q, a in d["links"]:                              # question -> answer 配对在阅读顺序里隔多远
        iq, ia = np.where(d["ent"] == q)[0], np.where(d["ent"] == a)[0]
        if len(iq) and len(ia):
            dist.append(int(np.abs(iq[:, None] - ia[None]).min()))
dist = np.array(dist)
print(f"  卷积感受野 {rf} 个词（中心词左右各 {half} 个）")
print(f"  每个词在页面上最近的 6 个词里，落在阅读顺序 ±{half} 以内的：{pct(hit2d / tot2d)}")
print(f"  FUNSD 标注的 question→answer 配对 {len(dist)} 对，两者在阅读顺序里的最近距离：")
for lo, hi, lab in [(1, 1, "紧挨着（=1）"), (2, half, f"2..{half}"), (half + 1, 10, f"{half + 1}..10"), (11, 10 ** 9, "> 10")]:
    k = ((dist >= lo) & (dist <= hi)).sum()
    print(f"    {lab:<12s} {k:4d} 对  {pct(k / len(dist))}")
print(f"  ±{half} 以内合计 {pct((dist <= half).mean())}：剩下的配对，卷积 head 自己看不到，只能靠 LiLT 特征里已经带的上下文。")

# ---------------------------------------------------------------------------
banner("3. 成本：参数量、文件大小、单线程延迟（batch=1，只算 head）")
lin = LinearHead(960, 4)
lat = {"linear": head_latency(lin, 960)}
print(f"  {'linear':<22s} 参数 {count_params(lin):>9,d}   文件 {file_bytes(lin) / 1024:8.1f} KB")
for key, (s, head) in runs.items():
    lat[key] = head_latency(head, 960)
    print(f"  {s['name']:<22s} 参数 {s['params']:>9,d}   文件 {s['file_bytes'] / 1024:8.1f} KB")
for key, l in lat.items():
    print_latency(key, l, 22)
r = {k: lat[k][512][0] / lat[k][64][0] for k in lat}
print("  n 从 64 增到 512（8 倍）时延迟涨了：" + "  ".join(f"{k} ×{v:.1f}" for k, v in r.items()))

for key, (s, _) in runs.items():
    s["latency"] = lat[key]
    save_result(key, s)
save_result("neighbors", {"rf": rf, "nn6_in_window": hit2d / tot2d, "link_dist": dist.tolist()})
