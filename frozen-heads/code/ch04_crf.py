"""
第 4 章：解码层 —— 线性发射 + CRF
1) 同一个线性发射分数，有 / 无 CRF（3 个种子）
2) CRF 学到的转移分数：扁平标签之间“谁后面跟谁”
3) 置信度：局部 softmax vs CRF 边际概率（分段准确率 + ECE）
4) 成本：多出来的参数和 Viterbi/前向-后向的延迟
"""
from common import *
from heads import CRFHead, LinearHead

data = load_funsd()
extract_features(data)

# ---------------------------------------------------------------------------
banner("1. 线性 vs 线性 + CRF（both 960，3 个种子）")
try:
    base = load_result("linear_both")
except FileNotFoundError:
    base, _ = run_seeds("linear · both 960", LinearHead, data, "both")
base["name"] = "linear（无 CRF）"
print_summary(base)
crf_s, crf = run_seeds("linear + CRF", CRFHead, data, "both")
print_summary(crf_s)
print(f"  训练轮数：{crf_s['epochs']}，每个种子训练 {crf_s['train_s']['mean']:.0f} s（无 CRF 的线性 {base['train_s']['mean']:.0f} s）")

# ---------------------------------------------------------------------------
banner("2. CRF 学到的转移分数 trans[前一个词, 当前词]（种子 2）")
T = crf.trans.detach().numpy()
print("  " + " " * 10 + "".join(f"{n:>10s}" for n in LABELS))
for i, n in enumerate(LABELS):
    print(f"  {n:<10s}" + "".join(f"{v:10.2f}" for v in T[i]))
print("  整行或整列加同一个常数，都会被发射分数的偏置和起止分数吸收，所以只看相对大小：")
print("  对角线比同行、同列的其他格子都高，意思是“倾向于连续同标签”。")

# ---------------------------------------------------------------------------
banner("3. 置信度：局部 softmax vs CRF 边际概率（测试集，种子 2）")
lin, _ = train_head(LinearHead, data, "both", seed=2)
rows = {}
for name, head in [("softmax（线性）", lin), ("CRF 边际", crf)]:
    y, p, c = flat_preds(data["test"], predict_docs(head, data["test"], "both"))
    rows[name] = {"rows": reliability(c, p == y), "ece": ece(c, p == y)}
    print(f"  {name}：ECE {rows[name]['ece']:.3f}")
    for r in rows[name]["rows"]:
        print(f"     ({r['lo']:.2f}, {r['hi']:.2f}]  {r['n']:5d} 个词  平均置信度 {r['conf']:.3f}  实际准确率 {r['acc']:.3f}")

# ---------------------------------------------------------------------------
banner("4. 成本：参数量与单线程延迟（batch=1）")
print(f"  linear + CRF 参数 {crf_s['params']:,}（比线性多 {4 * 4 + 4 + 4} 个：转移 4×4 + 起始 4 + 结束 4）")
lat = head_latency(crf, 960)
lat_lin = head_latency(LinearHead(960, 4), 960)
print_latency("linear", lat_lin, 22)
print_latency("linear + CRF", lat, 22)
print("  CRF 的预测 = Viterbi + 前向-后向（为了给每个词一个边际概率），两者都是沿序列的逐步循环。")

crf_s["latency"] = lat
crf_s["trans"] = T.tolist()
save_result("crf", crf_s)
save_result("calibration", rows)
