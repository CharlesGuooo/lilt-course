"""
第 2 章：基线 —— 逐词线性层与小 MLP
1) 同一个线性 head，换四种输出契约：text 768（HF）/ layout 192 / both 960（官方）/ last4 3264
2) 线性 vs 小 MLP（960 -> 256 -> 4）
3) 置信度：局部 softmax 的最大值靠不靠谱（分段准确率 + ECE）
4) 错在哪里：混淆矩阵
5) 参数量、文件大小、单线程 CPU 延迟
"""
from common import *
from heads import LinearHead, MLPHead

data = load_funsd()
extract_features(data)

# ---------------------------------------------------------------------------
banner("1. 同一个线性 head，四种输出契约（3 个种子，测试集 50 份）")
results = {}
for view in ["text", "layout", "both", "last4"]:
    s, head = run_seeds(f"linear · {view} {VIEWS[view]}", LinearHead, data, view)
    results[view] = (s, head)
    print_summary(s)
print("  Q/A/H = question/answer/header 的逐词 F1；宏F1 = 三者平均；实体全对 = 实体里每个词都标对的比例。")

# ---------------------------------------------------------------------------
banner("2. 线性 vs 小 MLP（都用 both 960）")
mlp_s, mlp = run_seeds("MLP · both 960→256→4", MLPHead, data, "both")
print_summary(results["both"][0])
print_summary(mlp_s)

# ---------------------------------------------------------------------------
banner("3. 置信度：softmax 最大值和实际准确率对得上吗（linear · both，种子 2）")
lin = results["both"][1]
y, p, c = flat_preds(data["test"], predict_docs(lin, data["test"], "both"))
rel = reliability(c, p == y)
for r in rel:
    print(f"  置信度 ({r['lo']:.2f}, {r['hi']:.2f}]  {r['n']:5d} 个词  平均置信度 {r['conf']:.3f}  实际准确率 {r['acc']:.3f}")
print(f"  ECE = {ece(c, p == y):.3f}（越接近 0 越可信）")

# ---------------------------------------------------------------------------
banner("4. 错在哪里：混淆矩阵（行 = 真值，列 = 预测；linear · both，种子 2）")
cm = np.zeros((4, 4), int)
for a, b in zip(y, p):
    cm[a, b] += 1
print("  " + " " * 10 + "".join(f"{n:>10s}" for n in LABELS))
for i, n in enumerate(LABELS):
    print(f"  {n:<10s}" + "".join(f"{v:10d}" for v in cm[i]))

# ---------------------------------------------------------------------------
banner("5. 成本：参数量、文件大小、单线程 CPU 延迟（batch=1，只算 head）")
lat = {}
for name, s, head, D in [("linear · text 768", results["text"][0], results["text"][1], 768),
                         ("linear · both 960", results["both"][0], results["both"][1], 960),
                         ("linear · last4 3264", results["last4"][0], results["last4"][1], 3264),
                         ("MLP · both", mlp_s, mlp, 960)]:
    lat[name] = head_latency(head, D)
    print(f"  {name:<22s} 参数 {s['params']:>9,d}   文件 {s['file_bytes'] / 1024:8.1f} KB")
for name, l in lat.items():
    print_latency(name, l, 22)

for view, (s, _) in results.items():
    save_result(f"linear_{view}", s)
mlp_s["latency"] = lat["MLP · both"]
results["both"][0]["latency"] = lat["linear · both 960"]
results["text"][0]["latency"] = lat["linear · text 768"]
save_result("linear_both", results["both"][0])
save_result("linear_text", results["text"][0])
save_result("mlp_both", mlp_s)
save_result("linear_reliability", {"rows": rel, "ece": ece(c, p == y), "cm": cm.tolist()})
