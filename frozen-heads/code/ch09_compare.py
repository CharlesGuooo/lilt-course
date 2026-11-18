"""
第 9 章：横向对比 —— 把前面各章的结果放进同一张表，再算多任务的总账
（先跑完 ch02 ~ ch08；这里只读它们存下的结果，不再训练）
"""
from common import *

ROWS = [  # 结果文件, 类别, 显示名
    ("linear_text", "线性", "linear · text 768"),
    ("linear_both", "线性", "linear · both 960"),
    ("mlp_both", "线性", "MLP 960→256"),
    ("conv1d", "序列", "Conv1D 感受野 7"),
    ("bilstm", "序列", "BiLSTM 2×128"),
    ("transformer", "序列", "Transformer 1 层"),
    ("crf", "解码", "linear + CRF"),
    ("layout2d", "版面", "Transformer + 2D 偏置"),
    ("span", "片段", "biaffine 片段"),
    ("tree", "树", "GBDT 964 维"),
    ("tree_pca", "树", "GBDT PCA64+4"),
    ("ncm", "原型", "NCM 类均值"),
    ("knn", "原型", "kNN k=10"),
    ("protoner", "原型", "ProtoNER 式"),
]
res = {k: load_result(k) for k, _, _ in ROWS}
tb = load_result("testbed")

# ---------------------------------------------------------------------------
banner("1. 同一张表（FUNSD 测试集 50 份，逐词指标；延迟 = 单线程、n=256 个词、只算 head）")
print(f"  {'head':<22s}{'宏F1':>13s}{'header':>8s}{'实体全对':>8s}{'ECE':>7s}{'参数/存量':>12s}{'文件':>10s}{'延迟':>9s}")
for k, fam, name in ROWS:
    s = res[k]
    m = s["macro_f1"]
    size = s["file_bytes"]
    size_s = f"{size / 1024:.0f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
    lat = s["latency"]["256"][0]
    print(f"  {name:<22s}{m['mean']:>8.3f}±{m['std']:.3f}{s['per_class_f1']['header']['mean']:>8.3f}"
          f"{s['ent_full']['mean']:>8.3f}{s['ece']['mean']:>7.3f}{s['params']:>12,d}{size_s:>10s}{lat:>7.2f}ms")
print("  ± 是 3 个随机种子的标准差；GBDT、NCM、kNN 没有随机性，只跑一次。")
print("  参数/存量：神经网络是参数个数；树是节点个数；NCM、kNN、ProtoNER 式是存下的浮点数个数。")

# ---------------------------------------------------------------------------
banner("2. 多任务的总账：一个共享 backbone + T 个任务 head")
from transformers import LiltModel
n_bb = count_params(LiltModel.from_pretrained(CKPT))
print(f"  backbone（LiLT-roberta-en-base）参数 {n_bb / 1e6:.1f}M → fp32 {n_bb * 4 / 1e6:.0f} MB · fp16 {n_bb * 2 / 1e6:.0f} MB")
T = 60
print(f"  T = {T} 个任务时，head 一共占多少（对照：每个任务各存一份微调过的 fp32 backbone = {T * n_bb * 4 / 1e9:.1f} GB）：")
for k, _, name in ROWS:
    tot = T * res[k]["file_bytes"]
    print(f"    {name:<22s} {tot / 1e6:9.2f} MB   = backbone 的 {pct(tot / (n_bb * 4))}")
print("  kNN 存的是训练词本身；任务越多、数据越多，它涨得越快。")

# ---------------------------------------------------------------------------
banner("3. 一份文档的 CPU 时间：backbone 和 head 各占多少（单线程）")
bb = tb["backbone_ms"]["1"][0]
print(f"  backbone 512 token：{bb:.0f} ms（第 1 章实测，{tb['cpu']}）")
for k, _, name in ROWS:
    h = res[k]["latency"]["256"][0]
    print(f"    + {name:<22s} {h:7.2f} ms   head 占 {pct(h / (bb + h))}")
save_result("compare", {"rows": [[k, fam, name] for k, fam, name in ROWS], "backbone_params": n_bb,
                        "backbone_ms_1t": bb, "T": T})
