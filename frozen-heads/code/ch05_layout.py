"""
第 5 章：版面感知 head —— 在小 Transformer 上加二维相对位置偏置
1) 同一个 1 层 Transformer head，有 / 无二维相对位置偏置（3 个种子）
2) 偏置学到了什么：横向、纵向距离分桶后的平均偏置
3) 成本：多出来的参数、要多传的输入（词框）、延迟
4) 网格化方案（BERTgrid 一类）的张量有多大：按公式估算
"""
from common import *
from heads import Layout2DHead, TransformerHead, rel_bucket

data = load_funsd()
extract_features(data)

# ---------------------------------------------------------------------------
banner("1. Transformer head：有 / 无二维相对位置偏置（both 960，3 个种子）")
try:
    tr = load_result("transformer")
except FileNotFoundError:
    tr, _ = run_seeds("Transformer 1 层 d256", TransformerHead, data, "both")
tr["name"] = "Transformer（无 2D 偏置）"
print_summary(tr)
lay_s, lay = run_seeds("Transformer + 2D 偏置", Layout2DHead, data, "both")
print_summary(lay_s)
d = {c: lay_s["per_class_f1"][c]["mean"] - tr["per_class_f1"][c]["mean"] for c in LABELS}
print("  加偏置后逐类 F1 变化：" + "  ".join(f"{c} {v:+.3f}" for c, v in d.items()))

# ---------------------------------------------------------------------------
banner("2. 偏置学到了什么（种子 2，4 个注意力头取平均）")
blk = lay.blocks[0]
side = blk.side
show = [-400, -100, -30, -10, 0, 10, 30, 100, 400]
for name, emb in [("横向 dx", blk.bx), ("纵向 dy", blk.by)]:
    vals = emb.weight.detach().mean(1)
    b = rel_bucket(torch.tensor(show, dtype=torch.float), side)
    print(f"  {name}：" + "  ".join(f"{s:+d}→{vals[i]:+.2f}" for s, i in zip(show, b.tolist())))
print("  坐标是 0..1000 的归一化单位；偏置为正 = 更愿意看那个方向、那个距离上的词。")

# ---------------------------------------------------------------------------
banner("3. 成本：参数、输入、单线程延迟（batch=1）")
tr_head = TransformerHead(960, 4)
extra = lay_s["params"] - count_params(tr_head)
print(f"  Transformer 参数 {count_params(tr_head):,}；加 2D 偏置后 {lay_s['params']:,}（多 {extra} 个：2 张 {2 * side - 1}×4 的表）")
print("  输入要多一样东西：每个词的框（4 个数）。冻结特征里虽然已经编码过位置，这里是显式再给一次。")
lat_tr = head_latency(tr_head, 960)
lat = head_latency(lay, 960)
print_latency("Transformer", lat_tr, 22)
print_latency("Transformer + 2D 偏置", lat, 22)

# ---------------------------------------------------------------------------
banner("4. 网格化方案的张量有多大（按公式估算，不是实测）")
sizes = [doc["size"] for doc in data["test"]]
w = int(statistics.median([s[0] for s in sizes]))
h = int(statistics.median([s[1] for s in sizes]))
print(f"  FUNSD 测试集图片中位尺寸 {w}×{h} 像素；把 768 维词向量铺满网格，fp32：")
for stride in [1, 4, 8]:
    gh, gw = math.ceil(h / stride), math.ceil(w / stride)
    print(f"    每 {stride} 像素一格 → {gh}×{gw}×768 = {gh * gw * 768 * 4 / 1e6:8.1f} MB / 页")
print(f"  对照：逐词特征（both 960，fp32）中位文档 {int(statistics.median([len(x['words']) for x in data['test']]))} 词 ≈ "
      f"{statistics.median([len(x['words']) for x in data['test']]) * 960 * 4 / 1e6:.2f} MB")

lay_s["latency"] = lat
lay_s["bias"] = {"dx": blk.bx.weight.detach().mean(1).tolist(), "dy": blk.by.weight.detach().mean(1).tolist(),
                 "side": side}
save_result("layout2d", lay_s)
