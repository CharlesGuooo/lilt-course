"""
第 3 章：Layout embedding
一个框 (x0, y0, x1, y1) 怎么变成一个 192 维向量。
"""
from common import *

model, real = load_model()
tok, _ = load_tokenizer()
LE = model.layout_embeddings

doc = get_words()
enc = encode(tok, doc["words"], [normalize_box(b) for b in doc["boxes_px"]])
taps = run_with_taps(model, enc)
bbox, pos = enc["bbox"], taps["position_ids"]

# ---------------------------------------------------------------------------
banner("1. 四张坐标表，六次查表")
for name in ["x_position_embeddings", "y_position_embeddings", "h_position_embeddings", "w_position_embeddings"]:
    w = getattr(LE, name).weight
    print(f"  {name:<24s} {shape(w)}   （1024 个坐标值，每个 768/6 = 128 维）")
print("  x0 和 x1 查同一张 x 表，y0 和 y1 查同一张 y 表；宽和高各有自己的表。")
print("  所以 6 次查表只用了 4 张表：左、上、右、下、高、宽。")

# ---------------------------------------------------------------------------
banner("2. 手算一遍，和模型输出对比")
with torch.no_grad():
    parts = [
        LE.x_position_embeddings(bbox[..., 0]),                       # 左 x0
        LE.y_position_embeddings(bbox[..., 1]),                       # 上 y0
        LE.x_position_embeddings(bbox[..., 2]),                       # 右 x1
        LE.y_position_embeddings(bbox[..., 3]),                       # 下 y1
        LE.h_position_embeddings(bbox[..., 3] - bbox[..., 1]),        # 高
        LE.w_position_embeddings(bbox[..., 2] - bbox[..., 0]),        # 宽
    ]
    cat = torch.cat(parts, dim=-1)                                    # [B, L, 768]
    p2d = LE.box_linear_embeddings(cat)                               # [B, L, 192]
    p1d = LE.box_position_embeddings(pos)                             # [B, L, 192]
    manual = LE.LayerNorm(p2d + p1d)
print(f"  六段拼接            {shape(cat)}")
print(f"  线性投影 768→192    {shape(p2d)}   ← 论文公式 (2) 的 P2D")
print(f"  + 1D 位置 embedding {shape(p1d)}   ← layout 流自己的位置表，不是文本流那张")
print(f"  LayerNorm 后        {shape(manual)}   ← 论文公式 (3) 的 EL")
print(f"  和模型输出一致？ {torch.allclose(manual, taps['layout_emb'], atol=1e-5)}")

# ---------------------------------------------------------------------------
banner("3. 为什么要再加 1D 位置")
# 找同一个 segment 里的两个词（'Invoice' 和 'No:'）：它们的框完全相同
wid = enc["word_ids"]
first = {w: k for k, w in reversed(list(enumerate(wid))) if w is not None}
i, j = first[4], first[5]
print(f"  token {i} '{enc['tokens'][i]}' 框 {bbox[0, i].tolist()}")
print(f"  token {j} '{enc['tokens'][j]}' 框 {bbox[0, j].tolist()}")
same_p2d = torch.allclose(p2d[0, i], p2d[0, j])
print(f"  两者的 P2D 完全一样？ {same_p2d}")
print("  Transformer 对顺序不敏感，同框的词只能靠 1D 位置区分，所以 layout 流也要有 1D 位置。")

# ---------------------------------------------------------------------------
banner("4. 学到的坐标表是不是“平滑”的")
X = LE.x_position_embeddings.weight
ref = 300
for k in [300, 305, 320, 350, 400, 600, 900]:
    c = torch.nn.functional.cosine_similarity(X[ref], X[k], dim=0).item()
    print(f"  cos(x={ref}, x={k:<4d}) = {c:+.3f}")
print("  如果坐标表学到了“坐标是连续的”，相似度会随距离增大而下降。")
print("  这一点没有任何显式约束，只能从预训练数据里学出来；用真实权重看看实际是不是这样。")
if not real:
    print("  [随机权重下都接近 0，没有意义]")

# ---------------------------------------------------------------------------
banner("5. 参数量")
n = count_params(LE)
print(f"  layout embedding 总参数 {n / 1e3:.0f} K")
print(f"    4 张坐标表 4x1024x128 = {4 * 1024 * 128 / 1e3:.0f} K")
print(f"    1D 位置表  514x192    = {514 * 192 / 1e3:.0f} K")
print(f"    线性投影   768x192+192 = {(768 * 192 + 192) / 1e3:.0f} K")
print("  对比 LayoutLM：它把 2D 位置 embedding 直接加到 768 维文本 embedding 上；")
print("  LiLT 让坐标单独走一条 192 维的流，不碰文本 embedding。")
