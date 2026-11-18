"""
第 5 章：预训练
1) 权重是怎么拼起来的（为什么能换文本模型）
2) DETACH 到底挡住了什么梯度
3) 三个预训练任务的标签长什么样（MVLM / KPL / CAI）
4) 文本流学习率放慢 1000 倍怎么写
"""
import math
import random
from common import *

model, real = load_model()
tok, _ = load_tokenizer()
cfg = model.config

# ---------------------------------------------------------------------------
banner("1. 一个 LiLT checkpoint = RoBERTa 权重 + lilt-only 权重")
sd = model.state_dict()
text_keys = [k for k in sd if not is_layout_param(k)]
layout_keys = [k for k in sd if is_layout_param(k)]
print(f"  文本流参数张量 {len(text_keys)} 个，例如：")
for k in text_keys[:3] + ["encoder.layer.0.attention.self.query.weight"]:
    print(f"     {k}")
print(f"  layout 流参数张量 {len(layout_keys)} 个，例如：")
for k in ["layout_embeddings.x_position_embeddings.weight",
          "encoder.layer.0.attention.self.layout_query.weight",
          "encoder.layer.0.layout_intermediate.dense.weight"]:
    print(f"     {k}")
print("\n  官方仓库的 gen_weight_roberta_like.py 做的就是：")
print("    text = torch.load(任意 RoBERTa)，把键名 'roberta.' 改成 'lilt.'")
print("    total = {**text, **torch.load(lilt-only-base)}   # lilt-only-base 只有 21MB")

# 演示：把文本流换成另一个同尺寸的 RoBERTa（这里用随机权重代替），layout 流原样保留
before = {k: sd[k].clone() for k in layout_keys}
fake_other_roberta = {k: torch.randn_like(sd[k]) * 0.02 for k in text_keys if sd[k].is_floating_point()}
model.load_state_dict(fake_other_roberta, strict=False)
unchanged = all(torch.equal(before[k], model.state_dict()[k]) for k in layout_keys)
print(f"\n  换掉文本流以后，layout 流参数一个没变？ {unchanged}")
print("  前提：新文本模型的层数、头数、最大长度要和 LiLT 一样（12 层、12 头、512），否则 BiACM 对不上。")
model, real = load_model()   # 恢复原始权重

# ---------------------------------------------------------------------------
banner("2. DETACH：预训练时挡住“layout 损失 → 文本流”的梯度")
doc = get_words()
enc = encode(tok, doc["words"], [normalize_box(b) for b in doc["boxes_px"]])
layer = model.encoder.layer[0]
sa = layer.attention.self
B, L = enc["input_ids"].shape
H = cfg.num_attention_heads


def sh(x, d):
    return x.view(B, L, H, d).permute(0, 2, 1, 3)


def layout_branch_loss(detach):
    model.zero_grad()
    t, pos = model.embeddings(input_ids=enc["input_ids"])
    l = model.layout_embeddings(bbox=enc["bbox"], position_ids=pos)
    sT = sh(sa.query(t), 64) @ sh(sa.key(t), 64).transpose(-1, -2) / math.sqrt(64)
    sL = sh(sa.layout_query(l), 16) @ sh(sa.layout_key(l), 16).transpose(-1, -2) / math.sqrt(16)
    s_for_layout = sL + (sT.detach() if detach else sT)   # 论文公式 (6)
    p = s_for_layout.softmax(-1)
    layout_ctx = p @ sh(sa.layout_value(l), 16)
    loss = layout_ctx.pow(2).mean()                        # 随便一个只依赖 layout 分支的损失
    loss.backward()
    gnorm = lambda w: 0.0 if w.grad is None else w.grad.norm().item()   # 没有梯度流过时 grad 是 None
    return gnorm(sa.query.weight), gnorm(sa.layout_query.weight)


for detach in (False, True):
    g_text, g_lay = layout_branch_loss(detach)
    print(f"  detach={str(detach):<5s}  文本 W_Q 的梯度范数 {g_text:.2e}   layout W_Q 的梯度范数 {g_lay:.2e}")
print("  detach 以后，layout 分支的损失完全碰不到文本流的参数。")
print("  目的：让 layout 流适应“任何一个 RoBERTa”，而不是和预训练时那一个文本流绑死。")
print("  fine-tune 时去掉 detach，两条流一起训（论文表 1(a) #3 vs #5）。")

# ---------------------------------------------------------------------------
banner("3. 三个预训练任务的标签")
random.seed(0)
ids, word_ids, tokens = tokenize_words(tok, doc["words"])
boxes = enc["bbox"][0].tolist()
n = len(ids)
cand = [i for i in range(n) if word_ids[i] is not None]

# MVLM：15% 的 token；其中 80% 换 <mask>，10% 换随机词，10% 不变；layout 不动
mvlm = random.sample(cand, max(1, int(0.15 * len(cand))))
print("  [MVLM] 被选中的 token 和处理方式：")
for i in mvlm[:5]:
    r = random.random()
    how = "→ <mask>" if r < 0.8 else ("→ 随机词" if r < 0.9 else "→ 保持不变")
    print(f"     {tokens[i]:<10s} {how}   目标：在整个词表上预测原 token")

# KPL：页面切成 7x7=49 个区域；15% 的框被遮住；预测框的三个关键点各落在哪个区域
def region(x, y, n=7):
    col = min(n - 1, x * n // 1001)
    row = min(n - 1, y * n // 1001)
    return row * n + col


kpl = random.sample(cand, max(1, int(0.15 * len(cand))))
print("\n  [KPL] 被遮住的框 → 三个分类目标（49 类）：")
for i in kpl[:4]:
    x0, y0, x1, y1 = boxes[i]
    tl, br, ct = region(x0, y0), region(x1, y1), region((x0 + x1) // 2, (y0 + y1) // 2)
    print(f"     {tokens[i]:<10s} 框 {boxes[i]} → 左上 {tl:2d}  右下 {br:2d}  中心 {ct:2d}")
print("     输入里这些框 80% 换成 (0,0,0,0)，10% 换成同 batch 的随机框，10% 不变。")
print("     预测区域而不是精确坐标：OCR 框本来就有误差，放宽要求反而效果更好。")

# CAI：被 MVLM/KPL 替换过的 token-box 对 = 没对齐(0)；保持不变的 = 对齐(1)
print("\n  [CAI] 每个被处理过的 token-box 对，二分类：文字和框还对得上吗？")
print("     被换成随机词 / 随机框的 → 0（mis-aligned）；保持原样的 → 1（aligned）")
print("     这逼模型学会“这个词应该出现在这个位置吗”，也就是跨模态的一致性。")

# ---------------------------------------------------------------------------
banner("4. 文本流学习率放慢 1000 倍（论文 2.3 节）")
print("""  base_lr = 2e-5
  text_params   = [p for n, p in model.named_parameters() if not is_layout_param(n)]
  layout_params = [p for n, p in model.named_parameters() if is_layout_param(n)]
  optim = torch.optim.AdamW([
      {"params": layout_params, "lr": base_lr},
      {"params": text_params,   "lr": base_lr / 1000},
  ], weight_decay=1e-2)
  论文表 1(c)：不放慢 0.7840，放慢 1000 倍 0.7963（最好），完全冻结 0.7893。""")
