"""
第 4 章：双流 encoder 与 BiACM
手算一整层 LiLT，验证“两条流共享同一张注意力图，但各自聚合自己的 value”。
"""
import math
from common import *

model, real = load_model()
tok, _ = load_tokenizer()
cfg = model.config

doc = get_words()
boxes = [normalize_box(b) for b in doc["boxes_px"]]
enc = encode(tok, doc["words"], boxes)
taps = run_with_taps(model, enc)

K = 5  # 看第 K 层（0 起数）
layer = model.encoder.layer[K]
t_in = taps["text_emb"] if K == 0 else taps["text"][K - 1]        # [1, L, 768]
l_in = taps["layout_emb"] if K == 0 else taps["layout"][K - 1]    # [1, L, 192]
B, L, _ = t_in.shape


def split_heads(x, d):
    return x.view(B, L, cfg.num_attention_heads, d).permute(0, 2, 1, 3)


def merge_heads(x):
    return x.permute(0, 2, 1, 3).reshape(B, L, -1)


def manual_layer(layer, t, l, use_layout_score=True, use_text_score=True):
    """手写一整层 LiLT。返回 (文本输出, layout 输出, 中间量 dict)"""
    sa = layer.attention.self
    dT = sa.attention_head_size                       # 64
    dL = dT // sa.channel_shrink_ratio                # 16
    qT, kT, vT = split_heads(sa.query(t), dT), split_heads(sa.key(t), dT), split_heads(sa.value(t), dT)
    qL, kL, vL = (split_heads(sa.layout_query(l), dL), split_heads(sa.layout_key(l), dL),
                  split_heads(sa.layout_value(l), dL))
    sT = qT @ kT.transpose(-1, -2) / math.sqrt(dT)    # 文本流的注意力分数 [B,12,L,L]
    sL = qL @ kL.transpose(-1, -2) / math.sqrt(dL)    # layout 流的注意力分数 [B,12,L,L]
    s = (sT if use_text_score else 0) + (sL if use_layout_score else 0)   # BiACM：两张分数相加
    p = s.softmax(-1)                                 # 两条流用的是同一个 p
    ctxT = merge_heads(p @ vT)                        # 用文本的 V 聚合 -> [B,L,768]
    ctxL = merge_heads(p @ vL)                        # 用 layout 的 V 聚合 -> [B,L,192]
    aT = layer.attention.output(ctxT, t)              # dense + 残差 + LN（文本）
    aL = layer.attention.layout_output(ctxL, l)       # dense + 残差 + LN（layout，192 维的一套）
    oT = layer.output(layer.intermediate(aT), aT)     # FFN 768→3072→768
    oL = layer.layout_output(layer.layout_intermediate(aL), aL)   # FFN 192→768→192
    return oT, oL, dict(qT=qT, qL=qL, sT=sT, sL=sL, p=p)


# ---------------------------------------------------------------------------
banner(f"1. 手算第 {K} 层，和模型实际输出对比")
with torch.no_grad():
    oT, oL, mid = manual_layer(layer, t_in, l_in)
print(f"  文本 Q   {shape(mid['qT'])}   layout Q {shape(mid['qL'])}")
print(f"  文本分数 {shape(mid['sT'])}   layout 分数 {shape(mid['sL'])}  ← 头数一样，所以能相加")
print(f"  文本输出 一致？ {torch.allclose(oT, taps['text'][K], atol=1e-4)}")
print(f"  layout输出 一致？ {torch.allclose(oL, taps['layout'][K], atol=1e-4)}")

# ---------------------------------------------------------------------------
banner("2. 源码里的两个 softmax 其实是同一个")
sT, sL = mid["sT"], mid["sL"]
a = sT + sL     # 源码: attention_scores = tmp_attention_scores + tmp_layout_attention_scores
b = sL + sT     # 源码: layout_attention_scores = tmp_layout_attention_scores + tmp_attention_scores
print(f"  逐元素完全相等？ {torch.equal(a, b)}")
print("  加法可交换，mask 也一样 → 推理时（无 dropout）两条流的注意力概率完全相同。")
print("  所以 BiACM（fine-tune 阶段）= 共享一张注意力图，各用各的 V。")
print("  部署时可以只算一次 softmax，每层省一个 [B,12,L,L] 的 softmax 和缓冲区。")

# ---------------------------------------------------------------------------
banner("3. 每一层里，layout 分数占多大分量")
print(f"  {'层':>3s}  {'std(文本分数)':>14s}  {'std(layout分数)':>16s}  {'比值':>6s}")
with torch.no_grad():
    for k in range(cfg.num_hidden_layers):
        ti = taps["text_emb"] if k == 0 else taps["text"][k - 1]
        li = taps["layout_emb"] if k == 0 else taps["layout"][k - 1]
        _, _, m = manual_layer(model.encoder.layer[k], ti, li)
        st, sl = m["sT"].std().item(), m["sL"].std().item()
        print(f"  {k:>3d}  {st:>14.3f}  {sl:>16.3f}  {sl / st:>6.2f}")
print("  比值越大，这一层的注意力越多由版面决定。")

# ---------------------------------------------------------------------------
banner("4. 一个具体的 token：有没有 layout 分数，它看向哪里")
word_ids = enc["word_ids"]
target_word = len(doc["words"]) - 1                    # 最后一个词 '$1,250.00'（总金额）
ti = word_ids.index(target_word)
with torch.no_grad():
    _, _, m_text_only = manual_layer(layer, t_in, l_in, use_layout_score=False)
p_both = mid["p"][0, :, ti].mean(0)                    # 12 个头平均
p_text = m_text_only["p"][0, :, ti].mean(0)


def top(p, n=5):
    v, idx = p.topk(n)
    return ", ".join(f"{enc['tokens'][i]}({x:.2f})" for x, i in zip(v.tolist(), idx.tolist()))


print(f"  查询 token：'{enc['tokens'][ti]}'（{doc['words'][target_word]}）")
print(f"  只用文本分数   → {top(p_text)}")
print(f"  文本 + layout → {top(p_both)}")
print("  用真实权重时，看 layout 分数会不会把注意力拉向同一行的 'Total Amount:'。")

# ---------------------------------------------------------------------------
banner("5. 消融：把框打乱 / 把框清零，文本流最后一层变多少")
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(boxes), generator=g).tolist()
variants = {
    "原始框": boxes,
    "框随机打乱": [boxes[i] for i in perm],
    "全部清零": [[0, 0, 0, 0]] * len(boxes),
}
base_out = taps["last_hidden_state"]
for name, bx in variants.items():
    e = encode(tok, doc["words"], bx)
    with torch.no_grad():
        out = model(**model_inputs(e)).last_hidden_state
    cos = torch.nn.functional.cosine_similarity(out, base_out, dim=-1)[0]
    print(f"  {name:<8s}: 与原始输出的平均余弦相似度 {cos.mean():.3f}")
print("  文本和 token 完全相同，只改了框；差异全部通过 BiACM 的分数相加传进文本流。")
if not real:
    print("  [随机权重下 layout 分数很小，差异接近 0；用真实权重再看]")
