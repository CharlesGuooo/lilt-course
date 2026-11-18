"""
第 1 章：RoBERTa 复习
LiLT 的文本流就是一个原封不动的 RoBERTa-base。这一章只看文本流，把 RoBERTa 的每一步手算一遍。
"""
import math
from common import *
from transformers import RobertaConfig, RobertaModel

model, real = load_model()
tok, tok_real = load_tokenizer()
cfg = model.config

# ---------------------------------------------------------------------------
banner("1. BPE 分词：一个词会被切成几个 subword")
for w in ["Invoice", "INV-2024-0042", "$1,250.00", "Toronto", "Maple"]:
    ids, word_ids, tokens = tokenize_words(tok, [w])
    print(f"{w:>15s} → {tokens[1:-1]}")
print("\n'Ġ' 表示这个 subword 前面有空格，也就是一个新词的开头。")
print("特殊 token：<s>=0（句首，相当于 BERT 的 [CLS]），<pad>=1，</s>=2（句尾，相当于 [SEP]）")

# ---------------------------------------------------------------------------
banner("2. position_ids：RoBERTa 从 2 开始数，pad 位置固定为 1")
ids_a, _, _ = tokenize_words(tok, ["Total", "Amount:"])
ids_b, _, _ = tokenize_words(tok, ["Date:"])
L = max(len(ids_a), len(ids_b))
batch = torch.tensor([ids_a + [1] * (L - len(ids_a)), ids_b + [1] * (L - len(ids_b))])
mask = batch.ne(cfg.pad_token_id).int()
position_ids = torch.cumsum(mask, dim=1) * mask + cfg.pad_token_id
print("input_ids   :", batch.tolist())
print("position_ids:", position_ids.tolist())
print("原因：fairseq 的习惯，位置表的第 0、1 行留给 padding_idx 相关位置，所以 max_position_embeddings = 514 = 512 + 2")

# ---------------------------------------------------------------------------
banner("3. 文本 embedding = LayerNorm(词向量 + 位置向量 + token_type 向量)")
doc = get_words()
enc = encode(tok, doc["words"], [normalize_box(b) for b in doc["boxes_px"]])
taps = run_with_taps(model, enc)
E = model.embeddings
ids = enc["input_ids"]
pos = taps["position_ids"]
manual = E.word_embeddings(ids) + E.position_embeddings(pos) + E.token_type_embeddings(torch.zeros_like(ids))
manual = E.LayerNorm(manual)
print(f"手算 {shape(manual)}，和模型输出一致？ {torch.allclose(manual, taps['text_emb'], atol=1e-5)}")
print("注意：文本 embedding 里完全没有 bbox。坐标只进 layout 流。")

# ---------------------------------------------------------------------------
banner("4. 用 LiLT 的文本流权重拼出一个纯 RoBERTa")
rcfg = RobertaConfig(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
                     num_hidden_layers=cfg.num_hidden_layers, num_attention_heads=cfg.num_attention_heads,
                     intermediate_size=cfg.intermediate_size, max_position_embeddings=cfg.max_position_embeddings,
                     type_vocab_size=cfg.type_vocab_size, pad_token_id=cfg.pad_token_id,
                     layer_norm_eps=cfg.layer_norm_eps)
roberta = RobertaModel(rcfg).eval()
text_sd = {k: v for k, v in model.state_dict().items() if not is_layout_param(k)}
missing, unexpected = roberta.load_state_dict(text_sd, strict=False)
print(f"缺失的参数: {[m for m in missing if 'position_ids' not in m]}")
print(f"多出来的参数: {unexpected}")
print("→ 参数名一一对应：LiLT 的文本流就是 RoBERTa，一个参数都不多。")

# ---------------------------------------------------------------------------
banner("5. 手算 RoBERTa 第 0 层（Post-LN 结构）")
layer = roberta.encoder.layer[0]
x = taps["text_emb"]                                   # [1, L, 768]
B, L, D = x.shape
H = cfg.num_attention_heads
dh = D // H


def heads(t):  # [B,L,D] -> [B,H,L,dh]
    return t.view(B, L, H, dh).permute(0, 2, 1, 3)


with torch.no_grad():
    a = layer.attention
    q, k, v = heads(a.self.query(x)), heads(a.self.key(x)), heads(a.self.value(x))
    scores = q @ k.transpose(-1, -2) / math.sqrt(dh)   # [B,H,L,L]
    probs = scores.softmax(-1)
    ctx = (probs @ v).permute(0, 2, 1, 3).reshape(B, L, D)
    h1 = a.output.LayerNorm(a.output.dense(ctx) + x)                # ① attention + 残差 + LN
    ffn = layer.output.dense(layer.intermediate(h1))                 # ② 768→3072(GELU)→768
    h2 = layer.output.LayerNorm(ffn + h1)                            #    + 残差 + LN
    ref = layer(x)
    ref = ref[0] if isinstance(ref, tuple) else ref   # 不同 transformers 版本返回 tuple 或 tensor
print(f"Q/K/V      {shape(q)}   (batch, heads, seq, head_dim)")
print(f"scores     {shape(scores)}")
print(f"层输出      {shape(h2)}")
print(f"手算 == RobertaLayer 输出 ? {torch.allclose(h2, ref, atol=1e-4)}")

# ---------------------------------------------------------------------------
banner("6. 预告：同样的文本权重，加上 layout 流以后输出变了多少")
with torch.no_grad():
    r_out = roberta(input_ids=ids).last_hidden_state
lilt_out = taps["last_hidden_state"]
cos = torch.nn.functional.cosine_similarity(r_out, lilt_out, dim=-1)[0]
print(f"纯 RoBERTa vs LiLT 文本流（最后一层）逐 token 余弦相似度：平均 {cos.mean():.3f}，最低 {cos.min():.3f}")
print("两者用的是同一套文本权重，差别全部来自 BiACM 加进来的 layout 注意力分数（第 4 章）。")
if not real:
    print("[随机权重下这个数字没有意义；用真实权重运行再看]")
