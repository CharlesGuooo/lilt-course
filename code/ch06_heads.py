"""
第 6 章：输出与任务头
1) 原版仓库：文本 768 + layout 192 = 960 维进 head
2) HF 实现：只有文本 768 维进 head，layout 流最后一层输出被丢掉
3) SER（逐 token 分类）和 RE（关系抽取）的 head 长什么样
"""
from common import *
from transformers import LiltForTokenClassification

LABELS = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]

model, real = load_model()
tok, _ = load_tokenizer()
doc = get_words()
enc = encode(tok, doc["words"], [normalize_box(b) for b in doc["boxes_px"]])
taps = run_with_taps(model, enc)
text_out, layout_out = taps["text"][11], taps["layout"][11]

# ---------------------------------------------------------------------------
banner("1. 两条流的最终输出")
print(f"  文本流   {shape(text_out)}")
print(f"  layout流 {shape(layout_out)}   ← HF 的 LiltModel 不返回它，这里是用 hook 取出来的")
print(f"  HF last_hidden_state {shape(taps['last_hidden_state'])}  pooler_output 768 维（取 <s> 过 dense+tanh）")

# ---------------------------------------------------------------------------
banner("2. 原版 vs HF 的 token 分类头")
hf_tc, _ = load_model(LiltForTokenClassification, num_labels=len(LABELS))
print(f"  HF  LiltForTokenClassification.classifier = {hf_tc.classifier}")
orig_head = torch.nn.Linear(768 + 192, len(LABELS))
print(f"  原版 LiLTRobertaLikeForTokenClassification.classifier = {orig_head}")
print("  原版：logits = Linear(cat[文本, layout])；HF：logits = Linear(文本)")
print("  HF 版本里，layout 信息只能通过 12 层 BiACM 的注意力分数间接进入文本流。")
print("  用 HF 权重 fine-tune 的模型和论文报告的数字，严格说不是同一个结构。")

# ---------------------------------------------------------------------------
banner("3. SER：逐 token 分类 → 还原成词级标签 → 拼成实体")
feat = torch.cat([text_out, layout_out], dim=-1)          # [1, L, 960]
with torch.no_grad():
    logits = orig_head(feat)                              # [1, L, 7]（head 未训练，预测是随机的）
pred = logits.argmax(-1)[0].tolist()
print(f"  特征 {shape(feat)} → logits {shape(logits)}")

# 每个词取它第一个 subword 的预测（和训练时只给首个 subword 打标签对应）
word_pred, seen = {}, set()
for i, wid in enumerate(enc["word_ids"]):
    if wid is not None and wid not in seen:
        seen.add(wid)
        word_pred[wid] = LABELS[pred[i]]
print("  词级预测（未训练的 head，只看格式）：")
for wid in range(6):
    print(f"     {doc['words'][wid]:<14s} 预测 {word_pred[wid]:<11s} 真值 {doc['funsd'][wid]}")
print("  BIO 解码：B-X 开头、后面连续的 I-X 合并成一个实体（例如 'Invoice No:' → QUESTION）。")

# ---------------------------------------------------------------------------
banner("4. RE：判断两个实体之间是不是 key-value 关系（官方 re.py 的结构）")
D = 960
entity_emb = torch.nn.Embedding(3, D)                     # 实体类型：0 其他 / 1 question / 2 answer
proj = lambda: torch.nn.Sequential(torch.nn.Linear(2 * D, 768), torch.nn.ReLU(),
                                   torch.nn.Linear(768, 384), torch.nn.ReLU())
ffnn_head, ffnn_tail = proj(), proj()
bilinear = torch.nn.Bilinear(384, 384, 2, bias=False)
linear = torch.nn.Linear(2 * 384, 2)

# 用真值构造实体：(实体第一个 token 的下标, 类型)
first_tok = {}
for i, wid in enumerate(enc["word_ids"]):
    if wid is not None and wid not in first_tok:
        first_tok[wid] = i
entities = []
for wid, lab in enumerate(doc["funsd"]):
    if lab.startswith("B-QUESTION"):
        entities.append((first_tok[wid], 1, doc["words"][wid]))
    elif lab.startswith("B-ANSWER"):
        entities.append((first_tok[wid], 2, doc["words"][wid]))
pairs = [(h, t) for h in entities if h[1] == 1 for t in entities if t[1] == 2]
print(f"  question 实体 {sum(e[1] == 1 for e in entities)} 个，answer 实体 {sum(e[1] == 2 for e in entities)} 个"
      f" → 候选对 {len(pairs)} 个（所有 question × answer）")
with torch.no_grad():
    hi = torch.tensor([h[0] for h, _ in pairs]); ti = torch.tensor([t[0] for _, t in pairs])
    hrep = torch.cat([feat[0, hi], entity_emb(torch.ones_like(hi))], -1)       # [P, 1920]
    trep = torch.cat([feat[0, ti], entity_emb(torch.full_like(ti, 2))], -1)    # [P, 1920]
    h, t = ffnn_head(hrep), ffnn_tail(trep)                                    # [P, 384]
    rel_logits = bilinear(h, t) + linear(torch.cat([h, t], -1))                # [P, 2]
print(f"  head 表示 {shape(hrep)} → MLP → {shape(h)}")
print(f"  biaffine 打分 → {shape(rel_logits)}   （是 / 不是 key-value 对）")
print("  例：", ", ".join(f"'{a[2]}'→'{b[2]}'" for a, b in pairs[:4]), "...")
print("  注意：实体只用它第一个 token 的特征代表。")
