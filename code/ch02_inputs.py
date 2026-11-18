"""
第 2 章：输入准备
OCR 结果（词 + 像素框）→ 归一化坐标 → subword 对齐 → 模型的四个输入张量。
"""
from common import *

tok, _ = load_tokenizer()

# ---------------------------------------------------------------------------
banner("1. OCR 的原始输出：segment（一行/一块文字）+ 像素框")
for text, box, lab, field in SAMPLE_DOC[:6]:
    print(f"  {text:<26s} {str(box):<22s} {lab}")
print(f"  ...（共 {len(SAMPLE_DOC)} 个 segment，页面 {PAGE_W}x{PAGE_H} 像素）")

# ---------------------------------------------------------------------------
banner("2. 坐标归一化到 0..1000：和页面分辨率无关")
for px in [(60, 140, 170, 160), (660, 460, 760, 480)]:
    print(f"  像素 {px}  →  {normalize_box(px)}")
print("  同一张发票扫成 300dpi 或 72dpi，归一化后的坐标几乎一样。")
print("  模型的坐标 embedding 表只有 1024 行（max_2d_position_embeddings），超过就会 IndexError。")

# ---------------------------------------------------------------------------
banner("3. 词级框 vs segment 级框")
seg = get_words("segment")
wrd = get_words("word")
for i in [4, 5, 8, 9, 10]:   # Invoice No: / March 15, 2024
    print(f"  {seg['words'][i]:<12s} segment框 {normalize_box(seg['boxes_px'][i])}   "
          f"词框 {normalize_box(wrd['boxes_px'][i])}")
print("  论文和官方 FUNSD 代码都用 segment 框：一行里所有词共享同一个框。")
print("  原因（论文 2.1.2 节）：很多 OCR 引擎不给细粒度的词框。")
print("  后果：同一行内的词，layout 流只能靠 1D 位置区分；词框则能多给一点行内位置信息。")

# ---------------------------------------------------------------------------
banner("4. subword 对齐：每个 token 继承它所属词的框和标签")
boxes = [normalize_box(b) for b in seg["boxes_px"]]
enc = encode(tok, seg["words"], boxes)
print(f"  {'idx':>3s}  {'token':<12s} {'word_id':>7s}  {'bbox':<22s} label(只给首个 subword)")
prev = None
for i, (t, wid) in enumerate(zip(enc["tokens"], enc["word_ids"])):
    if i > 16:
        print("  ...")
        break
    if wid is None:
        lab = "-100（特殊 token，不算 loss）"
    elif wid != prev:
        lab = seg["funsd"][wid]
    else:
        lab = "-100（非首个 subword）"
    prev = wid
    print(f"  {i:>3d}  {t:<12s} {str(wid):>7s}  {str(enc['bbox'][0, i].tolist()):<22s} {lab}")

# ---------------------------------------------------------------------------
banner("5. 特殊 token 的框：论文和代码不一样")
enc_paper = encode(tok, seg["words"], boxes, special_box="paper")
print(f"  官方 FUNSD 代码          ：<s> {enc['bbox'][0, 0].tolist()}   </s> {enc['bbox'][0, -1].tolist()}")
print(f"  论文 2.1.2 节写法        ：<s> {enc_paper['bbox'][0, 0].tolist()}   </s> {enc_paper['bbox'][0, -1].tolist()}")
print("  预训练用的是哪种，从公开代码里看不出来。推理时跟训练时保持一致就行。")

# ---------------------------------------------------------------------------
banner("6. 最终送进模型的张量")
for k in ("input_ids", "bbox", "attention_mask"):
    print(f"  {k:<15s} {shape(enc[k]):<10s} {enc[k].dtype}")
print("  position_ids 不需要传：模型用 input_ids 自己算，然后两条流共用。")
print("  word_ids 不进模型，但推理后要用它把 token 级预测映射回词。")
print("\n  HF 官方示例用 tokenizer(words, boxes=boxes)，它在内部做的就是第 4 步的对齐。")
