"""
LiLT 课程公共工具。每一章的脚本都 import 这里。

内容：
- load_model()      加载 HF 的 LiltModel（真实权重；连不上 HF 时退回随机初始化）
- load_tokenizer()  加载 RoBERTa 分词器（LiLT 的文本流就是 RoBERTa，所以词表相同）
- SAMPLE_DOC        一张手工构造的发票：每个 segment 有文字、像素框、FUNSD 风格标签和字段名
- encode()          words + boxes -> 模型输入（input_ids / bbox / attention_mask / word_ids）
- run_with_taps()   跑一次前向，用 hook 把两条流每一层的输出都取出来
                    （HF 的 LiltModel 只返回文本流，layout 流要靠 hook 拿）
"""
import hashlib
import torch

CKPT = "SCUT-DLVCLab/lilt-roberta-en-base"
TOKENIZER_CKPT = "roberta-base"  # LiLT-roberta-en-base 的文本流 = roberta-base，词表完全相同


# ----------------------------------------------------------------------------
# 打印工具
# ----------------------------------------------------------------------------
def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def shape(t):
    return "x".join(str(s) for s in t.shape)


# ----------------------------------------------------------------------------
# 模型与分词器
# ----------------------------------------------------------------------------
def load_model(cls=None, **kwargs):
    """返回 (model, is_real)。cls 默认 LiltModel，也可以传 LiltForTokenClassification。"""
    from transformers import LiltConfig, LiltModel

    cls = cls or LiltModel
    try:
        model = cls.from_pretrained(CKPT, **kwargs)
        real = True
    except Exception as e:  # 离线、被墙、没有缓存
        print(f"[提示] 无法下载 {CKPT}（{type(e).__name__}），改用随机初始化。"
              "shape 和流程一样，数值没有意义。")
        # 和真实 checkpoint 一样的 RoBERTa-base 尺寸（默认 LiltConfig 是 BERT 词表 30522，要改）
        cfg = LiltConfig(vocab_size=50265, max_position_embeddings=514, type_vocab_size=1,
                         pad_token_id=1, bos_token_id=0, eos_token_id=2, layer_norm_eps=1e-5, **kwargs)
        model = cls(cfg)
        real = False
    model.eval()
    return model, real


class ToyTokenizer:
    """离线时的替身：把单词按 4 个字符切块，模拟 BPE 的 subword 切分。只用于跑通流程。"""
    bos_token_id, pad_token_id, eos_token_id = 0, 1, 2

    def _ids(self, piece):
        h = int(hashlib.md5(piece.encode()).hexdigest(), 16)
        return 3 + h % 50000

    def encode_words(self, words, max_length=512):
        tokens, ids, word_ids = ["<s>"], [0], [None]
        for wi, w in enumerate(words):
            chunks = [w[i:i + 4] for i in range(0, len(w), 4)] or [w]
            for ci, c in enumerate(chunks):
                piece = ("Ġ" + c) if ci == 0 else c
                tokens.append(piece)
                ids.append(self._ids(piece))
                word_ids.append(wi)
        tokens, ids, word_ids = tokens[:max_length - 1], ids[:max_length - 1], word_ids[:max_length - 1]
        tokens.append("</s>"); ids.append(2); word_ids.append(None)
        return ids, word_ids, tokens


def load_tokenizer():
    """返回 (tokenizer, is_real)。"""
    try:
        from transformers import RobertaTokenizerFast
        # add_prefix_space=True：输入已经按词切好（is_split_into_words=True）时必须打开
        tok = RobertaTokenizerFast.from_pretrained(TOKENIZER_CKPT, add_prefix_space=True)
        return tok, True
    except Exception as e:
        print(f"[提示] 无法下载 {TOKENIZER_CKPT} 分词器（{type(e).__name__}），改用 ToyTokenizer。")
        return ToyTokenizer(), False


def tokenize_words(tok, words, max_length=512):
    """words(list[str]) -> (input_ids list, word_ids list, tokens list)"""
    if isinstance(tok, ToyTokenizer):
        return tok.encode_words(words, max_length)
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=max_length)
    return enc["input_ids"], enc.word_ids(), tok.convert_ids_to_tokens(enc["input_ids"])


# ----------------------------------------------------------------------------
# 示例文档：一张发票（像素坐标，页面 850 x 1100）
# 每个 segment = OCR 引擎给出的一行/一块文字。
#   funsd: FUNSD 风格的语义标签（HEADER / QUESTION / ANSWER / OTHER）
#   field: 字段级标签（没有 BIO 前缀，非目标字段一律 Other），第 7 章的字段抽取用
# ----------------------------------------------------------------------------
PAGE_W, PAGE_H = 850, 1100

SAMPLE_DOC = [
    # text,                       box (x0, y0, x1, y1)  funsd       field
    ("ACME SUPPLY CO.",           (60, 50, 330, 85),    "HEADER",   "Other"),
    ("INVOICE",                   (620, 50, 790, 90),   "HEADER",   "Other"),
    ("Invoice No:",               (60, 140, 170, 160),  "QUESTION", "Other"),
    ("INV-2024-0042",             (180, 140, 330, 160), "ANSWER",   "invoice_no"),
    ("Date:",                     (520, 140, 570, 160), "QUESTION", "Other"),
    ("March 15, 2024",            (580, 140, 720, 160), "ANSWER",   "date"),
    ("Bill To:",                  (60, 200, 140, 220),  "QUESTION", "Other"),
    ("Jane Smith",                (60, 225, 170, 245),  "ANSWER",   "name"),
    ("42 Maple Street, Toronto",  (60, 250, 300, 270),  "ANSWER",   "address"),
    ("Office chairs x 4",         (60, 360, 260, 380),  "OTHER",    "Other"),
    ("$1,000.00",                 (650, 360, 750, 380), "OTHER",    "Other"),
    ("Desk lamps x 5",            (60, 390, 230, 410),  "OTHER",    "Other"),
    ("$250.00",                   (670, 390, 750, 410), "OTHER",    "Other"),
    ("Total Amount:",             (520, 460, 650, 480), "QUESTION", "Other"),
    ("$1,250.00",                 (660, 460, 760, 480), "ANSWER",   "total"),
]


def get_words(box_level="segment"):
    """
    把 segment 拆成单词。返回 dict:
      words, boxes_px (每个词的像素框), funsd (BIO 标签), field, seg_id
    box_level="segment": 每个词用所在 segment 的框（论文和 FUNSD 代码的做法）
    box_level="word":    按字符数把 segment 框横向切开，给每个词一个自己的框
    """
    out = {"words": [], "boxes_px": [], "funsd": [], "field": [], "seg_id": []}
    for si, (text, box, lab, field) in enumerate(SAMPLE_DOC):
        ws = text.split()
        x0, y0, x1, y1 = box
        total = sum(len(w) for w in ws) + (len(ws) - 1)
        cursor = x0
        for wi, w in enumerate(ws):
            if box_level == "word":
                wx1 = cursor + (x1 - x0) * len(w) / total
                wbox = (int(cursor), y0, int(wx1), y1)
                cursor = wx1 + (x1 - x0) * 1 / total
            else:
                wbox = box
            out["words"].append(w)
            out["boxes_px"].append(wbox)
            if lab == "OTHER":
                out["funsd"].append("O")
            else:
                out["funsd"].append(("B-" if wi == 0 else "I-") + lab)
            out["field"].append(field)
            out["seg_id"].append(si)
    return out


def normalize_box(box, width=PAGE_W, height=PAGE_H):
    """像素坐标 -> 0..1000 整数坐标（和 LiLTfinetune/data/utils.py 的 normalize_bbox 一样）"""
    x0, y0, x1, y1 = box
    return [
        min(1000, max(0, int(1000 * x0 / width))),
        min(1000, max(0, int(1000 * y0 / height))),
        min(1000, max(0, int(1000 * x1 / width))),
        min(1000, max(0, int(1000 * y1 / height))),
    ]


def encode(tok, words, boxes_norm, special_box="zeros", max_length=512):
    """
    words + 归一化后的词级 boxes -> 模型输入。
    special_box: "zeros" -> <s> 和 </s> 都用 [0,0,0,0]（官方 FUNSD 代码 run_funsd.py 的做法）
                 "paper" -> <s>=[0,0,0,0]，</s>=[1000,1000,1000,1000]（论文 2.1.2 节的写法）
    """
    ids, word_ids, tokens = tokenize_words(tok, words, max_length)
    bbox = []
    for i, wid in enumerate(word_ids):
        if wid is None:
            is_eos = (i == len(word_ids) - 1)
            bbox.append([1000, 1000, 1000, 1000] if (special_box == "paper" and is_eos) else [0, 0, 0, 0])
        else:
            bbox.append(boxes_norm[wid])  # 同一个词的所有 subword 共用这个词的框
    return {
        "input_ids": torch.tensor([ids]),
        "bbox": torch.tensor([bbox]),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "word_ids": word_ids,
        "tokens": tokens,
    }


def model_inputs(enc):
    return {k: enc[k] for k in ("input_ids", "bbox", "attention_mask")}


# ----------------------------------------------------------------------------
# 用 hook 取出两条流每一层的输出
# ----------------------------------------------------------------------------
def run_with_taps(model, enc):
    """
    返回 dict:
      text_emb   [B,L,768]   文本 embedding 输出
      layout_emb [B,L,192]   layout embedding 输出
      position_ids
      text[i]    [B,L,768]   第 i 层文本流输出（i = 0..11）
      layout[i]  [B,L,192]   第 i 层 layout 流输出
      last_hidden_state      HF 返回的东西（= text[11]）
    model 可以是 LiltModel，也可以是带 .lilt 的任务模型。
    """
    base = model.lilt if hasattr(model, "lilt") else model
    taps = {"text": {}, "layout": {}}
    hooks = []

    def emb_hook(_m, _i, out):
        taps["text_emb"], taps["position_ids"] = out[0], out[1]

    def lay_emb_hook(_m, _i, out):
        taps["layout_emb"] = out

    hooks.append(base.embeddings.register_forward_hook(emb_hook))
    hooks.append(base.layout_embeddings.register_forward_hook(lay_emb_hook))
    for i, layer in enumerate(base.encoder.layer):
        # layer.output / layer.layout_output 是 FFN 之后、残差+LayerNorm 之后的输出，也就是这一层的最终输出
        hooks.append(layer.output.register_forward_hook(
            lambda _m, _i, out, i=i: taps["text"].__setitem__(i, out)))
        hooks.append(layer.layout_output.register_forward_hook(
            lambda _m, _i, out, i=i: taps["layout"].__setitem__(i, out)))
    try:
        with torch.no_grad():
            out = base(**model_inputs(enc))
    finally:
        for h in hooks:
            h.remove()
    taps["last_hidden_state"] = out.last_hidden_state
    return taps


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def is_layout_param(name):
    """LiLT 自己（layout 流）的参数名都带 layout_ 或 box_ 前缀；其余都是 RoBERTa 文本流"""
    return ("layout" in name) or ("box_" in name) or (".x_position" in name) or (".y_position" in name) \
        or (".h_position" in name) or (".w_position" in name)
