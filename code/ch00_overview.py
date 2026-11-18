"""
第 0 章：全局鸟瞰
跑一次完整前向，看清楚：进去的是什么，出来的是什么，参数分别属于哪条流。
"""
from common import *

model, real = load_model()
tok, _ = load_tokenizer()
cfg = model.config

banner("1. 配置：两条流的尺寸")
d_t = cfg.hidden_size
d_l = cfg.hidden_size // cfg.channel_shrink_ratio
print(f"层数 {cfg.num_hidden_layers}，头数 {cfg.num_attention_heads}，channel_shrink_ratio {cfg.channel_shrink_ratio}")
print(f"文本流  hidden {d_t:4d}  head_dim {d_t // cfg.num_attention_heads:3d}  FFN {cfg.intermediate_size}")
print(f"layout流 hidden {d_l:4d}  head_dim {d_l // cfg.num_attention_heads:3d}  FFN {cfg.intermediate_size // cfg.channel_shrink_ratio}")

banner("2. 参数归属：哪些来自 RoBERTa，哪些是 LiLT 自己的")
text_n = sum(p.numel() for n, p in model.named_parameters() if not is_layout_param(n) and not n.startswith("pooler"))
layout_n = sum(p.numel() for n, p in model.named_parameters() if is_layout_param(n))
pooler_n = sum(p.numel() for n, p in model.named_parameters() if n.startswith("pooler"))
print(f"文本流（= RoBERTa-base）: {text_n / 1e6:7.2f} M")
print(f"layout 流（LiLT 本体）  : {layout_n / 1e6:7.2f} M   ← 论文说 6.1M")
print(f"pooler                  : {pooler_n / 1e6:7.2f} M   （SER 用不到）")

banner("3. 一次前向")
doc = get_words()
boxes = [normalize_box(b) for b in doc["boxes_px"]]
enc = encode(tok, doc["words"], boxes)
print(f"输入 input_ids     {shape(enc['input_ids'])}")
print(f"输入 bbox          {shape(enc['bbox'])}   坐标范围 0..1000")
print(f"输入 attention_mask {shape(enc['attention_mask'])}")

taps = run_with_taps(model, enc)
print(f"\n文本 embedding     {shape(taps['text_emb'])}")
print(f"layout embedding   {shape(taps['layout_emb'])}")
print(f"第 12 层文本流输出  {shape(taps['text'][11])}")
print(f"第 12 层 layout 输出 {shape(taps['layout'][11])}")
print(f"HF 返回的 last_hidden_state {shape(taps['last_hidden_state'])}")
same = torch.allclose(taps["text"][11], taps["last_hidden_state"])
print(f"last_hidden_state == 第 12 层文本流输出 ? {same}")
print("→ HF 的 LiltModel 只返回文本流；layout 流最后的 192 维输出被丢掉了（第 6、7 章细讲）。")
