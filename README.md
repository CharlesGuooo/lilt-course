# LiLT 架构课

八章拆解 [LiLT](https://arxiv.org/abs/2202.13669)（ACL 2022）：文本流、layout 流、BiACM、预训练、任务头，以及“冻结 backbone + 轻量 head”的落地方式。

每一章对应 `code/` 里的一个脚本。页面上的数字都来自用真实权重 `SCUT-DLVCLab/lilt-roberta-en-base` 跑出的输出，每一步都手算一遍，再和模型输出对比。

**在线阅读**：https://charlesguooo.github.io/lilt-course/

## 目录

| 章 | 页面 | 配套脚本 |
|---|---|---|
| 00 全局鸟瞰 | [html/ch00_overview.html](html/ch00_overview.html) | [code/ch00_overview.py](code/ch00_overview.py) |
| 01 RoBERTa 复习 | [html/ch01_roberta.html](html/ch01_roberta.html) | [code/ch01_roberta.py](code/ch01_roberta.py) |
| 02 输入准备 | [html/ch02_inputs.html](html/ch02_inputs.html) | [code/ch02_inputs.py](code/ch02_inputs.py) |
| 03 Layout embedding | [html/ch03_layout_embedding.html](html/ch03_layout_embedding.html) | [code/ch03_layout_embedding.py](code/ch03_layout_embedding.py) |
| 04 双流 encoder 与 BiACM | [html/ch04_biacm.html](html/ch04_biacm.html) | [code/ch04_biacm.py](code/ch04_biacm.py) |
| 05 预训练 | [html/ch05_pretraining.html](html/ch05_pretraining.html) | [code/ch05_pretraining.py](code/ch05_pretraining.py) |
| 06 输出与任务头 | [html/ch06_heads.html](html/ch06_heads.html) | [code/ch06_heads.py](code/ch06_heads.py) |
| 07 落地：冻结 backbone | [html/ch07_frozen_backbone.html](html/ch07_frozen_backbone.html) | [code/ch07_frozen_backbone.py](code/ch07_frozen_backbone.py) |

## 本地阅读

克隆后用浏览器直接打开 `html/index.html`。不需要联网，不依赖任何 CDN。

## 运行配套脚本

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ch00_overview.py      # 第一次会从 HuggingFace 下载约 500 MB 权重
```

详见 [code/README.md](code/README.md)。

## 数据来源

- 页面里的运行输出：Python 3.13.12 · torch 2.15.0 · transformers 5.23.0 · macOS CPU。第 7 章的 CPU 计时随机器变化。
- `html/assets/data.js`：交互图用的数据，用同一套权重和 `code/common.py` 导出。
- 论文里的数字引自 Wang, Jin, Ding. *LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding*. ACL 2022.

## 致谢

- 论文与官方实现：[jpWang/LiLT](https://github.com/jpWang/LiLT)（MIT License，Copyright (c) 2022 Jiapeng Wang）。课程中引用了官方实现的少量代码片段。
- HuggingFace transformers 的 LiLT 实现。
