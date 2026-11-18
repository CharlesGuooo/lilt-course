# 冻结 LiLT 上的轻量 head

[LiLT 架构课](../README.md) 的续篇。LiLT 冻结成所有任务共享的特征抽取器以后，每个任务的抽取 head 选什么结构？课程把八类候选放到 FUNSD 上，同条件比较准确率、置信度校准、文件大小和 CPU 延迟。

**在线阅读**：https://charlesguooo.github.io/lilt-course/frozen-heads/

## 目录

| 章 | 页面 | 配套脚本 |
|---|---|---|
| 00 问题与证据规则 | [html/ch00_problem.html](html/ch00_problem.html) | — |
| 01 实验台：冻结特征与 FUNSD | [html/ch01_testbed.html](html/ch01_testbed.html) | [code/ch01_testbed.py](code/ch01_testbed.py) |
| 02 基线：线性层与小 MLP | [html/ch02_linear.html](html/ch02_linear.html) | [code/ch02_linear_mlp.py](code/ch02_linear_mlp.py) |
| 03 序列上下文：卷积 / LSTM / 注意力 | [html/ch03_sequence.html](html/ch03_sequence.html) | [code/ch03_sequence.py](code/ch03_sequence.py) |
| 04 解码层：CRF | [html/ch04_crf.html](html/ch04_crf.html) | [code/ch04_crf.py](code/ch04_crf.py) |
| 05 版面感知 head | [html/ch05_layout.html](html/ch05_layout.html) | [code/ch05_layout.py](code/ch05_layout.py) |
| 06 片段与 biaffine | [html/ch06_span.html](html/ch06_span.html) | [code/ch06_span.py](code/ch06_span.py) |
| 07 树模型 head | [html/ch07_tree.html](html/ch07_tree.html) | [code/ch07_tree.py](code/ch07_tree.py) |
| 08 原型与相似度 | [html/ch08_prototype.html](html/ch08_prototype.html) | [code/ch08_prototype.py](code/ch08_prototype.py) |
| 09 横向对比与验证方案 | [html/ch09_compare.html](html/ch09_compare.html) | [code/ch09_compare.py](code/ch09_compare.py) |

## 本地阅读

用浏览器直接打开 `html/index.html`，不需要联网，也不依赖任何 CDN。

## 重跑实验

见 [code/README.md](code/README.md)。

## 证据边界

- 网页上的实测数字都来自 FUNSD（199 份英文扫描表单，词级扁平标签）和一台 Apple M1，只代表这个设置。换成别的文档、别的标签、别的 CPU，结论要重新量。
- 主指标是逐词宏 F1，不是 FUNSD 官方的实体级 F1，也不是字段值的 exact match。
- 引用的论文数字只属于各自的原始设置，不能当作本设置下的预测。

## 数据与致谢

- FUNSD：Jaume, Ekenel, Thiran. *FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents*. ICDAR-OST 2019。仅限非商业的研究和教学用途；原始图像来自 RVL-CDIP，受版权保护。本仓库不包含数据集，只在 `html/assets/data.js` 里放了一份测试表单的词、框和标签，供网页交互组件演示用。
- LiLT 权重：[SCUT-DLVCLab/lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base)；论文与官方实现见 [jpWang/LiLT](https://github.com/jpWang/LiLT)。
- 各章引用的论文列在第 9 章末尾。
