# LiLT 课程配套代码

每一章一个脚本，和课程网页的章节一一对应。脚本以打印为主：
每一步都把真实的 tensor shape 和关键数值打印出来，并且手算一遍和模型输出对比。

## 环境

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

第一次运行会从 HuggingFace 下载两样东西（约 500MB）：
- `SCUT-DLVCLab/lilt-roberta-en-base`（模型权重）
- `roberta-base`（分词器）

下载不了时，脚本会自动退回随机初始化和一个替身分词器。shape 和流程不变，但数值没有意义。

## 运行

```bash
python ch00_overview.py
python ch01_roberta.py
...
python ch07_frozen_backbone.py
```

| 脚本 | 内容 |
|---|---|
| `common.py` | 加载模型/分词器、示例发票、输入编码、用 hook 取出两条流每一层的输出 |
| `ch00_overview.py` | 两条流的尺寸、参数归属、一次完整前向 |
| `ch01_roberta.py` | BPE、position_ids、手算 RoBERTa 第 0 层、用 LiLT 的文本权重拼出纯 RoBERTa |
| `ch02_inputs.py` | 坐标归一化、segment 框和词框、subword 对齐、特殊 token 的框 |
| `ch03_layout_embedding.py` | 手算 layout embedding、为什么要 1D 位置、坐标表的平滑性 |
| `ch04_biacm.py` | 手算一整层 LiLT、两个 softmax 相同、各层 layout 分数占比、打乱框的消融 |
| `ch05_pretraining.py` | 权重拼接、DETACH 挡住的梯度、MVLM/KPL/CAI 的标签、学习率放慢 |
| `ch06_heads.py` | 原版 960 维 head 和 HF 768 维 head 的差别、SER 解码、RE 的 biaffine |
| `ch07_frozen_backbone.py` | 冻结特征抽取器、线性 head 训练流程、CPU 计时、特征缓存大小 |

测试环境：Python 3.13，torch 2.15，transformers 5.22.2。transformers 4.4x 也能跑：脚本只用模块名做 hook，不依赖返回值的 tuple 结构。
