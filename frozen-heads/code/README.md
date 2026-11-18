# 冻结 LiLT 上的轻量 head · 配套代码

每章一个脚本，和课程网页一一对应。脚本以打印为主，网页里的输出框都是这些脚本的原样输出。

## 环境

```bash
cd frozen-heads/code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

第一次运行会下载：

- `SCUT-DLVCLab/lilt-roberta-en-base`（LiLT 权重）和 `roberta-base`（分词器），约 500 MB。
- FUNSD 数据集（约 17 MB，<https://guillaumejaume.github.io/FUNSD/>）。FUNSD 只允许非商业的研究和教学用途，下载前请读它的许可说明。

下载的数据、抽好的特征和各章结果都放在 `code/.cache/`。要换位置，可以设置环境变量 `HEADS_CACHE`。

## 运行顺序

```bash
python ch01_testbed.py      # 抽特征并缓存（CPU 上约 1 分钟），后面各章都用这份缓存
python ch02_linear_mlp.py
python ch03_sequence.py     # BiLSTM 最慢，约 5 分钟
python ch04_crf.py
python ch05_layout.py
python ch06_span.py         # 含 dev 上的小网格搜索，约 10 分钟
python ch07_tree.py
python ch08_prototype.py
python ch09_compare.py      # 只汇总前面存下的结果
python export_data.py       # 重新生成 ../html/assets/data.js
```

| 文件 | 内容 |
|---|---|
| `common.py` | FUNSD 读取与几何阅读顺序、冻结 LiLT 抽特征并缓存、统一的训练循环和早停、逐词指标、ECE、单线程计时 |
| `heads.py` | 神经网络类 head：线性、MLP、一维卷积、BiLSTM、Transformer、二维相对位置偏置、CRF、biaffine 片段 |
| `ch01_testbed.py` | 数据规模与标签分布、阅读顺序、连续同标签 vs 实例、窗口切分、输出契约的传输量、backbone 延迟 |
| `ch02_linear_mlp.py` | 四种输出契约、线性 vs MLP、置信度校准、混淆矩阵、成本 |
| `ch03_sequence.py` | 一维卷积 / BiLSTM / Transformer、一维相邻 vs 二维相邻、延迟随长度的变化 |
| `ch04_crf.py` | 有 / 无 CRF、学到的转移分数、边际概率当置信度 |
| `ch05_layout.py` | 二维相对位置偏置的消融、学到的偏置、网格化张量的大小估算 |
| `ch06_span.py` | 扁平 tag 派生片段、类别不平衡、片段级 exact match 与扁平 tag 的上限 |
| `ch07_tree.py` | 直方图 GBDT、PCA 降维、树的数量与延迟 |
| `ch08_prototype.py` | 类均值原型、kNN、ProtoNER 式多原型、少样本曲线、新增字段 |
| `ch09_compare.py` | 汇总表、多任务存储总账、backbone 与 head 的时间占比 |
| `export_data.py` | 把结果整理成网页用的 `data.js` |

## 说明

- 所有 head 在同一份冻结特征上训练，统一训练预算：AdamW（lr 2e-3）、每批 8 份文档、最多 60 轮、dev 宏 F1 连续 10 轮不涨就停、3 个随机种子。没有针对单个 head 调参，片段 head 的类别权重除外（第 6 章在 dev 上选）。
- 计时：单线程、batch=1、只算 head，不含特征传输。数字随机器变化，比例比绝对值更有参考意义。
- 测试环境：Python 3.13，torch 2.15，transformers 5.23，scikit-learn 1.9，Apple M1。
