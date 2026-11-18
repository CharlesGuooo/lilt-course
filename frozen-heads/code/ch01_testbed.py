"""
第 1 章：实验台 —— 冻结特征的输出契约 + FUNSD 上的受控对照怎么搭
1) 数据：FUNSD 的规模、扁平标签分布、dev 怎么切
2) 几何阅读顺序：一维 head 看到的“相邻”是什么
3) 扁平 tag 能不能还原实例：连续同标签片段 vs 金标实体
4) 分词与窗口：多少文档超过 512 token
5) 冻结 LiLT 抽特征（只跑一次，缓存）+ 输出契约的传输量
6) backbone 本身在 CPU 上多慢（给后面的 head 计时做参照）
"""
from common import *
from heads import flat_runs

data = load_funsd()

# ---------------------------------------------------------------------------
banner("1. FUNSD：规模与扁平标签分布（词级，没有 BIO 前缀）")
for split in ["train", "dev", "test"]:
    docs = data[split]
    y = np.concatenate([d["labels"] for d in docs])
    n_ent = sum(len(np.unique(d["ent"])) for d in docs)
    dist = "  ".join(f"{LABELS[k]} {pct((y == k).mean())}" for k in range(4))
    print(f"  {split:<5s} {len(docs):3d} 份  {len(y):6,d} 个词  {n_ent:5,d} 个实体   {dist}")
print("  dev = 官方训练集里每隔 5 份取 1 份；test = 官方测试集，只用来报告最终结果。")

# ---------------------------------------------------------------------------
banner("2. 几何阅读顺序：一维 head 眼里的“相邻”")
d = data["test"][0]
print(f"  文档 {d['id']}，{len(d['words'])} 个词，前 14 个（按行、行内从左到右）：")
for i in range(14):
    x0, y0, x1, y1 = d["boxes"][i]
    print(f"   {i:2d}  {d['words'][i]:<16s} {LABELS[d['labels'][i]]:<9s} 实体 {d['ent'][i]:3d}   框 ({x0:3d},{y0:3d})-({x1:3d},{y1:3d})")

# ---------------------------------------------------------------------------
banner("3. 扁平 tag 能不能还原实例：连续同标签片段 vs 金标实体")
for split in ["train", "test"]:
    n_ent = n_run = merged = split_ent = 0
    run_len = []
    for d in data[split]:
        runs = flat_runs(d["labels"].tolist())
        n_run += len(runs)
        run_len += [j - i + 1 for i, j, _ in runs]
        merged += sum(len(np.unique(d["ent"][i:j + 1])) > 1 for i, j, _ in runs)
        for e in np.unique(d["ent"]):
            idx = np.where(d["ent"] == e)[0]
            if d["labels"][idx[0]] == 0:
                continue
            n_ent += 1
            pieces = sum(((idx >= i) & (idx <= j)).any() for i, j, _ in runs)   # 这个实体的词落在几个片段里
            split_ent += pieces > 1
    q = np.percentile(run_len, [50, 90, 99, 100])
    print(f"  {split:<5s} 字段实体 {n_ent:5,d} 个，连续同标签片段 {n_run:5,d} 个")
    print(f"        一个片段里混了 ≥2 个实体：{merged:4d} 个片段（{pct(merged / n_run)}）")
    print(f"        一个实体被拆成 ≥2 个片段：{split_ent:4d} 个实体（{pct(split_ent / n_ent)}）")
    print(f"        片段长度（词）p50 {q[0]:.0f} · p90 {q[1]:.0f} · p99 {q[2]:.0f} · 最长 {q[3]:.0f}")
print("  前者：两个相邻的同类实例中间没有 other 隔开，扁平 tag 把它们并成一段；")
print("  后者：多行实体在几何阅读顺序里被别的词隔开。两种情况都说明“连续同标签 ≠ 同一实例”。")

# ---------------------------------------------------------------------------
banner("4. 分词与窗口：LiLT 一次最多看 512 个 token")
from transformers import RobertaTokenizerFast
tok = RobertaTokenizerFast.from_pretrained(TOKENIZER_CKPT, add_prefix_space=True)
all_docs = data["train"] + data["dev"] + data["test"]
n_tok = [len(tok(d["words"], is_split_into_words=True, add_special_tokens=False)["input_ids"]) for d in all_docs]
n_word = [len(d["words"]) for d in all_docs]
print(f"  每份文档 subword 数：中位数 {statistics.median(n_tok):.0f}，最多 {max(n_tok)}；"
      f"平均每词 {sum(n_tok) / sum(n_word):.2f} 个 subword")
over = sum(t > 510 for t in n_tok)
print(f"  超过 510 个 subword、要切成 2 个窗口的文档：{over} / {len(all_docs)}")

# ---------------------------------------------------------------------------
banner("5. 冻结 LiLT 抽特征（只跑一次，结果缓存）+ 输出契约的传输量")
extract_features(data)
enc_s = [d["enc_s"] for d in all_docs]
print(f"  {len(all_docs)} 份文档前向合计 {sum(enc_s):.1f} s，每份中位数 {1000 * statistics.median(enc_s):.0f} ms"
      f"（torch {all_docs[0]['enc_threads']} 线程，{cpu_name()}）")
d = data["test"][0]
print(f"  缓存里每个词存了：text4 {d['text4'].shape[1:]}（文本流第 9-12 层） + layout {d['layout'].shape[1:]}（layout 流第 12 层），fp16")
print("  不同输出契约，一份文档传给 head 的数据量：")
med_w, max_w = int(statistics.median(n_word)), max(n_word)
for view, dim in VIEWS.items():
    if view == "layout":
        continue
    print(f"    {view:<6s} {dim:5d} 维   中位文档 {med_w} 词: fp32 {med_w * dim * 4 / 1e6:5.2f} MB · fp16 {med_w * dim * 2 / 1e6:5.2f} MB"
          f"   最长 {max_w} 词: fp32 {max_w * dim * 4 / 1e6:5.2f} MB")
print("  只是张量本身的字节数；序列化、压缩和网络耗时要在真实服务上量。")

# ---------------------------------------------------------------------------
banner("6. 参照：backbone 一次前向在这台 CPU 上多慢")
model, tok = load_backbone()
long_doc = max(all_docs, key=lambda d: len(d["words"]))
enc = tok(long_doc["words"], is_split_into_words=True, add_special_tokens=False)
ids, wid = enc["input_ids"][:510], enc.word_ids()[:510]
inp = dict(input_ids=torch.tensor([[0] + ids + [2]]),
           bbox=torch.tensor([[[0] * 4] + [list(long_doc["boxes"][w]) for w in wid] + [[0] * 4]]),
           attention_mask=torch.ones(1, 512, dtype=torch.long))
res = {}
for th in [1, 4]:
    p50, p95 = latency_ms(lambda: model(**inp), repeat=10, warmup=2, threads=th)
    res[th] = (p50, p95)
    print(f"  512 token，{th} 线程：p50 {p50:7.1f} ms   p95 {p95:7.1f} ms")
save_result("testbed", {
    "cpu": cpu_name(), "backbone_ms": {str(k): v for k, v in res.items()},
    "n_tok_median": statistics.median(n_tok), "n_tok_max": max(n_tok), "over_510": over,
    "n_word_median": med_w, "n_word_max": max_w, "enc_total_s": sum(enc_s),
    "enc_median_ms": 1000 * statistics.median(enc_s),
})
print("  后面每一章的 head 计时都用 1 线程、batch=1，和这里的 1 线程数字对照着看。")
