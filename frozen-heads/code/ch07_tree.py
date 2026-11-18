"""
第 7 章：树模型 head —— 把每个词当成表格里的一行
1) 直方图 GBDT（sklearn 的 HistGradientBoostingClassifier，和 LightGBM 同一类算法）读 960 维特征 + 词框
2) 先用 PCA 降到 64 维再喂树
3) 模型大小、树的数量、训练时间、单线程预测延迟
"""
import pickle as pkl

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from common import *

data = load_funsd()
extract_features(data)


def table(docs, proj=None):
    """每个词一行：特征（可选先投影）+ 归一化后的词框 4 列"""
    rows = []
    for d in docs:
        f = word_feats(d, "both")
        if proj is not None:
            f = proj.transform(f)
        rows.append(np.concatenate([f, d["boxes"] / 1000.0], 1))
    return np.concatenate(rows).astype(np.float32)


def fit_eval(name, proj=None):
    Xtr, ytr = table(data["train"], proj), np.concatenate([d["labels"] for d in data["train"]])
    Xdv, ydv = table(data["dev"], proj), np.concatenate([d["labels"] for d in data["dev"]])
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1, max_leaf_nodes=31,
                                         early_stopping=True, n_iter_no_change=20, random_state=0)
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr, X_val=Xdv, y_val=ydv)              # 早停用我们自己的 dev 文档，不随机切词
    train_s = time.perf_counter() - t0
    preds = []
    for d in data["test"]:
        pr = clf.predict_proba(table([d], proj))
        preds.append((pr.argmax(1), pr.max(1)))
    r = evaluate(data["test"], preds)
    r.update({"best_epoch": int(clf.n_iter_), "dev_macro_f1": float("nan"), "train_s": train_s})
    s = summarize(name, [r])
    s["file_bytes"] = len(pkl.dumps(clf)) + (len(pkl.dumps(proj)) if proj is not None else 0)
    s["n_trees"] = int(clf.n_iter_) * len(LABELS)
    s["n_features"] = Xtr.shape[1]
    s["params"] = int(sum(len(t.nodes) for it in clf._predictors for t in it))   # 树节点总数（每个节点存特征、阈值、取值）
    return s, clf


# ---------------------------------------------------------------------------
banner("1. 直方图 GBDT：960 维冻结特征 + 4 个框坐标（单次训练，没有随机性）")
try:
    base = load_result("linear_both")
    base["name"] = "linear（基线）"
    print_summary(base)
except FileNotFoundError:
    pass
full_s, full = fit_eval("GBDT · 964 维")
print_summary(full_s)
print(f"  {full_s['n_features']} 个特征，{full.n_iter_} 轮 × {len(LABELS)} 类 = {full_s['n_trees']} 棵树、{full_s['params']:,} 个节点，"
      f"训练 {full_s['train_s']['mean']:.0f} s（{os.cpu_count()} 核全开）")

# ---------------------------------------------------------------------------
banner("2. 先 PCA 到 64 维再喂树")
pca = PCA(64, random_state=0).fit(np.concatenate([word_feats(d, "both") for d in data["train"]]))
print(f"  64 个主成分保留了 {pca.explained_variance_ratio_.sum():.1%} 的方差")
pca_s, small = fit_eval("GBDT · PCA64 + 4", pca)
print_summary(pca_s)
print(f"  {pca_s['n_features']} 个特征，{small.n_iter_} 轮 → {pca_s['n_trees']} 棵树、{pca_s['params']:,} 个节点，训练 {pca_s['train_s']['mean']:.0f} s")

# ---------------------------------------------------------------------------
banner("3. 大小与单线程预测延迟（batch = 一份文档的 n 个词）")
Xte = table(data["test"])
Xte_p = table(data["test"], pca)
lat = {}
for name, s, clf, X in [("GBDT · 964 维", full_s, full, Xte), ("GBDT · PCA64 + 4", pca_s, small, Xte_p)]:
    print(f"  {name:<18s} 模型文件 {s['file_bytes'] / 1024:8.1f} KB（pickle，PCA 版含投影矩阵）")
    out = {}
    with threadpool_limits(1):
        for n in (64, 128, 256, 512):
            out[n] = latency_ms(lambda: clf.predict_proba(X[:n]), repeat=30, warmup=3)
    lat[name] = out
for name, l in lat.items():
    print_latency(name, l, 18)
print("  PCA 版的计时不含投影（n×960 乘 960×64 的一次矩阵乘，本身很快）。")
print("  对照：960 维线性 head 的文件约 16 KB，见第 2 章。")

full_s["latency"] = lat["GBDT · 964 维"]
pca_s["latency"] = lat["GBDT · PCA64 + 4"]
save_result("tree", full_s)
save_result("tree_pca", pca_s)
