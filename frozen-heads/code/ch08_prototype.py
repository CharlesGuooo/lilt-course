"""
第 8 章：原型与相似度 head —— 不训练（或几乎不训练）的分类
1) 三种做法：类均值原型（NCM）、全训练集 kNN、ProtoNER 式多原型（线性 head 判对的词里抽样）
2) 要存什么、存多大、单线程延迟
3) 少样本：只给 1 / 2 / 5 / 10 / 30 份训练文档时，谁掉得少
4) 新增字段：旧标注里没有 header（都算 other），新来 k 份标了 header 的文档
"""
from common import *
from heads import LinearHead

data = load_funsd()
extract_features(data)
K = len(LABELS)


def l2n(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def stack(docs, relabel=None):
    X = torch.from_numpy(np.concatenate([word_feats(d, "both") for d in docs]))
    y = torch.from_numpy(np.concatenate([d["labels"] for d in docs]))
    if relabel:
        for a, b in relabel.items():
            y = torch.where(y == a, torch.tensor(b), y)
    return X, y


class NCM:
    """类均值原型：每类一个向量 = 该类所有词（单位化后）的平均。预测 = 余弦最近的原型"""

    def __init__(self, X, y, tau=None, dev=None):
        self.classes = [c for c in range(K) if (y == c).any()]
        self.P = l2n(torch.stack([l2n(X[y == c]).mean(0) for c in self.classes]))
        self.tau = tau or (self.fit_tau(*dev) if dev else 10.0)

    def scores(self, X):
        return l2n(X) @ self.P.T                      # [n, 类数] 余弦相似度

    def fit_tau(self, X, y):
        """温度缩放：在 dev 上找让负对数似然最小的温度，把相似度变成可用的概率"""
        s, yy = self.scores(X), torch.tensor([self.classes.index(int(v)) if int(v) in self.classes else -1 for v in y])
        keep = yy >= 0
        best = min(range(1, 101), key=lambda t: F.cross_entropy(t * s[keep], yy[keep]).item())
        return float(best)

    def predict(self, X):
        p = (self.tau * self.scores(X)).softmax(-1)
        conf, idx = p.max(-1)
        return torch.tensor(self.classes)[idx], conf


class KNN:
    """把训练词全部存下来（单位化），预测时找余弦最近的 k 个，按相似度加权投票"""

    def __init__(self, X, y, k=10):
        self.B, self.y, self.k = l2n(X), y, k

    def predict(self, X):
        sim, idx = (l2n(X) @ self.B.T).topk(min(self.k, len(self.y)), dim=-1)
        votes = torch.zeros(len(X), K).scatter_add_(1, self.y[idx], sim.clamp(min=0))
        conf, pred = (votes / votes.sum(1, keepdim=True).clamp(min=1e-9)).max(-1)
        return pred, conf


def eval_model(m, docs):
    preds = []
    for d in docs:
        p, c = m.predict(torch.from_numpy(word_feats(d, "both")))
        preds.append((p.numpy(), c.numpy()))
    return evaluate(docs, preds)


Xtr, ytr = stack(data["train"])
Xdv, ydv = stack(data["dev"])

# ---------------------------------------------------------------------------
banner("1. 三种原型 / 相似度 head（冻结特征 both 960，不更新 LiLT）")
try:
    base = load_result("linear_both")
    base["name"] = "linear（基线）"
    print_summary(base)
except FileNotFoundError:
    pass
ncm = NCM(Xtr, ytr, dev=(Xdv, ydv))
ncm_s = summarize("NCM 类均值（4 个原型）", [dict(eval_model(ncm, data["test"]), best_epoch=0, train_s=0.0)])
print_summary(ncm_s)
print(f"     温度 τ = {ncm.tau:.0f}（在 dev 上按负对数似然选的）")
knn = KNN(Xtr, ytr, k=10)
knn_s = summarize(f"kNN k=10（{len(ytr):,} 个词）", [dict(eval_model(knn, data["test"]), best_epoch=0, train_s=0.0)])
print_summary(knn_s)
proto_runs = []
for seed in SEEDS:
    lin, _ = train_head(LinearHead, data, "both", seed=seed)
    with torch.no_grad():
        ok = lin(Xtr).argmax(-1) == ytr                    # 只从“线性 head 判对的词”里取原型（ProtoNER 的规则）
    g = torch.Generator().manual_seed(seed)
    keep = []
    for c in range(K):
        idx = torch.where(ok & (ytr == c))[0]
        keep.append(idx[torch.randperm(len(idx), generator=g)[:50]])
    keep = torch.cat(keep)
    pm = KNN(Xtr[keep], ytr[keep], k=5)
    proto_runs.append(dict(eval_model(pm, data["test"]), best_epoch=0, train_s=0.0))
proto_s = summarize("ProtoNER 式 每类 50 个原型", proto_runs)
print_summary(proto_s)

# ---------------------------------------------------------------------------
banner("2. 要存什么、多大、多快（单线程，batch = 一份文档的 n 个词）")
store = {"NCM": 4 * 960 * 4, "kNN": len(ytr) * 960 * 2, "ProtoNER 式": 200 * 960 * 4}
print(f"  NCM         存 4 个 960 维原型（fp32）              {store['NCM'] / 1024:10.1f} KB")
print(f"  kNN         存全部 {len(ytr):,} 个训练词（fp16）        {store['kNN'] / 1024 / 1024:10.1f} MB")
print(f"  ProtoNER 式 存 200 个原型（fp32）                   {store['ProtoNER 式'] / 1024:10.1f} KB")
lat = {}
for name, m in [("NCM", ncm), ("kNN", knn), ("ProtoNER 式", pm)]:
    lat[name] = {}
    for n in (64, 128, 256, 512):
        x = torch.randn(n, 960)
        lat[name][n] = latency_ms(lambda: m.predict(x))
    print_latency(name, lat[name], 12)
print("  kNN 的延迟和存储都随“存了多少训练词”线性增长；新任务、新数据越多越重。")

# ---------------------------------------------------------------------------
banner("3. 少样本：训练文档只有 n 份时（测试集宏F1，3 次随机抽样取平均）")
sizes = [1, 2, 5, 10, 30, len(data["train"])]
curve = {"sizes": sizes, "linear": [], "NCM": [], "kNN": []}
print(f"  {'训练文档':>8s} {'linear':>8s} {'NCM':>8s} {'kNN':>8s}")
for n in sizes:
    vals = {"linear": [], "NCM": [], "kNN": []}
    for seed in SEEDS:
        rng = random.Random(100 + seed)
        sub = rng.sample(data["train"], n) if n < len(data["train"]) else data["train"]
        X, y = stack(sub)
        head, _ = train_head(LinearHead, {"train": sub, "dev": data["dev"]}, "both", seed=seed)
        vals["linear"].append(evaluate(data["test"], predict_docs(head, data["test"], "both"))["macro_f1"])
        vals["NCM"].append(eval_model(NCM(X, y, dev=(Xdv, ydv)), data["test"])["macro_f1"])
        vals["kNN"].append(eval_model(KNN(X, y, k=10), data["test"])["macro_f1"])
    for k in vals:
        curve[k].append(float(np.mean(vals[k])))
    print(f"  {n:>8d} {curve['linear'][-1]:8.3f} {curve['NCM'][-1]:8.3f} {curve['kNN'][-1]:8.3f}")
print("  linear 每次都重新训练（同样的早停规则）；NCM、kNN 只是存向量，没有梯度更新。")

# ---------------------------------------------------------------------------
banner("4. 新增字段：旧标注没有 header（header 词当 other 标），新来 k 份标了 header 的文档")
H = LABELS.index("header")
inc = {}
print(f"  {'k':>3s}  {'做法':<34s} {'header F1':>10s} {'宏F1':>8s}")
for k in [1, 5, 20]:
    res = {"NCM：Q/A 用全部，other/header 只用新文档": [], "linear：只用 k 份新文档重训": [],
           "linear：旧数据（header→other）+ 新文档": []}
    for seed in SEEDS:
        rng = random.Random(200 + seed)
        new = rng.sample(data["train"], k)
        ids = {d["id"] for d in new}
        old = [dict(d, labels=np.where(d["labels"] == H, 0, d["labels"])) for d in data["train"] if d["id"] not in ids]
        # 原型法：question / answer 原型用全部数据；other 原型丢掉旧的（旧 other 里混着 header），只用新文档重算
        Xo, yo = stack(old)
        Xn, yn = stack(new)
        Xm = torch.cat([Xo[(yo == 1) | (yo == 2)], Xn])
        ym = torch.cat([yo[(yo == 1) | (yo == 2)], yn])
        r = eval_model(NCM(Xm, ym, dev=(Xdv, ydv)), data["test"])
        res["NCM：Q/A 用全部，other/header 只用新文档"].append(r)
        for name, tr in [("linear：只用 k 份新文档重训", new), ("linear：旧数据（header→other）+ 新文档", old + new)]:
            head, _ = train_head(LinearHead, {"train": tr, "dev": data["dev"]}, "both", seed=seed)
            res[name].append(evaluate(data["test"], predict_docs(head, data["test"], "both")))
    inc[k] = {}
    for name, rs in res.items():
        hf = float(np.mean([r["per_class"]["header"]["f1"] for r in rs]))
        mf = float(np.mean([r["macro_f1"] for r in rs]))
        inc[k][name] = {"header_f1": hf, "macro_f1": mf}
        print(f"  {k:>3d}  {name:<34s} {hf:10.3f} {mf:8.3f}")
print("  dev（早停、选温度）用的是带 header 的完整标注；真实场景里新字段的验证集也得一起标。")

ncm_s["latency"], knn_s["latency"], proto_s["latency"] = lat["NCM"], lat["kNN"], lat["ProtoNER 式"]
ncm_s["file_bytes"], knn_s["file_bytes"], proto_s["file_bytes"] = store["NCM"], store["kNN"], store["ProtoNER 式"]
ncm_s["params"], knn_s["params"], proto_s["params"] = 4 * 960, len(ytr) * 960, 200 * 960
save_result("ncm", ncm_s)
save_result("knn", knn_s)
save_result("protoner", proto_s)
save_result("fewshot", {"curve": curve, "incremental": {str(k): v for k, v in inc.items()}})
