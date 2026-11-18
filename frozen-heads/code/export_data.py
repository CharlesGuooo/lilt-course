"""
把 ch01 ~ ch09 存下的结果整理成网页用的 ../html/assets/data.js（window.HEADS）。
先依次跑完 ch01 ~ ch09，再运行：python export_data.py
网页里的交互组件用一份 FUNSD 测试表单（词、框、标签、实体编号、question→answer 配对）。
"""
import json

from common import *

ROWS = load_result("compare")["rows"]
out = {"labels": LABELS, "testbed": load_result("testbed"), "compare": load_result("compare")}

res = {}
for k, fam, name in ROWS:
    s = load_result(k)
    res[k] = {
        "name": name, "fam": fam,
        "macro": [s["macro_f1"]["mean"], s["macro_f1"]["std"]],
        "micro": s["micro_f1"]["mean"],
        "pc": {c: [s["per_class_f1"][c]["mean"], s["per_class_f1"][c]["std"]] for c in LABELS},
        "ent": s["ent_full"]["mean"], "ece": s["ece"]["mean"],
        "params": s["params"], "bytes": s["file_bytes"],
        "lat": {n: s["latency"][n][0] for n in s["latency"]},
        "n_runs": s["n_runs"],
    }
out["results"] = res
for k in ["linear_layout", "linear_last4"]:
    s = load_result(k)
    out.setdefault("views", {})[k] = {"macro": [s["macro_f1"]["mean"], s["macro_f1"]["std"]], "params": s["params"],
                                      "pc": {c: s["per_class_f1"][c]["mean"] for c in LABELS}}
out["reliability"] = load_result("calibration")
out["linear_cm"] = load_result("linear_reliability")["cm"]
out["crf_trans"] = load_result("crf")["trans"]
out["bias"] = load_result("layout2d")["bias"]
out["neighbors"] = load_result("neighbors")
out["span_eval"] = load_result("span")["span_eval"]
out["fewshot"] = load_result("fewshot")

# 交互组件用的示例表单：测试集里词数 90~150、有 header、question→answer 配对最多的一份
data = load_funsd()
cands = [d for d in data["test"] if 90 <= len(d["words"]) <= 150 and (d["labels"] == 3).any()]
doc = max(cands, key=lambda d: (len(d["links"]), -abs(len(d["words"]) - 120)))
out["doc"] = {
    "id": doc["id"], "size": list(doc["size"]),
    "words": [{"t": w, "b": [int(v) for v in b], "l": int(l), "e": int(e)}
              for w, b, l, e in zip(doc["words"], doc["boxes"], doc["labels"], doc["ent"])],
    "links": [list(p) for p in doc["links"]],
}
path = os.path.join(HERE, "..", "html", "assets", "data.js")
with open(path, "w") as f:
    f.write("/* 由 code/export_data.py 生成：FUNSD 上冻结 LiLT + 各类 head 的实测结果。不要手改。 */\n")
    f.write("window.HEADS = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
print(f"写出 {os.path.relpath(path, HERE)}：{os.path.getsize(path) / 1024:.1f} KB；示例表单 {doc['id']}，{len(doc['words'])} 个词，"
      f"{len(doc['links'])} 对 question→answer")
