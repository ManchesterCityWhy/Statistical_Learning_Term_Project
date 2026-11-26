"""Compare the two confirmation tables of the README with the score CSVs, cell by cell."""
import csv, glob, collections, re
R = "/Volumes/Data/Projects/统计学习大作业/repo/experiments/round6_seed42/"
S = R + "scores/"
def load(pat):
    rows = []
    for f in sorted(glob.glob(S + pat)): rows += list(csv.DictReader(open(f)))
    return rows
MAP = {"R5 的做法": "R5 (round-5 file)", "**R6（候选，锚点 20%）**": "R6 (candidate)", "R6 第一版（锚点 37.5%）": "R6 at round-5 share",
       "R6，锚点不校正": "R6 raw anchor", "R6 第一版，锚点不校正": "R6 raw anchor at round-5 share", "R6，不加锚点": "R6 base only",
       "R6，基底换成全部文库": "R6 base, all libraries", "R5，不加锚点": "R5 base only", "R5 基底 + 校正锚点（37.5%）": "R5 base, harmonised anchor",
       "R5 基底 + 校正锚点（20%）": "R5 base, R6 anchor rule", "R5 基底 + 校正锚点（R5 的比例）": "R5 base, harmonised anchor",
       "R5 基底 + 校正锚点（R6 的比例）": "R5 base, R6 anchor rule"}
SCEN = ["heart/deep", "heart/mid", "heart/shallow", "all/asis"]
PROX = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all", "D/raw", "D/dm", "raw", "h3", "dm"]
W = {"heart/deep": .48, "heart/mid": .16, "heart/shallow": .16, "all/asis": .2}
sc = collections.defaultdict(dict); px = collections.defaultdict(dict)
for r in load("round6_final__scen_confirmation_rho0.4_trend0.csv"): sc[r["config"]][r["scenario"]] = float(r["score"])
for r in load("round6_final__proxy_confirmation_rho0.4_trend0_[ab].csv"): px[r["config"]][r["proxy"]] = float(r["score"])
text = open(R + "README.md").read()
sec_s = text.split("### 情景表（复核集）")[1].split("###")[0]
sec_p = text.split("### 有时间差的代理和 D（复核集）")[1].split("## 对 R6 不利的证据")[0]
bad = n = 0
for line in sec_s.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if cells and cells[0] in MAP:
        vals = [float(c) for c in cells[1:]]
        ref = [sc[MAP[cells[0]]][s] for s in SCEN]; ref.append(sum(W[s] * sc[MAP[cells[0]]][s] for s in SCEN))
        for a, b in zip(vals, ref):
            n += 1
            if abs(a - round(b, 2)) > 0.0051: bad += 1; print("SCEN mismatch", cells[0], a, b)
for line in sec_p.splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if cells and cells[0] in MAP:
        vals = [float(c) for c in cells[1:]]
        ref = [px[MAP[cells[0]]][p] for p in PROX]
        for a, b in zip(vals, ref):
            n += 1
            if abs(a - round(b, 2)) > 0.0051: bad += 1; print("PROXY mismatch", cells[0], a, b)
    if cells and cells[0] == "R6 − R5":
        vals = [float(c.replace("−", "-")) for c in cells[1:]]
        for a, p in zip(vals, PROX):
            n += 1; b = px["R6 (candidate)"][p] - px["R5 (round-5 file)"][p]
            if abs(a - round(b, 2)) > 0.0051: bad += 1; print("DIFF mismatch", p, a, b)
print("cells checked", n, "mismatches", bad)
