"""Markdown tables for the round-6 README, straight from the score CSVs (no hand-typed numbers)."""
import csv, glob, collections, sys
S = "/Volumes/Data/Projects/统计学习大作业/repo/experiments/round6_seed42/scores/"
def load(pat):
    rows = []
    for f in sorted(glob.glob(S + pat)):
        rows += list(csv.DictReader(open(f)))
    return rows
def md(header, lines):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    out += ["| " + " | ".join(l) + " |" for l in lines]
    return "\n".join(out)
f2 = lambda v: f"{float(v):.2f}"
f3 = lambda v: f"{float(v):.3f}"
NAMES = {"R5 (round-5 file)": "R5（第五轮文件）", "R6 (candidate)": "**R6（候选）**", "R6 raw anchor": "R6，锚点不校正",
         "R6 base only": "R6，不加锚点", "R6 base, all libraries": "R6，基底换成全部文库", "R5 base only": "R5，不加锚点",
         "R5 base, harmonised anchor": "R5 基底 + 校正锚点"}
SCEN = ["heart/deep", "heart/mid", "heart/shallow", "all/asis"]
PROX = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all", "D/raw", "D/dm", "raw", "h3", "dm"]
W = {"heart/deep": .48, "heart/mid": .16, "heart/shallow": .16, "all/asis": .2}
for split in ("development", "confirmation"):
    rows = load(f"round6_final__scen_{split}_rho0.75_trend0.csv")
    tab = collections.defaultdict(dict)
    for r in rows: tab[r["config"]][r["scenario"]] = r
    print(f"\n### scen {split}\n")
    print(md(["配置", "心脏/深", "心脏/中", "心脏/浅", "同 E10.5 全部文库", "加权"],
             [[NAMES[c]] + [f2(tab[c][s]["score"]) for s in SCEN] + [f2(sum(W[s] * float(tab[c][s]["score"]) for s in SCEN))] for c in NAMES]))
    for m, lab in (("skill_de_score", "DES"), ("skill_de_direction", "DCS"), ("skill_mmd_u", "MMD"), ("skill_variogram", "CSS")):
        print(f"\n{lab} 技能分（{split}）\n")
        print(md(["配置"] + ["心脏/深", "心脏/中", "心脏/浅", "同 E10.5 全部文库"], [[NAMES[c]] + [f3(tab[c][s][m]) for s in SCEN] for c in NAMES]))
    rows = load(f"round6_final__proxy_{split}_rho0.75_trend0_*.csv")
    tab = collections.defaultdict(dict)
    for r in rows: tab[r["config"]][r["proxy"]] = r
    print(f"\n### proxy {split}\n")
    lines = [[NAMES[c]] + [f2(tab[c][p]["score"]) for p in PROX] for c in NAMES]
    lines.append(["R6 − R5"] + [f"{float(tab['R6 (candidate)'][p]['score']) - float(tab['R5 (round-5 file)'][p]['score']):+.2f}" for p in PROX])
    lines.append(["（R5 基底 + 校正锚点）− R5"] + [f"{float(tab['R5 base, harmonised anchor'][p]['score']) - float(tab['R5 (round-5 file)'][p]['score']):+.2f}" for p in PROX])
    lines.append(["锚点比例"] + [f"{float(tab['R5 (round-5 file)'][p]['f']):.3f}" for p in PROX])
    print(md(["配置"] + PROX, lines))
    d = [float(tab['R5 base, harmonised anchor'][p]['score']) - float(tab['R5 (round-5 file)'][p]['score']) for p in ("gen", "p8/deep", "p8/all", "2d/deep", "2d/all", "raw", "h3", "dm")]
    print("\ncondition 2 mean", round(sum(d) / len(d), 3), "min", round(min(d), 3))
    for m, lab in (("skill_de_score", "DES"), ("skill_de_direction", "DCS"), ("skill_mmd_u", "MMD"), ("skill_variogram", "CSS")):
        print(f"\n{lab} 技能分（{split}）\n")
        print(md(["配置"] + PROX, [[NAMES[c]] + [f3(tab[c][p][m]) for p in PROX] for c in NAMES]))
# sweep 1
rows = load("round6_s1__scores_development_*.csv")
main = [r for r in rows if not r.get("noise_seed")]
print("\n### s1\n")
PX = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all", "D/raw", "D/dm"]
FR = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.375", "0.45"]
for base in ("oracle", "nat", "r4", "heart"):
    for h in ("0.0", "0.5", "1.0"):
        t = {(r["proxy"], r["f"]): r["score"] for r in main if r["base"] == base and (r["h"] == h or r["f"] == "0.0")}
        if base == "heart" and h != "0.0": continue
        print(f"\n基底 {base}，校正强度 {h}\n")
        print(md(["代理"] + FR, [[p] + [f2(t[(p, f)]) for f in FR] for p in PX]))
noise = collections.defaultdict(list)
for r in rows:
    if r["base"] == "nat" and r["h"] == "0.0" and r["f"] in ("0.0", "0.2"): noise[(r["proxy"], r["f"])].append(float(r["score"]))
import statistics as st
print("\nnoise sd (nat base, 5 seeds):", {k: round(st.stdev(v), 2) for k, v in noise.items() if len(v) >= 5})
