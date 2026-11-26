"""README tables for the revised candidate (rho = 0.4), straight from the score CSVs."""
import csv, glob, collections, json
S = "/Volumes/Data/Projects/统计学习大作业/repo/experiments/round6_seed42/scores/"
def load(pat):
    rows = []
    for f in sorted(glob.glob(S + pat)): rows += list(csv.DictReader(open(f)))
    return rows
def md(header, lines):
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"] + ["| " + " | ".join(l) + " |" for l in lines])
f2 = lambda v: f"{float(v):.2f}"; f3 = lambda v: f"{float(v):.3f}"
NAMES = {"R5 (round-5 file)": "R5 的做法", "R6 (candidate)": "**R6（候选，锚点 20%）**", "R6 at round-5 share": "R6，锚点 37.5%（第一版）",
         "R6 raw anchor": "R6，锚点不校正", "R6 raw anchor at round-5 share": "R6，锚点 37.5% 且不校正", "R6 base only": "R6，不加锚点",
         "R6 base, all libraries": "R6，基底换成全部文库", "R5 base only": "R5，不加锚点", "R5 base, harmonised anchor": "R5 基底 + 校正锚点",
         "R5 base, R6 anchor rule": "R5 基底 + 校正锚点，比例按 R6"}
SCEN = ["heart/deep", "heart/mid", "heart/shallow", "all/asis"]
PROX = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all", "D/raw", "D/dm", "raw", "h3", "dm"]
W = {"heart/deep": .48, "heart/mid": .16, "heart/shallow": .16, "all/asis": .2}
W5 = {"heart/deep": .3, "heart/mid": .1, "heart/shallow": .1, "all/asis": .5}
R5, R6, R5H, R6B, R5B, R6O = "R5 (round-5 file)", "R6 (candidate)", "R5 base, harmonised anchor", "R6 base only", "R5 base only", "R6 at round-5 share"
out = {}
for split in ("development", "confirmation"):
    sc = collections.defaultdict(dict)
    for r in load(f"round6_final__scen_{split}_rho0.4_trend0.csv"): sc[r["config"]][r["scenario"]] = r
    px = collections.defaultdict(dict)
    for r in load(f"round6_final__proxy_{split}_rho0.4_trend0_[ab].csv"): px[r["config"]][r["proxy"]] = r
    S_ = lambda c, s: float(sc[c][s]["score"]); P_ = lambda c, p: float(px[c][p]["score"])
    exp = lambda c, w: sum(w[s] * S_(c, s) for s in w)
    print(f"\n### scen {split}\n")
    print(md(["配置", "心脏/深", "心脏/中", "心脏/浅", "同 E10.5 全部文库", "按权重加总"], [[NAMES[c]] + [f2(S_(c, s)) for s in SCEN] + [f2(exp(c, W))] for c in NAMES if c in sc]))
    print("R6-R5 per scenario", [round(S_(R6, s) - S_(R5, s), 2) for s in SCEN], "R6old-R5", [round(S_(R6O, s) - S_(R5, s), 2) for s in SCEN], "R6-R6old", [round(S_(R6, s) - S_(R6O, s), 2) for s in SCEN])
    print("R6 - R6raw", [round(S_(R6, s) - S_("R6 raw anchor", s), 2) for s in SCEN], "R6raw-R5", [round(S_("R6 raw anchor", s) - S_(R5, s), 2) for s in SCEN])
    print(f"\n### proxy {split}\n")
    lines = [[NAMES[c]] + [f2(P_(c, p)) for p in PROX] for c in NAMES if c in px]
    lines.append(["R6 − R5"] + [f"{P_(R6, p) - P_(R5, p):+.2f}" for p in PROX])
    lines.append(["（R5 基底 + 校正锚点）− R5"] + [f"{P_(R5H, p) - P_(R5, p):+.2f}" for p in PROX])
    lines.append(["R6 的锚点比例"] + [f"{float(px[R6][p]['f']):.3f}" for p in PROX])
    print(md(["配置"] + PROX, lines))
    eight = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all", "raw", "h3", "dm"]
    d2 = [P_(R5H, p) - P_(R5, p) for p in eight]
    out[split] = {"1a D/raw": round(P_(R6B, "D/raw") - P_(R5B, "D/raw"), 2), "1a D/dm": round(P_(R6B, "D/dm") - P_(R5B, "D/dm"), 2), "1b gen": round(P_(R6, "gen") - P_(R5, "gen"), 2),
                  "2 mean": round(sum(d2) / 8, 2), "2 min": round(min(d2), 2), "3a": round(exp(R6, W) - exp(R5, W), 2), "3b": round(exp(R6, W5) - exp(R5, W5), 2),
                  "R6 - R6raw proxies": {p: round(P_(R6, p) - P_("R6 raw anchor", p), 2) for p in eight},
                  "R6 - R6old proxies": {p: round(P_(R6, p) - P_(R6O, p), 2) for p in PROX}}
    for m, lab in (("skill_de_score", "DES"), ("skill_de_direction", "DCS"), ("skill_mmd_u", "MMD"), ("skill_variogram", "CSS")):
        print(f"\n{lab}（情景表，{split}）\n"); print(md(["配置"] + SCEN, [[NAMES[c]] + [f3(sc[c][s][m]) for s in SCEN] for c in NAMES if c in sc]))
    for m, lab in (("skill_de_score", "DES"), ("skill_de_direction", "DCS"), ("skill_mmd_u", "MMD"), ("skill_variogram", "CSS")):
        print(f"\n{lab}（代理，{split}）\n"); print(md(["配置"] + PROX, [[NAMES[c]] + [f3(px[c][p][m]) for p in PROX] for c in NAMES if c in px]))
print(json.dumps(out, ensure_ascii=False, indent=1))
d = json.load(open(S + "round6_describe__describe_rho0.4.json"))
for tag in ("round5_file", "round6_file"):
    print(tag, json.dumps(d[tag], ensure_ascii=False))
