import csv, glob, collections
S = "/Volumes/Data/Projects/统计学习大作业/repo/experiments/round6_seed42/scores/"
def load(pat):
    rows = []
    for f in sorted(glob.glob(S + pat)): rows += list(csv.DictReader(open(f)))
    return rows
def md(header, lines):
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"] + ["| " + " | ".join(l) + " |" for l in lines])
f2 = lambda v: f"{float(v):.2f}"
print("\n### TE base only (f=0)\n")
te = {}
for name, pat in (("dev", "round6_te__scores_development.csv"), ("conf", "round6_te__scores_confirmation.csv"), ("harm", "round6_te__scores_development_harmtarget.csv")):
    for r in load(pat):
        te[(name, r["proxy"], r["base"], r["f"])] = r
bases = ["natural:all", "natural:lib12", "natural:lib3", "natural:heart_all", "natural:heart_lib3", "r2:all", "r4:all", "r2:lib3", "r4:lib3", "r2:heart_all", "r4:heart_all", "natural:lib3+bulk3", "natural:lib3+bulk6"]
cols = [("dev", "te/raw"), ("conf", "te/raw"), ("dev", "te/dm"), ("conf", "te/dm"), ("harm", "te-harm/raw")]
print(md(["基底"] + [f"{a} {b}" for a, b in cols], [[b] + [f2(te[(a, p, b, "0.0")]["score"]) for a, p in cols] for b in bases]))
print("\nTE per metric dev te/raw f=0")
print(md(["基底", "DES", "DCS", "MMD", "CSS"], [[b] + [f"{float(te[('dev','te/raw',b,'0.0')][m]):.3f}" for m in ("skill_de_score","skill_de_direction","skill_mmd_u","skill_variogram")] for b in bases]))
print("\n### gen (round6_gen)\n")
g = collections.defaultdict(dict)
for r in load("round6_gen__scores_development.csv"): g[r["base"]][r["f"]] = r
fr = ["0.0", "0.05", "0.1", "0.15", "0.2", "0.3"]
print(md(["基底"] + fr, [[b] + [f2(g[b][f]["score"]) for f in fr] for b in g]))
print(md(["基底 f=0", "DES", "DCS", "MMD", "CSS"], [[b] + [f"{float(g[b]['0.0'][m]):.3f}" for m in ("skill_de_score","skill_de_direction","skill_mmd_u","skill_variogram")] for b in g]))
print("\n### dis\n")
d = collections.defaultdict(dict)
for r in load("round6_dis__scores_development.csv"): d[r["base"]][r["proxy"]] = r
print(md(["基底", "D/raw", "D/dm"], [[b] + [f2(d[b][p]["score"]) for p in ("D/raw", "D/dm")] for b in d]))
print("\n### d2\n")
t = collections.defaultdict(dict)
for r in load("round6_d2__scores_development.csv"): t[(r["kind"], r["anchor"], r["shallow"], r["f"])][r["scenario"]] = r
print(md(["做法", "锚点", "浅文库份额", "锚点比例", "深", "中", "浅"], [[k[0], k[1], k[2], k[3]] + [f2(v[s]["score"]) for s in ("deep", "mid", "shallow")] for k, v in t.items()]))
print("\n### d2 extra\n")
t = collections.defaultdict(dict)
for r in load("round6_d2__scores_development_extra.csv"): t[(r["anchor"], r["trend"], r["shallow"], r["f"])][r["scenario"]] = r
print(md(["校正", "趋势", "浅文库份额", "锚点比例", "深", "中", "浅"], [list(k) + [f2(v[s]["score"]) for s in ("deep", "mid", "shallow")] for k, v in t.items()]))
print("\n### s2 (oracle base)\n")
rows = load("round6_s2__scores_development_*.csv")
PX = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all", "D/raw", "D/dm"]
for base in ("oracle", "nat", "heart"):
    t = collections.defaultdict(dict)
    for r in rows:
        if r["base"] == base: t[(r["mode"], r["trend"], r["f"])][r["proxy"]] = r["score"]
    print(f"\n基底 {base}\n")
    print(md(["校正方式", "趋势强度", "锚点比例"] + PX, [list(k) + [f2(v[p]) for p in PX] for k, v in t.items()]))
print("\n### tilt\n")
rows = load("round6_tilt__scores_development_*.csv")
print(len(rows), "tilt rows")
