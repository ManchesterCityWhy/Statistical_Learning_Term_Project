import pandas as pd, sys, os
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)
sk = {"skill_de_score": "DES", "skill_de_direction": "DCS", "skill_mmd_u": "MMD", "skill_variogram": "CSS"}
REFNAME = "R7h: file recipe (two-witness + trend)"
grid = sys.argv[1] if len(sys.argv) > 1 else "g15"
tags = sys.argv[2].split(",") if len(sys.argv) > 2 else ["dev15", "conf15"]
show = sys.argv[3].split("|") if len(sys.argv) > 3 else None
for split, tag in zip(("development", "confirmation"), tags):
    d = pd.read_csv(f"/home/andreib180/statlearn_real/runs/round7_eval/proxy_{grid}_{split}_{tag}.csv")
    for c, nm in sk.items():
        t = d.pivot_table(index="config", columns="proxy", values=c, sort=False) * 100
        diff = (t - t.loc[REFNAME]).round(1)
        if show: diff = diff.loc[[s for s in show if s in diff.index]]
        print(f"== {split} {nm}: minus R7h"); print(diff.to_string())
p = f"/home/andreib180/statlearn_real/runs/round7_eval/scen_{grid}_development_{tags[0]}.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    t = d.pivot_table(index="config", columns="scenario", values="score", sort=False)
    print("== zero elapsed time (development): score"); print(t.round(2).to_string())
    for c, nm in sk.items():
        t = d.pivot_table(index="config", columns="scenario", values=c, sort=False) * 100
        print(nm); print(t.round(1).to_string())
