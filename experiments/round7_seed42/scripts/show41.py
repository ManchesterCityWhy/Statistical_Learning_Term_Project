import pandas as pd, sys, os
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 62)
sk = {"skill_de_score": "DES", "skill_de_direction": "DCS", "skill_mmd_u": "MMD", "skill_variogram": "CSS"}
REFNAME = "R7h: file recipe (two-witness + trend)"
grid, tags = sys.argv[1], sys.argv[2].split(",")
for split, tag in zip(("development", "confirmation"), tags):
    d = pd.read_csv(f"/home/andreib180/statlearn_real/runs/round7_eval/proxy_{grid}_{split}_{tag}.csv")
    t = d.pivot_table(index="config", columns="proxy", values="score", sort=False)
    diff = t - t.loc[REFNAME]
    diff["mean5"] = diff[[c for c in ("gen", "p8/deep", "p8/all", "2d/deep", "2d/all") if c in diff]].mean(axis=1, skipna=False)
    diff["deep2"] = diff[[c for c in ("p8/deep", "2d/deep") if c in diff]].mean(axis=1)
    m = d.groupby(["config", "proxy"], sort=False)[list(sk)].mean() * 100
    mm = m.groupby(level="config", sort=False).mean(); mm.columns = ["d" + sk[c] for c in mm.columns]
    mm = mm - mm.loc[REFNAME]
    print(f"== {split}: minus R7h (score by proxy; skills = mean over the proxies the config has)")
    print(pd.concat([diff.round(2), mm.round(1)], axis=1).to_string())
p = f"/home/andreib180/statlearn_real/runs/round7_eval/scen_{grid}_development_{tags[0]}.csv"
if os.path.exists(p):
    d = pd.read_csv(p)
    t = d.pivot_table(index="config", columns="scenario", values="score", sort=False)
    hd = d[d.scenario == "heart/deep"].groupby("config", sort=False)[list(sk)].mean() * 100
    hd.columns = ["hd " + sk[c] for c in hd.columns]
    print("== zero elapsed time (development): score by scenario, skills on heart/deep"); print(pd.concat([t.round(2), hd.round(1)], axis=1).to_string())
