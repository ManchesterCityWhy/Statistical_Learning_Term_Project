import pandas as pd, sys
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)
sk = {"skill_de_score": "DES", "skill_de_direction": "DCS", "skill_mmd_u": "MMD", "skill_variogram": "CSS"}
REFNAME = "R7h: file recipe (two-witness + trend)"
for split, tag in (("development", "dev15"), ("confirmation", "conf15")):
    d = pd.read_csv(f"/home/andreib180/statlearn_real/runs/round7_eval/proxy_g15_{split}_{tag}.csv")
    t = d.pivot_table(index="config", columns="proxy", values="score", sort=False)
    t["mean5"] = t.mean(axis=1)
    print(f"== {split}: score (mean of seeds {sorted(d.seed.unique())})"); print(t.round(2).to_string())
    diff = (t - t.loc[REFNAME]).round(2)
    print("-- minus R7h"); print(diff.to_string())
    m = d.groupby("config", sort=False)[list(sk)].mean() * 100
    m.columns = [sk[c] for c in m.columns]
    print("-- skills, mean over the five proxies; and minus R7h"); print(pd.concat([m.round(1), (m - m.loc[REFNAME]).round(2).add_prefix("d")], axis=1).to_string())
