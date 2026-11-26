import sys, pandas as pd
pd.set_option("display.width", 250, "display.max_rows", 400)
f = pd.read_csv(sys.argv[1])
key = ["kind", "anchor", "shallow", "f"]
t = f.pivot_table(index=key, columns="scenario", values="score", sort=False)[["deep", "mid", "shallow"]]
t["E(.6,.2,.2)"] = .6 * t["deep"] + .2 * t["mid"] + .2 * t["shallow"]
t["E(.4,.3,.3)"] = .4 * t["deep"] + .3 * t["mid"] + .3 * t["shallow"]
t["min"] = t[["deep", "mid", "shallow"]].min(axis=1)
print(t.round(2).to_string())
for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
    print(m)
    print(f.pivot_table(index=key, columns="scenario", values=m, sort=False)[["deep", "mid", "shallow"]].round(3).to_string())
