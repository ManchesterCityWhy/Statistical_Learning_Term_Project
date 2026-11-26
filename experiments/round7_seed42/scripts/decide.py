"""Rule written at 10:02 / 10:11: per configuration, gain over R7h on the five proxies (development, confirmation),
change at zero elapsed time (library 3 / deep), and the sum 'value' = mean gain + zero-elapsed change."""
import pandas as pd, sys, os
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 66)
REFNAME = "R7h: file recipe (two-witness + trend)"
R = "/home/andreib180/statlearn_real/runs/round7_eval/"
out = []
for grid, n in [a.split(":") for a in sys.argv[1:]]:
    dv, cf = pd.read_csv(f"{R}proxy_{grid}_development_dev{n}.csv"), pd.read_csv(f"{R}proxy_{grid}_confirmation_conf{n}.csv")
    sc = pd.read_csv(f"{R}scen_{grid}_development_dev{n}.csv") if os.path.exists(f"{R}scen_{grid}_development_dev{n}.csv") else None
    def gain(d):
        t = d.pivot_table(index="config", columns="proxy", values="score", sort=False)
        return (t - t.loc[REFNAME]).mean(axis=1, skipna=False)
    def skills(d):
        m = d.groupby("config", sort=False)[["skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"]].mean() * 100
        return (m - m.loc[REFNAME])
    g1, g2 = gain(dv), gain(cf)
    sk = (skills(dv) + skills(cf)) / 2
    f = pd.DataFrame({"grid": grid, "gain dev": g1.round(2), "gain conf": g2.round(2)})
    f[["dDES", "dDCS", "dMMD", "dCSS"]] = sk.round(1)
    if sc is not None:
        z = sc[sc.scenario == "heart/deep"].groupby("config", sort=False)
        zs = z.score.mean()
        f["zero-gap"] = zs.round(2)
        f["zero d"] = (zs - zs.loc[REFNAME]).round(2)
        zk = z[["skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"]].mean() * 100
        f[["zDES", "zDCS", "zMMD", "zCSS"]] = (zk - zk.loc[REFNAME]).round(1)
        f["value"] = ((g1 + g2) / 2 + (zs - zs.loc[REFNAME])).round(2)
    f["valid"] = (g1 >= 0.3) & (g2 >= 0.3)
    out.append(f)
print(pd.concat(out).to_string())
