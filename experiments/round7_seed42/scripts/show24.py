import pandas as pd, glob, numpy as np
from pathlib import Path
pd.set_option("display.width", 250, "display.max_rows", 100)
base = Path.home() / "statlearn_real/runs"
TG = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all"]
res = {}
for split, tag in (("development", "dev"), ("confirmation", "conf")):
    p = pd.read_csv(base / f"round7_eval/proxy_g12_{split}_{tag}.csv")
    s = pd.read_csv(base / f"round7_eval/scen_g12_{split}.csv")
    pp = p.pivot_table(index="config", columns="proxy", values="score", sort=False)[TG]
    ss = s.pivot_table(index="config", columns="scenario", values="score", sort=False)
    print(f"\n##### {split}")
    print(pp.round(2).to_string()); print(ss.round(2).to_string())
    S = (0.85 * pp.mean(axis=1) + 0.15 * ss["heart/deep"]).round(3)
    print("mean of five proxies:", pp.mean(axis=1).round(3).to_dict()); print("S:", S.to_dict())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m)
        print(p.pivot_table(index="config", columns="proxy", values=m, sort=False)[TG].round(3).to_string())
        print(s.pivot_table(index="config", columns="scenario", values=m, sort=False).round(3).to_string())
for f in sorted(glob.glob(str(base / "round7_clean2/scores_tilt_*.csv"))):
    d = pd.read_csv(f)
    print("\n###", Path(f).name)
    print(d.pivot_table(index="config", columns="target", values="score", sort=False).round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(d.pivot_table(index="config", columns="target", values=m, sort=False).round(3).to_string())
