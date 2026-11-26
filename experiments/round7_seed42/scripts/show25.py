import pandas as pd, glob, numpy as np
from pathlib import Path
pd.set_option("display.width", 250, "display.max_rows", 100)
base = Path.home() / "statlearn_real/runs/round7_eval"
TG = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all"]
print("######## g12: tilt")
for split, tag in (("development", "devB"), ("confirmation", "confB")):
    p = pd.read_csv(base / f"proxy_g12_{split}_{tag}.csv")
    s = pd.read_csv(base / f"scen_g12_{split}.csv")
    pp = p.pivot_table(index="config", columns="proxy", values="score", sort=False)[TG]
    ss = s.pivot_table(index="config", columns="scenario", values="score", sort=False)
    print(f"\n## {split}"); print(pp.round(2).to_string())
    S = (0.85 * pp.mean(axis=1) + 0.15 * ss["heart/deep"])
    print("mean of five proxies:", pp.mean(axis=1).round(3).to_dict()); print("zero-gap heart/deep:", ss["heart/deep"].round(2).to_dict()); print("S:", S.round(3).to_dict())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(p.pivot_table(index="config", columns="proxy", values=m, sort=False)[TG].round(3).to_string())
print("\n######## g13: shallow cells as such (development, no shifts)")
p = pd.read_csv(base / "proxy_g13_development_dev.csv"); s = pd.read_csv(base / "scen_g13_development.csv")
print(p.pivot_table(index="config", columns="proxy", values="score", sort=False)[TG].round(2).to_string())
print(s.pivot_table(index="config", columns="scenario", values="score", sort=False).round(2).to_string())
for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
    print(m); print(p.pivot_table(index="config", columns="proxy", values=m, sort=False)[TG].round(3).to_string()); print(s.pivot_table(index="config", columns="scenario", values=m, sort=False).round(3).to_string())
