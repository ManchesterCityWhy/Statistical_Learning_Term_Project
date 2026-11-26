import sys, glob
import pandas as pd
from pathlib import Path
pd.set_option("display.width", 260, "display.max_rows", 500, "display.max_columns", 50)
base = Path.home() / "statlearn_real/runs/round7_probe"
files = sorted(glob.glob(str(base / sys.argv[1])))
frame = pd.concat([pd.read_csv(f) for f in files])
summary = []
for setting, sub in frame.groupby("setting", sort=False):
    sub = sub.copy()
    print(f"\n=== {setting}")
    cols = [c for c in ("actual", "D", "D_seed_se", "probe_total", "skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram", "extra_total") if c in sub]
    print(sub.set_index("recipe")[cols].round(4).sort_values("D").to_string())
    if "actual" in sub:
        sub = sub[~sub["recipe"].str.startswith("ref ")]
        best = sub.loc[sub["D"].idxmin()]
        rho = sub[["actual", "D"]].corr(method="spearman").iloc[0, 1]
        rank = int((sub["actual"] > best["actual"]).sum()) + 1
        summary.append(dict(setting=setting, picked=best["recipe"], picked_actual=round(best["actual"], 2), rank=f"{rank}/{len(sub)}", best=round(sub["actual"].max(), 2),
                            regret=round(sub["actual"].max() - best["actual"], 2), median=round(sub["actual"].median(), 2), spearman=round(rho, 3)))
if summary:
    print("\nSUMMARY"); print(pd.DataFrame(summary).to_string(index=False))
