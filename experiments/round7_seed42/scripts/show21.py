import pandas as pd, glob
from pathlib import Path
pd.set_option("display.width", 250, "display.max_rows", 100)
base = Path.home() / "statlearn_real/runs/round7_clean2"
for f in sorted(glob.glob(str(base / "scores_*.csv"))):
    d = pd.read_csv(f)
    print("\n###", Path(f).name, len(d), "rows")
    print(d.pivot_table(index="config", columns="target", values="score", sort=False).round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(d.pivot_table(index="config", columns="target", values=m, sort=False).round(3).to_string())
