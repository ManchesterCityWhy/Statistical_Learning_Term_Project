import sys, glob
import pandas as pd
from pathlib import Path
pd.set_option("display.width", 260, "display.max_rows", 500, "display.max_columns", 50)
base = Path.home() / "statlearn_real/runs/round7_eval"
pat, detail = sys.argv[1], (len(sys.argv) > 2 and sys.argv[2] == "detail")
files = sorted(glob.glob(str(base / pat)))
frame = pd.concat([pd.read_csv(f) for f in files])
col = "proxy" if "proxy" in frame.columns else "scenario"
print("files:", [Path(f).name for f in files])
t = frame.pivot_table(index="config", columns=col, values="score", sort=False)
print(t.round(2).to_string())
if detail:
    for m, short in (("skill_de_score", "DES"), ("skill_de_direction", "DCS"), ("skill_mmd_u", "MMD"), ("skill_variogram", "CSS")):
        print(short); print(frame.pivot_table(index="config", columns=col, values=m, sort=False).round(3).to_string())
