import pandas as pd, numpy as np, glob
from pathlib import Path
pd.set_option("display.width", 250)
base = Path.home() / "statlearn_real/runs/round7_regime"
REF, HALF, ALL = "re-sequenced, moved back 1", "nuclei replaced by real cells of the assigned type, share 0.5", "nuclei replaced by real cells of the assigned type, share 1"
for f in sorted(glob.glob(str(base / "scores_real_f0.45_*_seed*.csv"))):
    d = pd.read_csv(f)
    t = d.pivot_table(index="config", columns="w", values="score", sort=False)
    print("\n##", Path(f).name); print(t.round(2).to_string())
    b = t.loc["base only"]; ws = b.index.to_numpy(); bs = b.to_numpy()
    for lvl in (77.0, 79.0, 81.0):
        w_at = np.interp(lvl, bs, ws)
        v = {c: float(np.interp(w_at, ws, t.loc[c].to_numpy())) for c in t.index}
        print(f"  base-only {lvl:.0f} (w {w_at:.3f}): synthetic {v[REF]:.2f}; half real {v[HALF] - v[REF]:+.2f}; all real {v[ALL] - v[REF]:+.2f}")
