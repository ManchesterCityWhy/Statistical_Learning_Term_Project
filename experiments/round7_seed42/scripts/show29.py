import pandas as pd, numpy as np
from pathlib import Path
pd.set_option("display.width", 250)
base = Path.home() / "statlearn_real/runs/round7_regime"
for split in ("development", "confirmation"):
    d = pd.read_csv(base / f"scores_tilt_f0.45_{split}.csv")
    t = d.pivot_table(index="config", columns="w", values="score", sort=False)
    print("##", split); print(t.round(2).to_string())
    b = t.loc["base only"]; ws = b.index.to_numpy(); bs = b.to_numpy()
    for lvl in (77.0, 79.0, 81.0):
        w_at = np.interp(lvl, bs, ws)
        v = {c: float(np.interp(w_at, ws, t.loc[c].to_numpy())) for c in t.index}
        r0 = v["re-sequenced, moved back 1"]
        print(f"  base-only {lvl:.0f} (w {w_at:.3f}): raw {v['raw nuclei']:.2f}, tilt 0 {r0:.2f}, tilt 1.5 {v['re-sequenced, moved back 1, tilt 1.5'] - r0:+.2f}, tilt 3 {v['re-sequenced, moved back 1, tilt 3'] - r0:+.2f}, tilt 0 minus raw {r0 - v['raw nuclei']:+.2f}")
