import pandas as pd, numpy as np
from pathlib import Path
pd.set_option("display.width", 250)
base = Path.home() / "statlearn_real/runs/round7_regime"
for split in ("development", "confirmation"):
    d = pd.read_csv(base / f"scores_{split}.csv")
    t = d.pivot_table(index="config", columns="w", values="score", sort=False)
    print("\n##", split); print(t.round(2).to_string())
    b = t.loc["base only"]
    for lo in (77.0, 79.0, 81.0):
        # linear interpolation in the base-only score
        ws = b.index.to_numpy(); bs = b.to_numpy()
        w_at = np.interp(lo, bs, ws)
        vals = {c: float(np.interp(w_at, ws, t.loc[c].to_numpy())) for c in t.index}
        print(f"  base-only {lo:.0f} -> w {w_at:.3f}: raw {vals['raw nuclei']:.2f}, moved back 1 {vals['re-sequenced, moved back 1']:.2f} (diff {vals['re-sequenced, moved back 1'] - vals['raw nuclei']:+.2f}), "
              f"not moved back {vals['re-sequenced, not moved back']:.2f} (diff to raw {vals['re-sequenced, not moved back'] - vals['raw nuclei']:+.2f}), thinned {vals['base, 18% of the cells thinned to the anchor depth']:.2f}")
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(d.pivot_table(index="config", columns="w", values=m, sort=False).round(3).to_string())
