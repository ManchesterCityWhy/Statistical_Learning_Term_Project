import pandas as pd, numpy as np
from pathlib import Path
pd.set_option("display.width", 250)
base = Path.home() / "statlearn_real/runs/round7_regime"
for split in ("development", "confirmation"):
    d = pd.read_csv(base / f"scores_k_f0.45_{split}.csv")
    t = d.pivot_table(index="config", columns="w", values="score", sort=False)
    print("##", split); print(t.round(2).to_string())
    b = t.loc["base only"]; ws = b.index.to_numpy(); bs = b.to_numpy()
    ref = "re-sequenced, moved back 1, k_prof 20"
    for lvl in (77.0, 79.0, 81.0):
        w_at = np.interp(lvl, bs, ws)
        v = {c: float(np.interp(w_at, ws, t.loc[c].to_numpy())) for c in t.index}
        print(f"  base-only {lvl:.0f} (w {w_at:.3f}): k20 {v[ref]:.2f}; " + "; ".join(f"{c.split('k_prof ')[-1] if 'k_prof' in c else c}: {v[c] - v[ref]:+.2f}" for c in t.index if c not in (ref, "base only")))
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(d.pivot_table(index="config", columns="w", values=m, sort=False).round(3).to_string())
