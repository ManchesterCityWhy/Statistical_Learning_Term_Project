import pandas as pd, numpy as np
from pathlib import Path
pd.set_option("display.width", 250, "display.max_rows", 100)
base = Path.home() / "statlearn_real/runs"
TG = ["gen", "p8/deep", "p8/all", "2d/deep", "2d/all"]
F, R, H = "R7 (final): synthetic nuclei", "R7 with all nuclei replaced by real cells", "R7 with half of the nuclei replaced by real cells"
for split, tag in (("development", "dev14"), ("confirmation", "conf14")):
    p = pd.read_csv(base / f"round7_eval/proxy_g14_{split}_{tag}.csv"); s = pd.read_csv(base / f"round7_eval/scen_g14_{split}_{tag}.csv")
    pp = p.pivot_table(index="config", columns="proxy", values="score", sort=False)[TG]
    ss = s.pivot_table(index="config", columns="scenario", values="score", sort=False)
    print(f"\n##### {split} (mean of {p.seed.nunique()} seeds)"); print(pp.round(2).to_string()); print(ss.round(2).to_string())
    S = 0.85 * pp.mean(axis=1) + 0.15 * ss["heart/deep"]
    print("S:", S.round(2).to_dict())
    w = p.pivot_table(index=["seed", "proxy"], columns="config", values="score"); z = s.pivot_table(index=["seed", "scenario"], columns="config", values="score")
    for name, other in (("all real", R), ("half real", H)):
        print(f"{name} minus final, per proxy (mean, sd over seeds):", (w[other] - w[F]).groupby(level="proxy").agg(["mean", "std"]).round(2).T.to_dict())
        print(f"{name} minus final, zero-gap:", (z[other] - z[F]).groupby(level="scenario").agg(["mean", "std"]).round(2).T.to_dict())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(p.pivot_table(index="config", columns="proxy", values=m, sort=False)[TG].round(3).to_string()); print(s.pivot_table(index="config", columns="scenario", values=m, sort=False).round(3).to_string())
for split in ("development", "confirmation"):
    d = pd.read_csv(base / f"round7_regime/scores_real_f0.45_{split}.csv")
    t = d.pivot_table(index="config", columns="w", values="score", sort=False)
    print("\n## regime, share 45%,", split); print(t.round(2).to_string())
    b = t.loc["base only"]; ws = b.index.to_numpy(); bs = b.to_numpy()
    ref = "re-sequenced, moved back 1"
    for lvl in (77.0, 79.0, 81.0):
        w_at = np.interp(lvl, bs, ws)
        v = {c: float(np.interp(w_at, ws, t.loc[c].to_numpy())) for c in t.index}
        print(f"  base-only {lvl:.0f} (w {w_at:.3f}): synthetic {v[ref]:.2f}; " + "; ".join(f"{c}: {v[c] - v[ref]:+.2f}" for c in t.index if c not in (ref, "base only")))
