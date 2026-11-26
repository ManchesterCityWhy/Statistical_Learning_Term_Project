"""Round 7: what each recipe would score under several hypothetical E12.5 populations (clean design).

Library 3 is split at random: one third is reserved for the hypothetical truths, the recipes are built from the
other two thirds (so no cell is shared and neither pool is depleted of particular cell types); the anchor nuclei are
split 40% / 60% the same way. No returned score is used. Reference: the E9.5 sample of the test. Three seeds.

    recipes  round 7 with all 45% as synthetic nuclei | round 7 with half of them replaced by real cells (the file) |
             round 6 file's recipe (20% raw nuclei, no shifts, 4,000 cells) | library-3 cells as they are
    truths   T0 library 3 as it is; T1 real cells, expression moved (two-witness 0.5 + bulk trend 0.4);
             T2 real cells drawn to the interpolated lineage composition (0.45); T3 = T2 with the expression moved;
             T4 the all-synthetic round-7 construction
"""
import numpy as np, pandas as pd
from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import split_rows
from experiments.round6_common import REAL, FastEval, Mem, lineage_of
from experiments.round6_s1 import feasible_shares
from experiments.round7_method import Anchor7, composition_shares, predict7
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250)
_, genes = board_spec("T1:test")
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
ax, _ = load_anchor(genes, verbose=False)
trend, _ = load_trend(genes)
ref = e9.dense(split_rows(e9, e10)["reference"])
prev = (e9, np.arange(e9.n_cells))
lib3 = np.flatnonzero(e10.lib == "3")
r0 = np.random.default_rng(2026)
perm = r0.permutation(lib3)
pool_t, pool_f = np.sort(perm[:len(lib3) // 3]), np.sort(perm[len(lib3) // 3:])
nperm = r0.permutation(ax.shape[0])
nuc_t, nuc_f = np.sort(nperm[:int(.4 * ax.shape[0])]), np.sort(nperm[int(.4 * ax.shape[0]):])
print("truth pool", len(pool_t), "recipe pool", len(pool_f), "| nuclei", len(nuc_t), len(nuc_f), flush=True)
shifts = dict(prev=prev, days_to_anchor=4.0, dev=0.5, trend=trend, trend_share=0.4)
r7 = dict(bwd=1.0, harmonise=1.0, pool_k="resample", level="type", k_prof=20, anchor_rows=nuc_f, **shifts)
recipes = {
    "round 7, all synthetic": lambda: predict7(e10, pool_f, Anchor7(ax), 5000, 0.45, **r7),
    "round 7, half real (the file)": lambda: predict7(e10, pool_f, Anchor7(ax), 5000, 0.45, real_share=0.5, **r7),
    "round 6 file's recipe": lambda: predict7(e10, pool_f, Anchor7(ax), 4000, 0.2, anchor_rows=nuc_f),
    "library 3 as it is": lambda: predict7(e10, pool_f, Anchor7(ax), 5000, 0.0),
}
files = {k: f() for k, f in recipes.items()}
print("recipes built", {k: v.shape[0] for k, v in files.items()}, flush=True)

def truth(kind, seed):
    anchor = Anchor7(ax)
    n = 2 * 638
    if kind == "T0":
        return predict7(e10, pool_t, anchor, n, 0.0, seed=seed)
    if kind == "T1":
        return predict7(e10, pool_t, anchor, n, 0.0, seed=seed, **shifts)
    if kind == "T4":
        return predict7(e10, pool_t, anchor, n, 0.45, seed=seed, bwd=1.0, harmonise=1.0, pool_k="resample", level="type", k_prof=20, anchor_rows=nuc_t, **shifts)
    shares = composition_shares(e10, pool_t, anchor, 0.45, 0.0, n, feasible_shares)
    return predict7(e10, pool_t, anchor, n, 0.0, seed=seed, shares=shares, **(shifts if kind == "T3" else {}))

names = {"T0": "library 3 as it is", "T1": "real cells, expression moved", "T2": "real cells, composition moved (0.45)",
         "T3": "real cells, composition and expression moved", "T4": "all-synthetic round-7 construction"}
rows = []
for kind in names:
    for seed in (101, 102, 103):
        t = truth(kind, seed)
        order = np.random.default_rng(seed).permutation(len(t))
        ev = FastEval(ref, t[order[:638]], t[order[638:]])
        for fname, f in files.items():
            s = ev(f)
            rows.append(dict(truth=names[kind], recipe=fname, seed=seed, score=s["score"], DES=s["skill_de_score"], DCS=s["skill_de_direction"], MMD=s["skill_mmd_u"], CSS=s["skill_variogram"]))
    print(kind, "done", flush=True)
frame = pd.DataFrame(rows)
out = REAL / "runs" / "round7_payoff"
out.mkdir(parents=True, exist_ok=True)
frame.to_csv(out / "payoff_recipes.csv", index=False)
print(frame.pivot_table(index="truth", columns="recipe", values="score", sort=False).round(2).to_string())
p = frame.pivot_table(index=["truth", "seed"], columns="recipe", values="score", sort=False)
for a, b in (("round 7, half real (the file)", "round 7, all synthetic"), ("round 7, half real (the file)", "round 6 file's recipe"), ("round 7, all synthetic", "round 6 file's recipe")):
    print(f"{a} minus {b}:"); print((p[a] - p[b]).groupby(level="truth", sort=False).agg(["mean", "std"]).round(2).to_string())
for m in ("DES", "DCS", "MMD", "CSS"):
    print(m); print(frame.pivot_table(index="truth", columns="recipe", values=m, sort=False).round(3).to_string())
