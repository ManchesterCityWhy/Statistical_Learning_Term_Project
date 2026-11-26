"""Round 7: what each of the two exported files would score under several hypothetical E12.5 populations.

No returned score is used. Hypothetical truths are built from E10.5 cells that neither file contains and from
anchor nuclei the round-7 file did not pick; each truth has 2 x 638 cells (truth half, ceiling half), the
reference is the E9.5 sample of the test, three seeds.

    T0  library 3 as it is
    T1  library-3 cells, expression moved (two-witness 0.5 + bulk trend 0.4), natural composition
    T2  library-3 cells drawn to the interpolated lineage composition (tc = 0.45), expression unchanged
    T3  T2 with the expression moved as in T1
    T4  the round-7 construction itself with another seed (45% re-sequenced nuclei, both shifts)
Files: the exported round-7 file and the round-6 file (R6b), read from disk.
"""
import numpy as np, pandas as pd, anndata as ad
from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import split_rows
from experiments.round6_common import REAL, FastEval, Mem, depth_of
from experiments.round6_s1 import feasible_shares
from experiments.round7_export import build
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

def read(name):
    a = ad.read_h5ad(REAL / "predictions" / name / "E12.5_prediction.h5ad")
    return np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float32)

r7 = read("round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4")
r6 = read("round6_heartlib_anchor0.2_harm1_trend0")
# rows of E10.5 used by the files: round 7 from its own build, round 6 by matching its unchanged library-3 cells
cells, info, _, main, _, _ = build(e10, ax, 0.45, 0.0, 5000, return_info=True, level="type", dev=0.5, prev=prev, days=4.0, trend=trend, trend_share=0.4, k_prof=20)
assert np.array_equal(cells, r7), "rebuilt round-7 cells differ from the file"
used7, nuclei7 = info["base_rows"], info["anchor_rows"]
x3 = e10.X[lib3]
key = lambda nnz, tot, mx: (int(nnz), round(float(tot), 3), round(float(mx), 4))
table = {key(n, s, m): r for r, n, s, m in zip(lib3, np.diff(x3.indptr), np.asarray(x3.astype(np.float64).sum(axis=1)).ravel(), x3.max(axis=1).toarray().ravel())}
n6 = int(round(len(r6) * 0.8))
used6 = np.array([table.get(key((row > 0).sum(), row.astype(np.float64).sum(), row.max()), -1) for row in r6[:n6]])
print("round-6 base rows matched:", int((used6 >= 0).sum()), "of", n6, "| distinct", len(np.unique(used6[used6 >= 0])), flush=True)
assert (used6 >= 0).mean() > 0.99
# the candidate "half of the nuclei replaced by real cells": the round-7 configuration with real_share = 0.5
anchor_h = Anchor7(ax)
half, info_h = predict7(e10, main, anchor_h, 5000, 0.45, fwd=0.0, bwd=1.0, harmonise=1.0, pool_k="resample", level="type", return_info=True,
                        trend=trend, trend_share=0.4, k_prof=20, prev=prev, days_to_anchor=4.0, dev=0.5, real_share=0.5)
assert np.array_equal(info_h["base_rows"][:info_h["n_drawn_base"]], used7), "the drawn base of the half-real file differs from the round-7 base"
print("half-real file: real cells", len(info_h["base_rows"]), "synthetic nuclei", len(info_h["anchor_rows"]), "distinct E10.5 rows", len(np.unique(info_h["base_rows"])), flush=True)
usedh = info_h["base_rows"]
free = np.setdiff1d(lib3, np.concatenate([used7, usedh, used6[used6 >= 0]]))
free_nuclei = np.setdiff1d(np.arange(ax.shape[0]), nuclei7)
print("library-3 cells free for the hypothetical truths:", len(free), "| anchor nuclei free:", len(free_nuclei), flush=True)

def truth(kind, seed):
    anchor = Anchor7(ax)
    shifts = dict(prev=prev, days_to_anchor=4.0, dev=0.5, trend=trend, trend_share=0.4)
    n = 2 * 638
    if kind == "T0":
        return predict7(e10, free, anchor, n, 0.0, seed=seed)
    if kind == "T1":
        return predict7(e10, free, anchor, n, 0.0, seed=seed, **shifts)
    shares = composition_shares(e10, free, anchor, 0.45, 0.0, n, feasible_shares)
    if kind == "T2":
        return predict7(e10, free, anchor, n, 0.0, seed=seed, shares=shares)
    if kind == "T3":
        return predict7(e10, free, anchor, n, 0.0, seed=seed, shares=shares, **shifts)
    if kind == "T4":
        return predict7(e10, free, anchor, n, 0.45, seed=seed, bwd=1.0, harmonise=1.0, pool_k="resample", level="type", k_prof=20,
                        anchor_rows=free_nuclei, **shifts)

names = {"T0": "library 3 as it is", "T1": "real cells, expression moved", "T2": "real cells, composition moved (0.45)",
         "T3": "real cells, composition and expression moved", "T4": "round-7 construction, other seed"}
rows = []
for kind in names:
    for seed in (101, 102, 103):
        t = truth(kind, seed)
        order = np.random.default_rng(seed).permutation(len(t))
        ev = FastEval(ref, t[order[:638]], t[order[638:]])
        for fname, f in (("round 7 file", r7), ("round 6 file", r6), ("round 7, half of the nuclei real", half)):
            s = ev(f)
            rows.append(dict(truth=names[kind], file=fname, seed=seed, score=s["score"], DES=s["skill_de_score"], DCS=s["skill_de_direction"], MMD=s["skill_mmd_u"], CSS=s["skill_variogram"]))
    print(kind, "done", flush=True)
frame = pd.DataFrame(rows)
out = REAL / "runs" / "round7_payoff"
out.mkdir(parents=True, exist_ok=True)
frame.to_csv(out / "payoff_half.csv", index=False)
g = frame.groupby(["truth", "file"], sort=False)[["score", "DES", "DCS", "MMD", "CSS"]].mean()
print(g.round(3).to_string())
p = frame.pivot_table(index=["truth", "seed"], columns="file", values="score", sort=False)
d = (p["round 7 file"] - p["round 6 file"]).groupby(level="truth", sort=False).agg(["mean", "std"])
print("round 7 minus round 6:"); print(d.round(2).to_string())
d = (p["round 7, half of the nuclei real"] - p["round 7 file"]).groupby(level="truth", sort=False).agg(["mean", "std"])
print("half-real minus round 7:"); print(d.round(2).to_string())
