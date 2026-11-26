"""Round 8, afternoon of 11-26: known-answer calibration WITH elapsed time (setting p8s of round7_probe) for the dimensions
depth and mixture ratio (and, for reference, trend strength and added share).

Reference E8.5; source = E9.5 rebuilt with the test's library structure (pseudo_library_source); truth = a half of E10.5
library 3 ("p8s/deep") or of all E10.5 libraries ("p8s/all"); probe = the h1 method re-run on that source. One day elapses out
of five to the anchor (elapsed 0.2), so every test-time setting is scaled: share-type settings by 0.2 / 0.5, the thinning
factor to the power 1/2 (one day instead of two).

For every variant: its actual score against the truth, and D from the probe's actual skills with the variant as the
hypothetical truth. Question: does "smallest D" pick variants that really score well?

    python -m experiments.round8_calib3 <tag> <targets> <variant keys or all> <seeds> [development|confirmation]
"""
import copy
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import h3_rows, split_rows
from experiments.round6_common import REAL, SEED, FastEval, Mem, depth_of, thin_rows
from experiments.round7_method import Anchor7, predict7
from experiments.round7_probe import h1_like, pseudo_library_source
from t1_pipeline.data import board_spec

W = {"skill_de_score": .25, "skill_de_direction": .25, "skill_mmd_u": .30, "skill_variogram": .20}
tag, targets = sys.argv[1], sys.argv[2].split(",")
seeds = [int(v) for v in sys.argv[4].split(",")]
split = sys.argv[5] if len(sys.argv) > 5 else "confirmation"
a, b = ("development", "ceiling") if split == "development" else ("confirmation", "confirmation_ceiling")
t0 = time.time()
_, genes = board_spec("T1:val")
trend, _ = load_trend(genes)
ax, _ = load_anchor(genes, verbose=False)
anchor = Anchor7(ax)
e8 = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
split10, h3 = split_rows(e9, e10), h3_rows(e10)
ref = e8.dense(np.sort(np.random.default_rng(SEED).choice(e8.n_cells, round(.1 * e8.n_cells), replace=False)))
src = pseudo_library_source(e9, e10)
pool = np.arange(src.n_cells)
probe, probe_rows = h1_like(src, pool, trend, 2000)
heart = np.flatnonzero(src.lib == "3")
free = np.setdiff1d(heart, probe_rows)
prev = (e8, np.arange(e8.n_cells))
E = 0.2
print("pseudo source: heart library", len(heart), "free of the probe", len(free), f"{time.time() - t0:.0f}s", flush=True)
_THIN = {}


def thinned(s):
    if s >= 1.0:
        return src
    key = round(float(s), 4)
    if key not in _THIN:
        rng = np.random.default_rng(900 + int(round(s * 1000)))
        out = copy.copy(src)
        other = np.setdiff1d(np.arange(src.n_cells), heart)
        t = thin_rows(src.X[heart], s * depth_of(src.X[heart]), rng)
        back = np.empty(src.n_cells, dtype=np.int64)
        back[np.concatenate([other, heart])] = np.arange(src.n_cells)
        out.X = sparse.vstack([src.X[other], t]).tocsr()[back].astype(np.float32).tocsr()
        _THIN[key] = out
    return _THIN[key]


def make(rows, n, seed, depth=1.0, f=0.45, ts=0.4, mix=0.25, real=1.0, dev=0.5):
    s = thinned(depth ** 0.5)
    kw = dict(prev=prev, days_to_anchor=5.0, dev=dev / .5 * E, trend=trend, trend_share=ts / .4 * E, seed=seed, level="type")
    if mix:
        kw.update(mix=mix / .5 * E, mix_low=0.3)
    if f > 0:
        return predict7(s, rows, anchor, n, f / .5 * E, bwd=1.0, harmonise=1.0, pool_k="resample", k_prof=20, real_share=real, **kw)
    return predict7(s, rows, anchor, n, 0.0, **kw)


H = {}
for d in (1.0, .85, .70):
    for m in (0.0, .25, .5, .75, 1.0):
        H[f"depth{d:g} mix{m:g}"] = dict(depth=d, mix=m)
H.update({"f0.6": dict(f=.6), "f0.2": dict(f=.2), "trend0.2": dict(ts=.2), "trend0.8": dict(ts=.8), "R7h": dict(mix=0.0, real=.5),
          "depth0.55 mix0.25": dict(depth=.55, mix=.25), "depth0.55 mix0.5": dict(depth=.55, mix=.5)})
keys = list(H) if sys.argv[3] == "all" else sys.argv[3].split(";")
out = REAL / "runs" / "round8_probe2"
out.mkdir(parents=True, exist_ok=True)
records = []
half = 1249
for name in targets:
    rows_t = h3 if name == "p8s/deep" else split10
    ev = FastEval(ref, e10.dense(rows_t[a]), e10.dense(rows_t[b]))
    obs = ev(probe)
    print(f"[{name}] probe actual {obs['score']:.2f}  " + " ".join(f"{k[6:]} {obs[k]:.3f}" for k in W), f"{time.time() - t0:.0f}s", flush=True)
    for k in keys:
        actual = ev(make(heart, 2000, SEED, **H[k]))
        rows = []
        for seed in seeds:
            cells = make(free, 2 * half, 1000 + seed, **H[k])
            order = np.random.default_rng(seed).permutation(len(cells))
            rows.append(FastEval(ref, cells[order[:half]], cells[order[half:]])(probe))
        mean = {m: float(np.mean([r[m] for r in rows])) for m in W}
        d = float(sum(w * (mean[m] - obs[m]) ** 2 for m, w in W.items()))
        per = [float(sum(w * (r[m] - obs[m]) ** 2 for m, w in W.items())) for r in rows]
        se = float(np.std(per, ddof=1) / np.sqrt(len(per))) if len(per) > 1 else float("nan")
        records.append({"target": name, "variant": k, "actual": actual["score"], **{"actual_" + m[6:]: actual[m] for m in W}, "D": d, "D_se": se,
                        "probe_if_true": float(np.mean([r["score"] for r in rows])), **{"probe_" + m[6:]: mean[m] for m in W}})
        print(f"[{name}] {k:<20} actual {actual['score']:6.2f} (" + " ".join(f"{actual[m]:.3f}" for m in W) + f")  D {d:.5f} (se {se:.5f})  probe-if-true "
              + " ".join(f"{mean[m]:.3f}" for m in W) + f"  {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(records).to_csv(out / f"calib3_{tag}.csv", index=False)
    sub = pd.DataFrame([r for r in records if r["target"] == name])
    best = sub.loc[sub["D"].idxmin()]
    rho = sub[["actual", "D"]].corr(method="spearman").iloc[0, 1]
    print(f"[{name}] smallest D: '{best['variant']}' actual {best['actual']:.2f}; best actual {sub['actual'].max():.2f} ('{sub.loc[sub['actual'].idxmax(), 'variant']}'); "
          f"K0 actual {float(sub.loc[sub['variant'] == 'depth1 mix0.25', 'actual'].iloc[0]) if (sub['variant'] == 'depth1 mix0.25').any() else float('nan'):.2f}; Spearman(actual, D) {rho:.3f}", flush=True)
print(f"done {time.time() - t0:.0f}s")
