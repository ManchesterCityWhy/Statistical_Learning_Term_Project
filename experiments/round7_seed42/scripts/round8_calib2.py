"""Round 8, afternoon of 11-26: known-answer calibration of the rule "take the variant with the smallest D" for the new
dimensions (depth, blood share, trend strength), zero-elapsed setting of round7_probe.

Truth = a half of E10.5 library 3 that the source pool does not contain: as it is ("heart/deep"), thinned to the depth of
library 2 ("heart/mid") or of library 1 ("heart/shallow"). Probe = the h1 method re-run on the source pool. For every
variant: its actual score against the truth, and D from the probe's actual skills (the variant as hypothetical truth).

    python -m experiments.round8_calib2 <tag> <scenarios> <variant keys> <seeds> [development|confirmation]
"""
import copy
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import split_rows
from experiments.round6_common import LINEAGES, REAL, SEED, FastEval, Mem, depth_of, lineage_of, thin_rows
from experiments.round6_final import scenario_setup
from experiments.round6_method import lineage_shares
from experiments.round6_s1 import feasible_shares
from experiments.round7_method import Anchor7, predict7
from experiments.round7_probe import h1_like
from t1_pipeline.data import board_spec

W = {"skill_de_score": .25, "skill_de_direction": .25, "skill_mmd_u": .30, "skill_variogram": .20}
tag, scens, keys = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
seeds = [int(v) for v in sys.argv[4].split(",")]
split = sys.argv[5] if len(sys.argv) > 5 else "development"
t0 = time.time()
_, genes = board_spec("T1:val")
trend, _ = load_trend(genes)
ax, _ = load_anchor(genes, verbose=False)
anchor = Anchor7(ax)
e10, pool, _, evs = scenario_setup(split, genes)
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
ref = e9.dense(split_rows(e9, e10)["reference"])
prev = (e9, np.arange(e9.n_cells))
lib3 = np.flatnonzero(e10.lib == "3")
probe, probe_rows = h1_like(e10, pool, trend, 2000)
print("scenario pool", len(pool), "probe library shares", {k: round(float((e10.lib[probe_rows] == k).mean()), 3) for k in "123"}, f"{time.time() - t0:.0f}s", flush=True)
pool3 = pool[e10.lib[pool] == "3"]
free3 = np.setdiff1d(pool3, probe_rows)
_THIN = {}


def thinned(s):
    if s >= 1.0:
        return e10
    if s not in _THIN:
        rng = np.random.default_rng(900 + int(round(s * 100)))
        x3 = e10.X[lib3]
        t = thin_rows(x3, s * depth_of(x3), rng)
        other = np.setdiff1d(np.arange(e10.n_cells), lib3)
        stacked = sparse.vstack([e10.X[other], t]).tocsr()
        back = np.empty(e10.n_cells, dtype=np.int64)
        back[np.concatenate([other, lib3])] = np.arange(e10.n_cells)
        out = copy.copy(e10)
        out.X = stacked[back].astype(np.float32).tocsr()
        _THIN[s] = out
    return _THIN[s]


def make(rows, n, seed, depth=1.0, f=0.45, ts=0.4, mix=0.25, blood=None, real=1.0, plain=False, dev=0.5):
    src = thinned(depth)
    if plain:
        return predict7(src, rows, anchor, n, 0.0, seed=seed)
    kw = dict(prev=prev, days_to_anchor=4.0, dev=dev, trend=trend, trend_share=ts, seed=seed, level="type")
    if mix:
        kw.update(mix=mix, mix_low=0.3)
    shares = None
    if blood is not None:
        nb = n - int(round(n * f))
        s = lineage_shares(src, rows)
        b = min(blood * n / nb, 0.9)
        rest = 1.0 - s["Blood"]
        s = {k: (b if k == "Blood" else v * (1.0 - b) / rest) for k, v in s.items()}
        shares = feasible_shares(src, rows, s, nb)
    if f > 0:
        return predict7(src, rows, anchor, n, f, bwd=1.0, harmonise=1.0, pool_k="resample", k_prof=20, real_share=real, shares=shares, **kw)
    return predict7(src, rows, anchor, n, 0.0, shares=shares, **kw)


H = {
    "K0": {}, "depth0.85": dict(depth=.85), "depth0.70": dict(depth=.70), "depth0.55": dict(depth=.55), "depth0.45": dict(depth=.45),
    "plain": dict(plain=True), "plain0.85": dict(plain=True, depth=.85), "plain0.70": dict(plain=True, depth=.70), "plain0.55": dict(plain=True, depth=.55),
    "blood16.5": dict(blood=.165), "blood25": dict(blood=.25), "f0": dict(f=0.0), "f0.20": dict(f=.2),
    "trend0.6": dict(ts=.6), "trend0.8": dict(ts=.8), "trend0.2": dict(ts=.2), "mix0": dict(mix=0.0), "mix0.5": dict(mix=.5),
    "depth0.85+blood16.5": dict(depth=.85, blood=.165), "depth0.70+blood16.5": dict(depth=.70, blood=.165),
    "depth0.92": dict(depth=.92), "depth0.78": dict(depth=.78), "depth0.62": dict(depth=.62),
}
out = REAL / "runs" / "round8_probe2"
out.mkdir(parents=True, exist_ok=True)
records = []
half = 1249
for scen in scens:
    ev = evs[scen]
    obs = ev(probe)
    print(f"[{scen}] probe actual {obs['score']:.2f}  " + " ".join(f"{k[6:]} {obs[k]:.3f}" for k in W), f"{time.time() - t0:.0f}s", flush=True)
    for k in keys:
        actual = ev(make(pool3, 2000, SEED, **H[k]))
        rows = []
        for seed in seeds:
            cells = make(free3, 2 * half, 1000 + seed, **H[k])
            order = np.random.default_rng(seed).permutation(len(cells))
            rows.append(FastEval(ref, cells[order[:half]], cells[order[half:]])(probe))
        mean = {m: float(np.mean([r[m] for r in rows])) for m in W}
        d = float(sum(w * (mean[m] - obs[m]) ** 2 for m, w in W.items()))
        per = [float(sum(w * (r[m] - obs[m]) ** 2 for m, w in W.items())) for r in rows]
        se = float(np.std(per, ddof=1) / np.sqrt(len(per))) if len(per) > 1 else float("nan")
        records.append({"scenario": scen, "variant": k, "actual": actual["score"], **{"actual_" + m[6:]: actual[m] for m in W}, "D": d, "D_se": se,
                        "probe_if_true": float(np.mean([r["score"] for r in rows])), **{"probe_" + m[6:]: mean[m] for m in W}})
        print(f"[{scen}] {k:<22} actual {actual['score']:6.2f} (" + " ".join(f"{actual[m]:.3f}" for m in W) + f")  D {d:.5f} (se {se:.5f})  probe-if-true "
              + " ".join(f"{mean[m]:.3f}" for m in W) + f"  {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(records).to_csv(out / f"calib2_{tag}.csv", index=False)
print(f"done {time.time() - t0:.0f}s")
