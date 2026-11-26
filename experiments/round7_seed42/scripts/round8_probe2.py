"""Round 8, afternoon of 11-26: are single-dimension variants of the round-8 recipe consistent with the returned scores of h1?

Every variant K is a complete prediction recipe. If K were the E12.5 truth, h1 would score s_hat(h1 | K); the consistency
distance D(K) = sum_m w_m (s_hat_m - s_obs_m)^2 is the same statistic as in round7_probe. New here: the hypotheses are the
round-8 recipe (all real cells, low-gene mixture, two-witness shift, bulk trend) and variants of it in ONE dimension
(depth of the library-3 cells, blood share, added share f, trend strength, mixture ratio).

Truths are built from one third of the library-3 cells h1 did not use and 40% of the anchor nuclei; files (for the payoff
matrix: what a variant file scores if another variant is the truth) from the other two thirds / 60%.

    python -m experiments.round8_probe2 <tag> <hypothesis keys, comma separated, or all> <seeds> [file keys or none]
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
from experiments.round6_common import LINEAGES, REAL, FastEval, Mem, depth_of, lineage_of, thin_rows
from experiments.round6_method import lineage_shares
from experiments.round6_s1 import feasible_shares
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.data import board_spec

W = {"skill_de_score": .25, "skill_de_direction": .25, "skill_mmd_u": .30, "skill_variogram": .20}
H1_OBSERVED = {"skill_de_score": .663, "skill_de_direction": .769, "skill_mmd_u": .690, "skill_variogram": .705}

tag, which = sys.argv[1], sys.argv[2]
seeds = [int(v) for v in sys.argv[3].split(",")]
file_keys = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "none" else []
split_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 2031
t0 = time.time()
_, genes = board_spec("T1:test")
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
ax, _ = load_anchor(genes, verbose=False)
trend, _ = load_trend(genes)
ref = e9.dense(split_rows(e9, e10)["reference"])
prev = (e9, np.arange(e9.n_cells))
saved = np.load(REAL / "runs" / "round7_h1" / "h1.npz")
probe, probe_rows = saved["cells"], saved["rows"]
lib3 = np.flatnonzero(e10.lib == "3")
free3 = np.setdiff1d(lib3, probe_rows)
r0 = np.random.default_rng(split_seed)
perm = r0.permutation(free3)
pool_t, pool_f = np.sort(perm[:len(free3) // 3]), np.sort(perm[len(free3) // 3:])
nperm = r0.permutation(ax.shape[0])
nuc_t, nuc_f = np.sort(nperm[:int(.4 * ax.shape[0])]), np.sort(nperm[int(.4 * ax.shape[0]):])
anchor_t, anchor_f = Anchor7(ax[nuc_t]), Anchor7(ax[nuc_f])
print("library 3", len(lib3), "free of h1", len(free3), "truth pool", len(pool_t), "file pool", len(pool_f), f"{time.time() - t0:.0f}s", flush=True)

_THIN = {}


def thinned(s):
    """E10.5 with every library-3 cell binomially thinned to the share s of its molecules (other libraries untouched)."""
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
        d0, d1 = np.median(depth_of(x3)), np.median(depth_of(out.X[lib3]))
        g0, g1 = np.median(np.diff(x3.indptr)), np.median(np.diff(out.X[lib3].indptr))
        print(f"thinned x{s:g}: median depth {d0:.0f} -> {d1:.0f}, median detected genes {g0:.0f} -> {g1:.0f}", flush=True)
        _THIN[s] = out
    return _THIN[s]


def make(pool, anchor, n, seed, depth=1.0, f=0.45, ts=0.4, mix=0.25, blood=None, real=1.0, plain=False, dev=0.5, info=None):
    src = thinned(depth)
    if plain:
        return predict7(src, pool, anchor, n, 0.0, seed=seed)
    kw = dict(prev=prev, days_to_anchor=4.0, dev=dev, trend=trend, trend_share=ts, seed=seed, level="type")
    if mix:
        kw.update(mix=mix, mix_low=0.3)
    shares = None
    if blood is not None:
        # lineage quotas of the base so that blood is `blood` of the whole file (the added cells carry almost none)
        nb = n - int(round(n * f))
        s = lineage_shares(src, pool)
        b = min(blood * n / nb, 0.9)
        rest = 1.0 - s["Blood"]
        s = {k: (b if k == "Blood" else v * (1.0 - b) / rest) for k, v in s.items()}
        shares = feasible_shares(src, pool, s, nb)
    if f > 0:
        out = predict7(src, pool, anchor, n, f, bwd=1.0, harmonise=1.0, pool_k="resample", k_prof=20, real_share=real, shares=shares, return_info=True, **kw)
    else:
        out = predict7(src, pool, anchor, n, 0.0, shares=shares, return_info=True, **kw)
    cells, meta = out
    if info is not None:
        lin = lineage_of(src.labels[meta["base_rows"]])
        info.update({"lin_" + k: float((lin == k).mean()) for k in LINEAGES})
        info["real_rows"] = len(meta["base_rows"])
        info["genes_median"] = float(np.median((cells > 0).sum(axis=1)))
    return cells


H = {
    "K0": {},
    "depth0.85": dict(depth=.85), "depth0.70": dict(depth=.70), "depth0.55": dict(depth=.55),
    "blood16.5": dict(blood=.165), "blood25": dict(blood=.25),
    "f0.20": dict(f=.2), "f0": dict(f=0.0),
    "trend0.6": dict(ts=.6), "trend0.8": dict(ts=.8), "trend0.2": dict(ts=.2),
    "mix0": dict(mix=0.0), "mix0.5": dict(mix=.5),
    "refR7h": dict(mix=0.0, real=.5), "refLib3": dict(plain=True),
    # second-step combinations (only run when named)
    "depth0.85+blood16.5": dict(depth=.85, blood=.165), "depth0.70+blood16.5": dict(depth=.70, blood=.165),
    "depth0.85+trend0.6": dict(depth=.85, ts=.6), "depth0.70+trend0.6": dict(depth=.70, ts=.6),
    "blood16.5+trend0.6": dict(blood=.165, ts=.6), "blood25+trend0.6": dict(blood=.25, ts=.6),
    "depth0.85+f0": dict(depth=.85, f=0.0), "depth0.70+f0": dict(depth=.70, f=0.0),
    "depth0.92": dict(depth=.92), "depth0.78": dict(depth=.78), "depth0.62": dict(depth=.62), "depth0.45": dict(depth=.45),
    "blood12.5": dict(blood=.125), "blood20": dict(blood=.20), "blood33": dict(blood=.33),
    "f0.60": dict(f=.6), "f0.30": dict(f=.3),
    "dev0": dict(dev=0.0), "dev1": dict(dev=1.0),
    "depth0.85+mix0.5": dict(depth=.85, mix=.5), "depth0.85+trend0.8": dict(depth=.85, ts=.8), "depth0.85+f0.60": dict(depth=.85, f=.6),
    "mix0.75": dict(mix=.75), "mix1": dict(mix=1.0), "depth0.85+mix0.75": dict(depth=.85, mix=.75), "depth0.85+mix0.35": dict(depth=.85, mix=.35),
    "depth0.85+mix1": dict(depth=.85, mix=1.0), "depth0.78+mix0.5": dict(depth=.78, mix=.5), "depth0.92+mix0.5": dict(depth=.92, mix=.5),
    "mix0.35": dict(mix=.35), "mix0.15": dict(mix=.15),
    "depth0.85+dev1": dict(depth=.85, dev=1.0), "depth0.85+mix0": dict(depth=.85, mix=0.0), "depth0.85+refR7h": dict(depth=.85, mix=0.0, real=.5),
}
keys = list(H) [:15] if which == "all" else which.split(",")
files = {}
for k in file_keys:
    info = {}
    files[k] = make(pool_f, anchor_f, 5000, 42, info=info, **H[k])
    print(f"file {k}: genes median {info.get('genes_median')}, " + " ".join(f"{a[4:]} {v:.3f}" for a, v in info.items() if a.startswith("lin_")), f"{time.time() - t0:.0f}s", flush=True)

out = REAL / "runs" / "round8_probe2"
out.mkdir(parents=True, exist_ok=True)
records = []
for k in keys:
    for seed in seeds:
        info = {}
        cells = make(pool_t, anchor_t, 2 * 638, 1000 + seed, info=info, **H[k])
        order = np.random.default_rng(seed).permutation(len(cells))
        ev = FastEval(ref, cells[order[:638]], cells[order[638:]])
        r = ev(probe)
        rec = {"hypothesis": k, "seed": seed, "file": "h1", "score": r["score"], **{m: r[m] for m in W}, **info}
        records.append(rec)
        for fk, fc in files.items():
            q = ev(fc)
            records.append({"hypothesis": k, "seed": seed, "file": fk, "score": q["score"], **{m: q[m] for m in W}})
        pd.DataFrame(records).to_csv(out / f"probe2_{tag}.csv", index=False)
    sub = pd.DataFrame([x for x in records if x["hypothesis"] == k and x["file"] == "h1"])
    mean = {m: float(sub[m].mean()) for m in W}
    d = sum(w * (mean[m] - H1_OBSERVED[m]) ** 2 for m, w in W.items())
    per = [sum(w * (row[m] - H1_OBSERVED[m]) ** 2 for m, w in W.items()) for _, row in sub.iterrows()]
    se = float(np.std(per, ddof=1) / np.sqrt(len(per))) if len(per) > 1 else float("nan")
    extra = "  ".join(f"{fk} {np.mean([x['score'] for x in records if x['hypothesis'] == k and x['file'] == fk]):.2f}" for fk in files)
    print(f"[{k:<22}] D {d:.5f} (se {se:.5f})  h1-if-true {sub['score'].mean():6.2f}  " + " ".join(f"{m[6:]} {mean[m]:.3f}" for m in W)
          + f"  | blood {sub['lin_Blood'].iloc[0] if 'lin_Blood' in sub else float('nan'):.3f} CM {sub['lin_CM'].iloc[0] if 'lin_CM' in sub else float('nan'):.3f} genes {sub['genes_median'].iloc[0] if 'genes_median' in sub else float('nan'):.0f}"
          + (f"  | files: {extra}" if extra else "") + f"  {time.time() - t0:.0f}s", flush=True)
print(f"done {time.time() - t0:.0f}s")
