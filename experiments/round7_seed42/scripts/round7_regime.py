"""Round 7: how does the price of each anchor form depend on how close the base already is to the target?

The E10.5-target proxies prefer raw (shallow, un-pooled) anchor nuclei; the zero-elapsed scenario prefers the
re-sequenced ones by a wide margin. In the first the base scores 50-65 on its own, in the second 100. The hidden
test lies in between (entries of real E10.5 cells score 77-81 there). This script measures the whole curve.

    target   E10.5 library 3 without blood (selection half or confirmation half, plus ceiling half), as round7_clean2 A
    base     a share w of "target-like" cells (library-3 cells outside all target halves, without blood) and 1 - w
             depth-matched heart cells of E9.5 (as round7_clean2); w = 0, 0.2, ..., 1
    files    base only | base with 18% of its cells thinned to the anchor depth | 18% raw nuclei |
             18% re-sequenced nuclei moved back 1 | 18% re-sequenced nuclei not moved back      (k_prof 20, no shifts)

The anchor nuclei are superfluous for this target at every w (the target is E10.5 itself); the question is only
which form costs less, and how that changes with the base. Nothing here enters a prediction.

    python -m experiments.round7_regime [--split confirmation]
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
from scipy import sparse

from experiments.round5_anchor import load_anchor
from experiments.round5_common import h3_rows
from experiments.round6_common import HEART, REAL, SEED, FastEval, Log, Mem, depth_of, lineage_of, thin_rows
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.data import board_spec

N = 2000
F = 0.18
POOL = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--share", type=float, default=F, help="share of anchor nuclei (0.18 = the proxy-like share of the first run, 0.45 = the test's)")
    parser.add_argument("--tilt", default="", help="comma-separated tilt strengths: scores the re-sequenced, moved-back form with each (separate table)")
    parser.add_argument("--kprofs", default="", help="comma-separated neighbourhood sizes: scores the re-sequenced, moved-back form with each (separate table)")
    args = parser.parse_args()
    f_share = args.share
    tilts = [float(t) for t in args.tilt.split(",")] if args.tilt else []
    kprofs = [int(k) for k in args.kprofs.split(",")] if args.kprofs else []
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    _, genes = board_spec("T1:val")
    ax, _ = load_anchor(genes, verbose=False)
    t0 = time.time()
    e8 = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
    e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    ref = e8.dense(np.sort(np.random.default_rng(SEED).choice(e8.n_cells, round(.1 * e8.n_cells), replace=False)))
    h3 = h3_rows(e10)
    lib3 = np.flatnonzero(e10.lib == "3")
    used = np.concatenate([np.asarray(v) for v in h3.values()])
    free = np.setdiff1d(lib3, used)
    free = free[e10.labels[free] != "Blood"]
    nb = lambda rows: rows[e10.labels[rows] != "Blood"]
    ta, tb = nb(h3[a]), nb(h3[b])
    ev = FastEval(ref, e10.dense(ta), e10.dense(tb))
    # depth-matched E9.5 heart cells (every lineage thinned to the depth distribution of the same lineage in library 3 without blood)
    pool9 = np.flatnonzero(np.isin(e9.labels, list(HEART)) & (e9.labels != "Blood"))
    lib3_nb = lib3[e10.labels[lib3] != "Blood"]
    lin_t, depth_t, lin_9 = lineage_of(e10.labels[lib3_nb]), depth_of(e10.X[lib3_nb]), lineage_of(e9.labels)
    rng = np.random.default_rng(SEED)
    blocks, labels = [], []
    for lin in np.unique(lin_9[pool9]):
        rows = pool9[lin_9[pool9] == lin]
        tgt = depth_t[lin_t == lin] if (lin_t == lin).sum() >= 50 else depth_t
        blocks.append(sparse.csr_matrix(thin_rows(e9.X[rows], tgt, rng)))
        labels.append(e9.labels[rows])
    x9, lab9 = sparse.vstack(blocks, format="csr"), np.concatenate(labels)
    print(f"free library-3 cells without blood {len(free)}, E9.5 heart cells {x9.shape[0]} (median UMI {np.median(depth_of(x9)):.0f}),"
          f" target {len(ta)} cells (median UMI {np.median(depth_of(e10.X[ta])):.0f})  {time.time() - t0:.0f}s", flush=True)
    fname = f"scores_k_f{f_share:g}_{args.split}.csv" if kprofs else (f"scores_tilt_f{f_share:g}_{args.split}.csv" if tilts else f"scores_{args.split}.csv")
    log = Log(REAL / "runs" / "round7_regime" / fname)
    rs = dict(pool_k="resample", level="type", k_prof=20)
    for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        # one combined source: w x POOL target-like cells, (1 - w) x POOL depth-matched E9.5 cells
        r = np.random.default_rng(SEED + int(round(w * 100)))
        n10 = int(round(w * POOL))
        i10 = np.sort(r.choice(len(free), n10, replace=False)) if n10 else np.array([], dtype=int)
        i9 = np.sort(r.choice(x9.shape[0], POOL - n10, replace=False)) if POOL - n10 else np.array([], dtype=int)
        parts = ([e10.X[free[i10]]] if n10 else []) + ([x9[i9]] if POOL - n10 else [])
        src = copy.copy(e10)
        src.X = sparse.vstack(parts, format="csr")
        src.labels = np.concatenate(([e10.labels[free[i10]]] if n10 else []) + ([lab9[i9]] if POOL - n10 else []))
        src.lib = np.array(["3"] * src.X.shape[0])
        src.n_cells = src.X.shape[0]
        pool = np.arange(src.n_cells)
        anchor = Anchor7(ax)
        cfgs = [("base only", 0.0, {}), ("base, 18% of the cells thinned to the anchor depth", 0.0, {}), ("raw nuclei", f_share, {}),
                ("re-sequenced, moved back 1", f_share, dict(bwd=1.0, **rs)), ("re-sequenced, not moved back", f_share, dict(bwd=0.0, **rs))]
        if tilts:
            cfgs = [cfgs[0], cfgs[2], cfgs[3]] + [(f"re-sequenced, moved back 1, tilt {t:g}", f_share, dict(bwd=1.0, tilt=t, **rs)) for t in tilts]
        if kprofs:
            cfgs = [cfgs[0], cfgs[2]] + [(f"re-sequenced, moved back 1, k_prof {k}", f_share, dict(bwd=1.0, pool_k="resample", level="type", k_prof=k)) for k in kprofs]
        for name, f, kw in cfgs:
            cells, info = predict7(src, pool, anchor, N, f, seed=SEED, return_info=True, **kw)
            if "thinned" in name:
                r2 = np.random.default_rng(SEED + 7)
                rows = info["base_rows"]
                sub = np.sort(r2.choice(len(rows), int(round(f_share * N)), replace=False))
                cells[sub] = thin_rows(src.X[rows[sub]], r2.choice(anchor.depth, len(sub), replace=True), r2).toarray()
            log.add(ev, cells, w=w, config=name, f=f)
        print(f"w {w:g} done {time.time() - t0:.0f}s", flush=True)
    frame = pd.DataFrame(log.records)
    pd.set_option("display.width", 250)
    print(frame.pivot_table(index="config", columns="w", values="score", sort=False).round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(frame.pivot_table(index="config", columns="w", values=m, sort=False).round(3).to_string())
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
