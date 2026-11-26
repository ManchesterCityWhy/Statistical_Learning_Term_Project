"""Round 7: round-7 against round-6 anchors on time-gap proxies WITHOUT the depth coincidence.

The library-3 proxies (p8/deep, 2d/deep) reward shallow, un-pooled anchor nuclei for two reasons unrelated to
development: library 3 holds 16% very sparse blood cells that E8.5/E9.5 lack, and the E8.5/E9.5 heart cells are
about 27% deeper than library 3. Here both are removed:

    target   E10.5 library 3 without its blood cells (selection half / confirmation half, plus ceiling halves)
    base     heart-labelled non-blood cells of the source stage (E9.5 or E8.5), binomially thinned so that every
             lineage has the depth distribution of the same lineage in the target ("dm"); also the un-thinned
             base ("raw") for comparison
    reference  10% of E8.5, as in p8 / 2d

Configurations (anchor share = rho x elapsed, as in round7_eval): round-6 form (un-pooled, harmonised nuclei) and
round-7 form (re-sequenced at the base's lineage depth, moved back by cell type), with the round-7 add-ons.

    python -m experiments.round7_clean --source E9.5|E8.5 [--split confirmation]
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import h3_rows
from experiments.round6_common import HEART, REAL, SEED, FastEval, Log, Mem, depth_of, lineage_of, thin_rows
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.data import board_spec

N = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="E9.5")
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    args = parser.parse_args()
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    ax, _ = load_anchor(genes, verbose=False)
    t0 = time.time()
    e8 = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    is9 = args.source == "E9.5"
    src = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes) if is9 else e8
    elapsed, days = (.2, 5.0) if is9 else (1 / 3, 6.0)
    prev = (e8, np.arange(e8.n_cells)) if is9 else None
    ref = e8.dense(np.sort(np.random.default_rng(SEED).choice(e8.n_cells, round(.1 * e8.n_cells), replace=False)))
    h3 = h3_rows(e10)
    nb = lambda rows: rows[e10.labels[rows] != "Blood"]
    ta, tb = nb(h3[a]), nb(h3[b])
    ev = FastEval(ref, e10.dense(ta), e10.dense(tb))
    pool = np.flatnonzero(np.isin(src.labels, list(HEART)) & (src.labels != "Blood"))
    # depth-matched base: every lineage thinned to the depth distribution of the same lineage in library 3 (no blood)
    lib3_nb = np.flatnonzero((e10.lib == "3") & (e10.labels != "Blood"))
    lin_t, lin_s = lineage_of(e10.labels[lib3_nb]), lineage_of(src.labels)
    depth_t = depth_of(e10.X[lib3_nb])
    dm = copy.copy(src)
    x = src.X.tolil()
    rng = np.random.default_rng(SEED)
    for lin in np.unique(lin_s[pool]):
        rows = pool[lin_s[pool] == lin]
        tgt = depth_t[lin_t == lin] if (lin_t == lin).sum() >= 50 else depth_t
        x[rows] = thin_rows(src.X[rows], tgt, rng)
    dm.X = x.tocsr()
    print(f"source {args.source}: base cells {len(pool)}, median UMI raw {np.median(depth_of(src.X[pool])):.0f} -> depth-matched {np.median(depth_of(dm.X[pool])):.0f};"
          f" target without blood {len(ta)} cells, median UMI {np.median(depth_of(e10.X[ta])):.0f}  {time.time() - t0:.0f}s", flush=True)
    log = Log(REAL / "runs" / "round7_clean" / f"scores_{args.source}_{args.split}.csv")
    r7 = dict(bwd=1.0, pool_k="resample", level="type")
    dkw = dict(prev=prev, days_to_anchor=days, dev=elapsed) if prev is not None else {}
    tkw = dict(trend=trend, trend_share=elapsed * days / 5.0)
    for base_name, stage in (("depth-matched base", dm), ("released base", src)):
        anchor = Anchor7(ax)
        cfgs = [("base only", 0.0, {}),
                ("round 6 form, rho 0.4", .4, {}), ("round 6 form, rho 0.75", .75, {}), ("round 6 form, rho 0.9", .9, {}),
                ("round 7 form, rho 0.4", .4, r7), ("round 7 form, rho 0.75", .75, r7), ("round 7 form, rho 0.9", .9, r7),
                ("round 7 form, rho 0.9 + trend", .9, {**r7, **tkw}),
                ("round 7 candidate (rho 0.9 + two-witness + trend)", .9, {**r7, **tkw, **dkw}),
                ("round 6 form, rho 0.75 + trend", .75, tkw)]
        for name, rho, kw in cfgs:
            cells = predict7(stage, pool, anchor, N, rho * elapsed, seed=SEED, **kw)
            log.add(ev, cells, base=base_name, config=name, f=round(rho * elapsed, 4))
    frame = pd.DataFrame(log.records)
    pd.set_option("display.width", 250)
    print(frame.pivot_table(index="config", columns="base", values="score", sort=False).round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(frame.pivot_table(index="config", columns="base", values=m, sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
