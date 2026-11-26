"""Round 7 check: is the gain from shallow anchor nuclei on the library-3 proxies a depth-distribution artefact?

Library 3 of E10.5 holds 16% blood cells with a median of 4.6k UMI; E8.5/E9.5 hold 1-3%. A prediction built
from E8.5/E9.5 cells lacks that sparse population, and un-pooled anchor nuclei (3k UMI) happen to fill the gap.
Here the blood cells are removed from the target (and from the base), so that source and target have comparable
depth distributions; raw and pooled anchor nuclei are then compared at equal shares.

    python -m experiments.round7_nb --source E9.5|E8.5
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.round5_anchor import load_anchor
from experiments.round5_common import h3_rows
from experiments.round6_common import HEART, LINEAGES, REAL, SEED, FastEval, Log, Mem, depth_of, lineage_of
from experiments.round6_s1 import feasible_shares
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.data import board_spec

N = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="E9.5")
    args = parser.parse_args()
    _, genes = board_spec("T1:val")
    ax, _ = load_anchor(genes, verbose=False)
    anchor = Anchor7(ax)
    t0 = time.time()
    e8 = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    src = e8 if args.source == "E8.5" else Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    ref = e8.dense(np.sort(np.random.default_rng(SEED).choice(e8.n_cells, round(.1 * e8.n_cells), replace=False)))
    h3 = h3_rows(e10)
    nb = lambda rows: rows[e10.labels[rows] != "Blood"]
    ta, tb = nb(h3["development"]), nb(h3["ceiling"])
    ev = FastEval(ref, e10.dense(ta), e10.dense(tb))
    ev_full = FastEval(ref, e10.dense(h3["development"]), e10.dense(h3["ceiling"]))
    lin_t = lineage_of(e10.labels[ta])
    target_shares = {k: float((lin_t == k).mean()) for k in LINEAGES}
    heart_nb = np.flatnonzero(np.isin(src.labels, list(HEART)) & (src.labels != "Blood"))
    heart = np.flatnonzero(np.isin(src.labels, list(HEART)))
    print("target without blood:", len(ta), "cells, median UMI", float(np.median(depth_of(e10.X[ta]))), "| with blood",
          float(np.median(depth_of(e10.X[h3["development"]]))), "| base median UMI", float(np.median(depth_of(src.X[heart_nb]))),
          "| depth q10 target-nb / target / base", [float(np.quantile(depth_of(x), .1)) for x in (e10.X[ta], e10.X[h3["development"]], src.X[heart_nb])],
          f"{time.time() - t0:.0f}s", flush=True)
    log = Log(REAL / "runs" / "round7_nb" / f"scores_{args.source}.csv")
    oracle = feasible_shares(src, heart_nb, target_shares, N)
    for base_name, pool, shares, evaluator in (("heart-nb natural", heart_nb, None, ev), ("heart-nb oracle", heart_nb, oracle, ev),
                                               ("heart natural | target WITH blood", heart, None, ev_full)):
        for f in (0.0, .1, .2, .3):
            for form, kw in (("raw", {}), ("pooled", {"pool_k": "match"}), ("pooled back", {"pool_k": "match", "bwd": 1.0})):
                if f == 0 and form != "raw":
                    continue
                cells = predict7(src, pool, anchor, N, f, shares=shares, seed=SEED, **kw)
                log.add(evaluator, cells, base=base_name, f=f, form=form)
    frame = pd.DataFrame(log.records)
    pd.set_option("display.width", 250)
    print(frame.pivot_table(index=["base", "form"], columns="f", values="score", sort=False).round(2).to_string())
    for m in ("skill_mmd_u", "skill_variogram"):
        print(m); print(frame.pivot_table(index=["base", "form"], columns="f", values=m, sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
