"""Round 6: what the cells of the round-5 file and of the round-6 file are (no scoring, no target involved).

Rebuilds the donor rows of both files with the code that exported them and prints library, cell-type and lineage
shares, depth, and the heart-lineage share of every released library.

    python -m experiments.round6_describe
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_exp2 import deconv_q
from experiments.round6_common import HEART, LINEAGES, REAL, SEED, Mem, centroid_transfer, depth_of, lineage_of
from experiments.round6_config import build
from experiments.round6_final import TEST_ELAPSED, configs
from experiments.round6_method import Anchor
from t1_pipeline.data import board_spec


def shares(values, order=None):
    s = pd.Series(values).value_counts(normalize=True)
    if order is not None:
        s = s.reindex(order).fillna(0.0)
    return {k: round(float(v) * 100, 2) for k, v in s.items()}


def round5_rows(source, trend, n=2000):
    """Donor rows of the round-5 base: the sampling lines of round5_exp4.r4_cells with pool = all cells."""
    pool = np.arange(source.n_cells)
    labels = source.labels[pool]
    names = np.unique(labels)
    prof = np.stack([np.asarray(source.X[pool[labels == g]].expm1().mean(axis=0)).ravel() for g in names], axis=1)
    p = np.array([(labels == g).mean() for g in names])
    q = deconv_q(prof, p, trend, 3.0, 10)
    w = np.zeros(len(pool))
    for g, prob in zip(names, q):
        w[labels == g] = prob / (labels == g).sum()
    return pool[np.random.default_rng(SEED).choice(len(pool), n, replace=False, p=w / w.sum())]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, default=0.4, help="anchor share per elapsed share of the round-6 file")
    parser.add_argument("--cells", type=int, default=4000)
    args = parser.parse_args()
    _, genes = board_spec("T1:test")
    trend, _ = load_trend(genes)
    ax, _ = load_anchor(genes, verbose=False)
    out = {}
    stages = {s: Mem(REAL / f"data/T1/{s}_RNA.h5ad", genes) for s in ("E8.5", "E9.5", "E10.5")}
    for name, st in stages.items():
        heart = np.isin(st.labels, list(HEART))
        lin = lineage_of(st.labels)
        per_lib = {}
        for lib in np.unique(st.lib):
            rows = st.lib == lib
            within = pd.Series(lin[rows & heart]).value_counts(normalize=True)
            per_lib[str(lib)] = {"cells": int(rows.sum()), "heart_lineage_share_%": round(float(heart[rows].mean()) * 100, 2),
                                 "median_umi": float(np.median(depth_of(st.X[np.flatnonzero(rows)]))),
                                 "lineage_shares_%": shares(lin[rows], LINEAGES),
                                 "share_among_heart_lineage_cells_%": {k: round(float(v) * 100, 2) for k, v in within.items()}}
        within = pd.Series(lin[heart]).value_counts(normalize=True)
        out[name] = {"cells": int(st.n_cells), "heart_lineage_share_%": round(float(heart.mean()) * 100, 2), "libraries": per_lib,
                     "share_among_heart_lineage_cells_%": {k: round(float(v) * 100, 2) for k, v in within.items()}}
    e10 = stages["E10.5"]
    lib3 = np.flatnonzero(e10.lib == "3")
    near_all = centroid_transfer(e10.X[lib3], e10.labels[lib3], ax)      # anchor lineage, one reference for both files
    lin_anchor, depth_anchor = lineage_of(near_all), depth_of(ax)
    out["anchor"] = {"nuclei": int(ax.shape[0]), "median_umi": float(np.median(depth_anchor)),
                     "lineage_shares_%_nearest_library3_type": shares(lin_anchor, LINEAGES),
                     "nearest_library3_type_%": shares(near_all)}
    # round-5 file
    rows5 = round5_rows(e10, trend)
    take5 = np.sort(np.random.default_rng(SEED).choice(ax.shape[0], 1200, replace=False))
    # round-6 file
    info = {}
    cfg = configs(args.rho, 0.0)["R6 (candidate)"]
    build(cfg, e10, np.arange(e10.n_cells), Anchor(ax), trend, args.cells, elapsed=TEST_ELAPSED, heart_target=True, key="export", info=info)
    rows6, take6 = info["base_rows"], info["anchor_rows"]
    depth10 = depth_of(e10.X)
    for tag, rows, take in (("round5_file", rows5, take5), ("round6_file", rows6, take6)):
        nb, na = len(rows), len(take)
        base_lin, anc_lin = lineage_of(e10.labels[rows]), lin_anchor[take]
        whole = {k: round(float(((base_lin == k).sum() + (anc_lin == k).sum()) / (nb + na)) * 100, 2) for k in LINEAGES}
        out[tag] = {"base_cells": nb, "anchor_nuclei": na, "anchor_share_%": round(na / (nb + na) * 100, 2),
                    "base_library_shares_%": shares(e10.lib[rows]), "base_median_umi": float(np.median(depth10[rows])),
                    "base_celltype_shares_%": shares(e10.labels[rows]), "base_lineage_shares_%": shares(base_lin, LINEAGES),
                    "anchor_lineage_shares_%": shares(anc_lin, LINEAGES), "anchor_median_umi": float(np.median(depth_anchor[take])),
                    "whole_file_lineage_shares_%": whole,
                    "base_celltypes_present": int(len(np.unique(e10.labels[rows]))), "celltypes_in_stage": int(len(np.unique(e10.labels)))}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (REAL / "runs" / "round6_describe").mkdir(parents=True, exist_ok=True)
    (REAL / "runs" / "round6_describe" / f"describe_rho{args.rho:g}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
