"""Build a depth-matched validation proxy: thin E8.5/E9.5 UMIs down to the E10.5 depth distribution.

E8.5 and E9.5 were sequenced at the same depth (median ~10,900 UMIs per cell) but E10.5 at a much lower
one (~6,800). In the local validation (E9.5 -> E10.5) that technical drop dominates the change of
thousands of low-expressed genes and the zero structure; in the hidden test (E10.5 -> E12.5) the source
already carries the shallow depth. To choose methods under test-like conditions we thin the earlier
stages to the E10.5 depth and validate on "thinned E9.5 -> E10.5".

Counts are recovered exactly from the released log1p(10^4 x counts / UMI) values: the smallest non-zero
normalised value of a cell is 10^4 / UMI (one count). Each source cell gets a target depth by quantile
matching against the E10.5 depth distribution, and counts are binomially thinned.

    python -m experiments.depth_match --out data/proxy      # writes E8.5_RNA.h5ad, E9.5_RNA.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from experiments.round3_common import open_stage
from t1_pipeline.data import write_json

SEED = 42


def counts_of(x):
    """Recover integer UMI counts and depth from log1p-normalised rows."""
    norm = np.expm1(x.astype(np.float64))
    masked = np.where(norm > 0, norm, np.inf)
    depth = 1e4 / masked.min(axis=1)
    counts = norm * (depth / 1e4)[:, None]
    return counts, depth


def depth_distribution(stage, n=6000, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(stage.n_cells, min(n, stage.n_cells), replace=False))
    _, depth = counts_of(stage.dense(rows))
    return np.sort(depth)


def thin_stage(stage, target_depths, out_path, seed=SEED, batch=512):
    rng = np.random.default_rng(seed)
    source_depths = np.concatenate([counts_of(stage.dense(np.arange(s, min(s + batch, stage.n_cells))))[1]
                                    for s in range(0, stage.n_cells, batch)])
    quantile = (pd.Series(source_depths).rank(method="first").to_numpy() - .5) / len(source_depths)
    goal = np.quantile(target_depths, quantile)
    keep_prob = np.minimum(1.0, goal / source_depths)
    blocks, max_rounding, achieved = [], 0.0, []
    for start in range(0, stage.n_cells, batch):
        rows = np.arange(start, min(start + batch, stage.n_cells))
        counts, _ = counts_of(stage.dense(rows))
        integer = np.rint(counts)
        max_rounding = max(max_rounding, float(np.abs(counts - integer).max()))
        thinned = rng.binomial(integer.astype(np.int64), keep_prob[rows][:, None]).astype(np.float64)
        total = thinned.sum(axis=1, keepdims=True)
        achieved.append(total.ravel())
        x = np.log1p(thinned / np.maximum(total, 1) * 1e4).astype(np.float32)
        blocks.append(sparse.csr_matrix(x))
    genes = np.asarray(stage.data.var_names)[stage.order]
    data = ad.AnnData(X=sparse.vstack(blocks).tocsr(),
                      obs=pd.DataFrame({"celltype": stage.labels}, index=np.asarray(stage.data.obs_names)),
                      var=pd.DataFrame(index=genes))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(out_path, compression="gzip")
    achieved = np.concatenate(achieved)
    return {"source_depth_median": float(np.median(source_depths)), "target_depth_median": float(np.median(goal)),
            "achieved_depth_median": float(np.median(achieved)), "max_rounding_error": max_rounding,
            "cells": int(stage.n_cells)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/proxy")
    args = parser.parse_args()
    out = Path(args.out)
    target = depth_distribution(open_stage("E10.5"))
    report = {"E10.5_depth_median": float(np.median(target))}
    for name in ("E8.5", "E9.5"):
        stage = open_stage(name)
        report[name] = thin_stage(stage, target, out / f"{name}_RNA.h5ad")
        stage.close()
        print(name, report[name], flush=True)
    write_json(out / "depth_match.json", report)


if __name__ == "__main__":
    main()
