"""Round 5 export: E12.5 = round-4 cells drawn from E10.5 + real E14.5 Multiome nuclei as a future anchor.

    python -m experiments.round5_export --pool cardiac --h 2 --fraction 0.375

--pool all      every E10.5 cell is a candidate donor (as in round 4)
--pool cardiac  only cardiac / blood lineages (the data protocol says most surrounding tissue is removed at E12.5)
--pool lib3     only E10.5 library 3 (the deep heart-proper library)
The anchor fraction is the share of E14.5 nuclei in the final population; 2,000 round-4 cells are kept and
round(2000 * f / (1 - f)) anchor nuclei are added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import CARDIAC, REAL, SEED, Mem, pb
from experiments.round5_exp4 import R4_W, r4_cells
from t1_pipeline.cli import make_prediction_file
from t1_pipeline.data import board_spec, file_hash, validate_prediction, write_json

N_BASE = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="cardiac", choices=("all", "cardiac", "lib3"))
    parser.add_argument("--h", type=float, default=2.0)
    parser.add_argument("--fraction", type=float, default=0.375)
    args = parser.parse_args()
    name = f"round5_anchor{args.fraction:g}_{args.pool}_h{args.h:g}"
    out = REAL / "predictions" / name
    out.mkdir(parents=True, exist_ok=False)
    _, genes = board_spec("T1:test")
    trend, bulk_info = load_trend(genes)
    anchor, anchor_info = load_anchor(genes)
    source = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    pool = {"all": np.arange(source.n_cells), "lib3": np.flatnonzero(source.lib == "3"),
            "cardiac": np.flatnonzero(np.isin(source.labels, list(CARDIAC)))}[args.pool]
    ref = source.dense(np.sort(np.random.default_rng(SEED).choice(source.n_cells, N_BASE, replace=False)))
    cells, base = r4_cells(source, pool, ref, trend, N_BASE, strength=3.0 * args.h, bulk=3.0 * args.h)
    k = round(N_BASE * args.fraction / (1 - args.fraction))
    take = np.sort(np.random.default_rng(SEED).choice(anchor.shape[0], k, replace=False))
    pred = np.vstack([cells, anchor[take].toarray().astype(np.float32)])
    path = out / "E12.5_prediction.h5ad"
    make_prediction_file(path, pred, genes)
    library = np.expm1(pred.astype(np.float64)).sum(axis=1)
    old = REAL / "predictions" / "round4_bulk+deconv_s1.0_h1" / "E12.5_prediction.h5ad"
    check = {}
    if old.exists():
        import anndata as ad
        prev = np.asarray(ad.read_h5ad(old).X.todense() if hasattr(ad.read_h5ad(old).X, "todense") else ad.read_h5ad(old).X)
        check = {"corr_pseudobulk_with_round4_file": float(np.corrcoef(pb(prev), pb(cells))[0, 1]),
                 "max_abs_pseudobulk_diff_base_vs_round4_file": float(np.abs(pb(prev) - pb(cells)).max())}
    write_json(out / "manifest.json", {
        "name": name, "method": "round-4 direction-imposed deconvolution cells from E10.5 + E14.5 Multiome anchor nuclei",
        "pool": args.pool, "pool_cells": int(len(pool)), "h": args.h, "deconv_strength": 3.0 * args.h,
        "bulk_strength": 3.0 * args.h, "round4_weights": R4_W, "anchor_fraction": args.fraction,
        "base_cells": N_BASE, "anchor_cells": int(k), "seed": SEED,
        "input_sha256": {"E10.5_RNA.h5ad": file_hash(source.path)},
        "external": {"bulk_trend": bulk_info, "anchor": anchor_info},
        "validation": validate_prediction(path, "T1:test"),
        "nonzero_fraction": float((pred > 0).mean()), "nonzero_fraction_base": float((cells > 0).mean()),
        "library_min": float(library.min()), "library_max": float(library.max()),
        "regression_check": check, "status": "not_submitted"})
    print((out / "manifest.json").read_text()[:600])
    print("pred", pred.shape, "nonzero", (pred > 0).mean(), "lib sums", library.min(), library.max(), check)


if __name__ == "__main__":
    main()
