"""Round 6 export: the E12.5 prediction from E10.5, by the same `build` the evaluation calls.

    python -m experiments.round6_export --rho 0.75 --trend 0 [--cells 4000] [--selfcheck scen_development_rho0.75_trend0.csv]

The configuration is taken from `round6_final.configs(rho, trend)["R6 (candidate)"]`, i.e. the very dictionary
the scenario table and the proxies were scored with; only the inputs differ (all of E10.5 as the source pool,
elapsed share 0.5). `--selfcheck` re-scores that configuration on the development scenario table through this
script and compares with the stored table, to show that the code path is the one that was validated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round6_anchor import common_shift
from experiments.round6_common import REAL, SEED, Mem, centroid_transfer, depth_of, lineage_of
from experiments.round6_config import build
from experiments.round6_final import N, TEST_ELAPSED, cells_hash, cells_of, configs, scenario_setup
from experiments.round6_method import Anchor
from t1_pipeline.cli import make_prediction_file
from t1_pipeline.data import board_spec, file_hash, validate_prediction, write_json

CANDIDATE = "R6 (candidate)"
CODE = ("round6_export.py", "round6_final.py", "round6_config.py", "round6_method.py", "round6_anchor.py", "round6_common.py",
        "round5_anchor.py", "round5_common.py", "round5_exp4.py", "bulk_trend.py")


def selfcheck(cfg, anchor, trend, genes, table):
    """Rebuild the candidate's development scenario cells through this script's imports and compare them with the
    validated run: the cells themselves (SHA-256 stored next to the score table) and the four scenario scores."""
    stored = pd.read_csv(table)
    stored = stored[stored["config"] == CANDIDATE].set_index("scenario")["score"]
    stored_hash = json.loads(Path(table).with_suffix(".cells_sha256.json").read_text())[CANDIDATE]
    e10, pool, r4_ref, evs = scenario_setup("development", genes)
    cells = cells_of(cfg, e10, pool, r4_ref, anchor, trend, TEST_ELAPSED, True, "scen", n=N)
    now_hash = cells_hash(cells)
    assert now_hash == stored_hash, ("scenario cells differ from the validated run", now_hash, stored_hash)
    out = {"cells_sha256": {"stored": stored_hash, "recomputed": now_hash, "identical": True}}
    for scen, ev in evs.items():
        now = ev(cells)["score"]
        out[scen] = {"stored": float(stored[scen]), "recomputed": float(now), "abs_diff": float(abs(now - stored[scen]))}
        # identical cells; the scorer's float sums may differ in the last digits with the BLAS thread count
        assert abs(now - stored[scen]) < 1e-3, (scen, now, stored[scen])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--trend", type=float, default=0.0)
    parser.add_argument("--cells", type=int, default=4000)
    parser.add_argument("--name", default="")
    parser.add_argument("--selfcheck", default="", help="scenario table written by round6_final scen (development)")
    args = parser.parse_args()
    cfg = configs(args.rho, args.trend)[CANDIDATE]
    name = args.name or f"round6_heartlib_anchor{cfg['f_per_elapsed'] * TEST_ELAPSED:g}_harm{cfg['harmonise']:g}_trend{cfg['trend']:g}"
    out = REAL / "predictions" / name
    out.mkdir(parents=True, exist_ok=False)
    spec, genes = board_spec("T1:test")
    assert genes == board_spec("T1:val")[1], "validation and test boards must share the gene order used in validation"
    trend, bulk_info = load_trend(genes)
    ax, anchor_info = load_anchor(genes)
    anchor = Anchor(ax)
    source = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    info = {}
    pred = build(cfg, source, np.arange(source.n_cells), anchor, trend, args.cells, elapsed=TEST_ELAPSED, heart_target=True,
                 seed=SEED, key="export", info=info)
    again = build(cfg, source, np.arange(source.n_cells), Anchor(ax), trend, args.cells, elapsed=TEST_ELAPSED, heart_target=True,
                  seed=SEED, key="export")       # a fresh Anchor: the harmonisation shift is recomputed, not read from the cache
    assert np.array_equal(pred, again), "the export is not deterministic"
    path = out / "E12.5_prediction.h5ad"
    make_prediction_file(path, pred, genes)
    written = ad.read_h5ad(path)
    back = np.asarray(written.X.todense() if hasattr(written.X, "todense") else written.X, dtype=np.float32)
    assert np.array_equal(back, pred), "file content differs from the array that was built"
    library = np.expm1(pred.astype(np.float64)).sum(axis=1)
    rows, picked = info.pop("base_rows"), info.pop("anchor_rows")
    types = pd.Series(source.labels[rows]).value_counts(normalize=True)
    near = centroid_transfer(source.X[rows], source.labels[rows], ax[picked]) if len(picked) else np.array([])
    depth_a = depth_of(ax)
    # what the harmonisation removed: recomputed here with the same function and the same rows the build used
    main = np.flatnonzero(np.isin(source.lib, info["heart_libraries"]))
    shift, ctx = common_shift(source, main, ax)
    g = np.array(genes)
    ribo = np.char.startswith(g, "Rpl") | np.char.startswith(g, "Rps")
    harmonisation = {
        "lineages": ctx["info"]["lineages"], "source_rows": int(len(main)),
        "genes_with_shared_shift": int((shift != 0).sum()),
        "genes_shifted_by_more_than_e^0.5": int((np.abs(shift) >= .5).sum()),
        "genes_shifted_by_more_than_e^1": int((np.abs(shift) >= 1).sum()),
        "ribosomal_protein_genes_shifted_by_more_than_e^0.5": int((ribo & (np.abs(shift) >= .5)).sum()),
        "mean_shift_of_those_ribosomal_genes": float(shift[ribo & (np.abs(shift) >= .5)].mean()),
        "anchor_lower_top20": {str(g[i]): round(float(shift[i]), 2) for i in np.argsort(-shift)[:20]},
        "anchor_higher_top20": {str(g[i]): round(float(shift[i]), 2) for i in np.argsort(shift)[:20]},
        "note": "shift = log(mean counts in the E10.5 heart library / mean counts in the anchor), the part shared by all lineages",
    }
    manifest = {
        "name": name,
        "method": "E10.5 heart-proper library in its natural composition + real E14.5 Multiome nuclei (anchor), "
                  "anchor harmonised against the E10.5 heart library (lineage-shared shift removed)",
        "configuration": cfg, "configuration_name": CANDIDATE, "elapsed_share": TEST_ELAPSED, "seed": SEED,
        "cells_total": int(len(pred)), **{k: v for k, v in info.items()},
        "base": {"rule": "libraries of E10.5 with >= 90% heart-lineage cells, cells drawn uniformly without replacement",
                 "library_shares": pd.Series(source.lib[rows]).value_counts(normalize=True).round(4).to_dict(),
                 "celltype_shares": types.round(4).to_dict(),
                 "lineage_shares": pd.Series(lineage_of(source.labels[rows])).value_counts(normalize=True).round(4).to_dict(),
                 "median_umi": float(np.median(depth_of(source.X[rows])))},
        "anchor": {"rule": "nuclei drawn uniformly without replacement from the QC-passing nuclei of GSM8394736",
                   "nearest_E10.5_celltype_shares": pd.Series(near).value_counts(normalize=True).round(4).to_dict(),
                   "lineage_shares": pd.Series(lineage_of(near)).value_counts(normalize=True).round(4).to_dict(),
                   "median_umi": float(np.median(depth_a[picked])) if len(picked) else None},
        "input_sha256": {"E10.5_RNA.h5ad": file_hash(source.path)},
        "harmonisation": harmonisation,
        "external": {"bulk_trend": bulk_info if cfg["trend"] else "not used by this configuration", "anchor": anchor_info},
        "code_sha256": {f: file_hash(Path(__file__).parent / f) for f in CODE},
        "validation": validate_prediction(path, "T1:test"),
        "board": {k: spec[k] for k in ("min_cells", "max_cells", "n_genes")},
        "prediction_cells_sha256": cells_hash(pred),
        "nonzero_fraction": float((pred > 0).mean()),
        "library_sum_min": float(library.min()), "library_sum_max": float(library.max()),
        "status": "not_submitted",
    }
    if args.selfcheck:
        manifest["selfcheck_same_code_path"] = selfcheck(cfg, anchor, trend, genes, REAL / "runs" / "round6_final" / args.selfcheck)
    write_json(out / "manifest.json", manifest)
    print((out / "manifest.json").read_text())


if __name__ == "__main__":
    main()
