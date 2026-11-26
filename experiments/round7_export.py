"""Round 7 export: the E12.5 prediction from E10.5, by the same `round7_method.predict7` the evaluation calls.

    python -m experiments.round7_export --f 0.3 --pos 0.5 [--cells 4000] [--name ...]

    base    E10.5 heart-proper library (>= 90% heart-lineage cells; library 3), natural composition; moved forward by
            `pos` along its source -> anchor log-ratio only if pos > 0 (the submitted configuration uses pos = 0)
    anchor  share `f` of the file: E14.5 nuclei (GSE272154 / GSM8394736) re-sequenced at the depth of E10.5 cells
            of the same lineage, harmonised (lineage-shared shift removed), moved back by 1 - pos to the E10.5 means
            of their cell type (`--level type`, the default) or lineage
    --dev   two-witness developmental shift of both parts (share of the E10.5 -> E14.5 path)
    --trend within-cell bulk trend of both parts (share of the five-day E9.5 -> E14.5 bulk interval)

Checks written to the manifest: determinism (two builds with fresh objects are identical), the file read back
equals the array, format validation for T1:test, library sums, row uniqueness, composition, depth, and a
same-code-path check (the zero-elapsed scenario table of this configuration is recomputed through this script and
compared with the stored evaluation table when `--selfcheck` names it, and this script's `build`, called with the
scenario pool, must return exactly the cells the evaluation scored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from experiments.round5_anchor import load_anchor
from experiments.round6_common import REAL, SEED, Mem, depth_of, lineage_of
from experiments.round6_config import heart_libraries
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.cli import make_prediction_file
from t1_pipeline.data import board_spec, file_hash, validate_prediction, write_json

CODE = ("round7_export.py", "round7_method.py", "round7_eval.py", "round7_probe.py", "round6_method.py", "round6_anchor.py",
        "round6_common.py", "round6_config.py", "round6_s1.py", "round6_final.py", "round6_suite.py", "round5_anchor.py", "round5_common.py",
        "round5_exp4.py", "round5_exp2.py", "round4_direction.py", "bulk_trend.py", "geo_transport.py",
        "../t1_pipeline/data.py", "../t1_pipeline/cli.py", "../t1_pipeline/evaluation.py")


def cells_hash(cells):
    return hashlib.sha256(np.ascontiguousarray(cells, dtype=np.float32).tobytes()).hexdigest()


def build(source, ax, f, pos, n, seed=SEED, return_info=False, level="type", dev=0.0, prev=None, days=4.0, trend=None, trend_share=0.0, k_prof=10,
          pool=None):
    """The exported configuration. A fresh Anchor7 every call, so nothing is read from a cache of another call.

    pool: source rows that may be used (default: all). The export uses all of E10.5; the same-code-path check calls
    this very function with the pool of the zero-elapsed scenario table and compares the cells it returns with the
    cells the evaluation scored."""
    anchor = Anchor7(ax)
    pool = np.arange(source.n_cells) if pool is None else np.asarray(pool)
    libs = heart_libraries(source, pool)
    assert libs, "no heart-proper library in the source"
    main = pool[np.isin(source.lib[pool], libs)]
    dkw = dict(prev=prev, days_to_anchor=days, dev=dev) if dev else {}
    out = predict7(source, main, anchor, n, f, fwd=pos, bwd=1.0 - pos, harmonise=1.0, pool_k="resample" if f > 0 else 1,
                   level=level, seed=seed, return_info=True, trend=trend, trend_share=trend_share, k_prof=k_prof, **dkw)
    if return_info:
        return out[0], out[1], anchor, main, libs, dkw
    return out[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=float, required=True)
    parser.add_argument("--pos", type=float, required=True)
    parser.add_argument("--cells", type=int, default=4000)
    parser.add_argument("--level", default="type", choices=("type", "lineage"))
    parser.add_argument("--dev", type=float, default=0.0, help="two-witness developmental shift, share of the E10.5 -> E14.5 path")
    parser.add_argument("--k-prof", type=int, default=10, help="number of nearest anchor nuclei pooled into the profile a nucleus is re-sequenced from")
    parser.add_argument("--trend", type=float, default=0.0, help="within-cell bulk trend, share of the five-day E9.5 -> E14.5 bulk interval (two days = 0.4)")
    parser.add_argument("--name", default="")
    parser.add_argument("--selfcheck", default="", help="scenario table of round7_eval (development) holding this configuration")
    parser.add_argument("--selfcheck-config", default="")
    args = parser.parse_args()
    name = args.name or f"round7_heartlib_f{args.f:g}_pos{args.pos:g}_{args.level}_k{args.k_prof}_dev{args.dev:g}_trend{args.trend:g}"
    out = REAL / "predictions" / name
    out.mkdir(parents=True, exist_ok=False)
    spec, genes = board_spec("T1:test")
    assert genes == board_spec("T1:val")[1]
    ax, anchor_info = load_anchor(genes)
    source = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    previous = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes) if args.dev else None
    prev = (previous, np.arange(previous.n_cells)) if args.dev else None
    trend, bulk_info = (None, None)
    if args.trend:
        from experiments.bulk_trend import load_trend
        trend, bulk_info = load_trend(genes)
    kw = dict(level=args.level, dev=args.dev, prev=prev, days=4.0, trend=trend, trend_share=args.trend, k_prof=args.k_prof)
    pred, info, anchor, main, libs, dkw = build(source, ax, args.f, args.pos, args.cells, return_info=True, **kw)
    again = build(source, ax, args.f, args.pos, args.cells, **kw)
    assert np.array_equal(pred, again), "the export is not deterministic (two builds in this process differ)"
    path = out / "E12.5_prediction.h5ad"
    make_prediction_file(path, pred, genes)
    written = ad.read_h5ad(path)
    back = np.asarray(written.X.todense() if hasattr(written.X, "todense") else written.X, dtype=np.float32)
    assert np.array_equal(back, pred), "file content differs from the array that was built"
    assert list(written.var_names.astype(str)) == list(genes), "gene order differs from the official panel"
    assert np.isfinite(pred).all() and (pred >= 0).all()
    library = np.expm1(pred.astype(np.float64)).sum(axis=1)
    rows, picked = info["base_rows"], info["anchor_rows"]
    n_base = len(rows)
    ckw = {k: v for k, v in dkw.items() if k != "dev"}
    ctx = anchor.context(source, main, "min", args.level, **ckw)
    genes_detected = (pred > 0).sum(axis=1)
    lin_base = lineage_of(source.labels[rows])
    lin_anchor = ctx["lineage"][picked]
    total_lineage = pd.Series(np.concatenate([lin_base, lin_anchor])).value_counts(normalize=True).round(4).to_dict()
    g = np.array(genes)
    c = ctx["c"]
    ribo = np.char.startswith(g, "Rpl") | np.char.startswith(g, "Rps")
    manifest = {
        "name": name,
        "method": "E10.5 heart-proper library in its natural composition, plus E14.5 Multiome nuclei re-sequenced at E10.5 depth, "
                  "harmonised and moved back to the E10.5 cell-type means (share bwd); both parts moved by the two-witness "
                  "developmental shift (share dev of the E10.5 -> E14.5 path) and the within-cell bulk trend (share of its five days)"
                  + ("; base additionally moved forward along the lineage log-ratio (pos > 0)" if args.pos else ""),
        "configuration": {"f": args.f, "pos": args.pos, "fwd": args.pos, "bwd": 1.0 - args.pos, "harmonise": 1.0, "mode": "min",
                          "level": args.level, "dev": args.dev, "days_to_anchor": 4.0, "previous_stage": "E9.5 (all cells)" if args.dev else None,
                          "bulk_trend_share": args.trend,
                          "anchor_form": f"resample (k_prof = {args.k_prof}, multinomial at source-lineage depth)"},
        "seed": SEED, "cells_total": int(len(pred)), "base_cells": int(n_base), "anchor_cells": int(len(picked)),
        "heart_libraries": libs,
        "base": {"library_shares": pd.Series(source.lib[rows]).value_counts(normalize=True).round(4).to_dict(),
                 "celltype_shares": pd.Series(source.labels[rows]).value_counts(normalize=True).round(4).to_dict(),
                 "lineage_shares": pd.Series(lin_base).value_counts(normalize=True).round(4).to_dict(),
                 "median_umi_of_source_rows": float(np.median(depth_of(source.X[rows])))},
        "anchor": {"lineage_shares": pd.Series(lin_anchor).value_counts(normalize=True).round(4).to_dict(),
                   "nearest_E10.5_celltype_shares": pd.Series(ctx["near"][picked]).value_counts(normalize=True).round(4).to_dict(),
                   "median_umi_raw_nuclei": float(np.median(anchor.depth[picked])) if len(picked) else None,
                   "median_genes_detected_resequenced": float(np.median(genes_detected[n_base:])) if len(picked) else None},
        "file": {"lineage_shares_total": total_lineage,
                 "median_genes_detected": float(np.median(genes_detected)),
                 "median_genes_detected_base": float(np.median(genes_detected[:n_base])),
                 "nonzero_fraction": float((pred > 0).mean()), "distinct_rows": int(len(np.unique(pred, axis=0))),
                 "library_sum_min": float(library.min()), "library_sum_max": float(library.max())},
        "harmonisation": {"lineages": ctx["used"], "genes_with_shared_shift": int((c != 0).sum()),
                          "genes_shifted_by_more_than_e^0.5": int((np.abs(c) >= .5).sum()),
                          "ribosomal_protein_genes_shifted_by_more_than_e^0.5": int((ribo & (np.abs(c) >= .5)).sum()),
                          "anchor_lower_top15": {str(g[i]): round(float(c[i]), 2) for i in np.argsort(-c)[:15]},
                          "anchor_higher_top15": {str(g[i]): round(float(c[i]), 2) for i in np.argsort(c)[:15]}},
        "two_witness_shift": {k: {"genes_moved": int((v != 0).sum()), "genes_abs_over_0.5": int((np.abs(v) >= .5).sum()),
                                  "up_top15": {str(g[i]): round(float(v[i]), 2) for i in np.argsort(-v)[:15]},
                                  "down_top15": {str(g[i]): round(float(v[i]), 2) for i in np.argsort(v)[:15]}}
                              for k, v in ctx.get("dev", {}).items()},
        "input_sha256": {"E10.5_RNA.h5ad": file_hash(source.path)},
        "external": {"anchor": anchor_info, "bulk_trend": bulk_info if args.trend else "not used by this configuration"},
        "input_previous_stage_sha256": {"E9.5_RNA.h5ad": file_hash(previous.path)} if args.dev else None,
        "code_sha256": {f: file_hash(Path(__file__).parent / f) for f in CODE},
        "validation": validate_prediction(path, "T1:test"),
        "board": {k: spec[k] for k in ("min_cells", "max_cells", "n_genes")},
        "prediction_cells_sha256": cells_hash(pred), "file_sha256": file_hash(path), "file_bytes": path.stat().st_size,
        "status": "not_submitted",
    }
    if args.selfcheck:
        from experiments.round6_final import scenario_setup
        from experiments.round7_eval import C, cells_of
        stored = pd.read_csv(REAL / "runs" / "round7_eval" / args.selfcheck)
        stored = stored[stored["config"] == args.selfcheck_config].set_index("scenario")["score"]
        e10, pool, _, evs = scenario_setup("development", genes)
        cfg = C(args.selfcheck_config, rho_f=args.f / .5, rho_t=args.pos / .5, bwd="interp", pool_k="resample", level=args.level, rho_dev=args.dev / .5,
                days_trend=args.trend * 5.0 / 2.0, k_prof=args.k_prof)
        cells, f, t, bwd = cells_of(cfg, e10, pool, Anchor7(ax), .5, True, prev=prev, days_to_anchor=4.0)
        # the export's own `build`, given the scenario pool and the evaluation's cell count, must return the very cells
        # the evaluation built and scored
        mine = build(e10, ax, args.f, args.pos, len(cells), pool=pool, **kw)
        assert np.array_equal(mine, cells), "export build and evaluation build differ on the scenario pool"
        check = {"f": f, "pos": t, "bwd": bwd, "export_build_equals_evaluation_cells": True, "scenario_cells_sha256": cells_hash(cells)}
        for scen, ev in evs.items():
            now = ev(cells)["score"]
            check[scen] = {"stored": float(stored[scen]), "recomputed": float(now), "abs_diff": float(abs(now - stored[scen]))}
            assert abs(now - stored[scen]) < 1e-3, (scen, now, stored[scen])
        manifest["selfcheck_same_code_path"] = check
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("name", "configuration", "cells_total", "base_cells", "anchor_cells", "file", "validation",
                                               "prediction_cells_sha256", "file_sha256", "file_bytes")}, indent=1))


if __name__ == "__main__":
    main()
