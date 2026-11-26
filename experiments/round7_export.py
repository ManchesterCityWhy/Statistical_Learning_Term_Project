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
from experiments.round6_common import REAL, SEED, Mem, depth_of, lineage_of, thin_rows
from experiments.round6_config import heart_libraries
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.cli import make_prediction_file
from t1_pipeline.data import board_spec, file_hash, validate_prediction, write_json

CODE = ("round7_export.py", "round7_method.py", "round7_eval.py", "round7_probe.py", "round6_method.py", "round6_anchor.py",
        "round6_common.py", "round6_config.py", "round6_s1.py", "round6_final.py", "round6_suite.py", "round5_anchor.py", "round5_common.py",
        "round5_exp4.py", "round5_exp2.py", "round4_direction.py", "bulk_trend.py", "geo_transport.py",
        "../t1_pipeline/data.py", "../t1_pipeline/cli.py", "../t1_pipeline/evaluation.py", "../t1_pipeline/model.py",
        "../t1_pipeline/distribution.py")


def cells_hash(cells):
    return hashlib.sha256(np.ascontiguousarray(cells, dtype=np.float32).tobytes()).hexdigest()


THIN_SEED = 985


def thinned_source(source, share, libs=("3",), seed=THIN_SEED):
    """A copy of the source stage in which every cell of the given libraries keeps each of its molecules with probability
    `share` (binomial thinning of the UMI counts, then log1p of counts per 10k again). share = 1 returns the source itself.
    Round 8, afternoon of 11-26: the cells of the E12.5 sample are assumed to be somewhat sparser than E10.5 library 3."""
    if share >= 1.0:
        return source, None
    import copy
    from scipy import sparse
    rows = np.flatnonzero(np.isin(source.lib, list(libs)))
    other = np.setdiff1d(np.arange(source.n_cells), rows)
    x = source.X[rows]
    depth = depth_of(x)
    t = thin_rows(x, share * depth, np.random.default_rng(seed))
    back = np.empty(source.n_cells, dtype=np.int64)
    back[np.concatenate([other, rows])] = np.arange(source.n_cells)
    out = copy.copy(source)
    out.X = sparse.vstack([source.X[other], t]).tocsr()[back].astype(np.float32).tocsr()
    after = depth_of(out.X[rows])
    info = {"share_of_molecules_kept": share, "libraries": list(libs), "cells_thinned": int(len(rows)), "seed": seed,
            "median_umi_before": float(np.median(depth)), "median_umi_after": float(np.median(after)),
            "median_genes_detected_before": float(np.median(np.diff(x.indptr))), "median_genes_detected_after": float(np.median(np.diff(out.X[rows].indptr)))}
    return out, info


def build(source, ax, f, pos, n, seed=SEED, return_info=False, level="type", dev=0.0, prev=None, days=4.0, trend=None, trend_share=0.0, k_prof=10,
          pool=None, real_share=0.0, shared=0.0, dev_kind="dev", shift_mode="scale", mix=0.0, mix_low=None, mix_k=20):
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
    assert not (pos and shared), "pos (lineage-specific forward shift) and shared (forward shift along the shared part) are alternatives"
    # shared > 0 (round 8): all cells of the lineages present in the anchor move forward along the part of the anchor / source
    # log-ratio that all lineages share, by the share `shared` of it (only in genes the cell type expresses itself)
    fkw = dict(fwd=shared, fwd_kind="dc", bwd=1.0) if shared else dict(fwd=pos, bwd=1.0 - pos)
    out = predict7(source, main, anchor, n, f, harmonise=1.0, pool_k="resample" if f > 0 else 1,
                   level=level, seed=seed, return_info=True, trend=trend, trend_share=trend_share, k_prof=k_prof, real_share=real_share,
                   dev_kind=dev_kind, shift_mode=shift_mode, mix=mix, mix_low=mix_low, mix_k=mix_k, **fkw, **dkw)
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
    parser.add_argument("--real-share", type=float, default=0.0, help="share of the picked anchor nuclei represented by a real E10.5 cell of the assigned cell type (1 = no synthetic nucleus)")
    parser.add_argument("--shared", type=float, default=0.0, help="forward shift along the part of the anchor / E10.5 log-ratio shared by all lineages, as a share of it (round 8)")
    parser.add_argument("--dev-kind", default="dev", choices=("dev", "devw"), help="two-witness rule on the lineage-specific part of the ratio (dev) or on the whole ratio (devw)")
    parser.add_argument("--shift-mode", default="scale", choices=("scale", "counts"))
    parser.add_argument("--mix", type=float, default=0.0, help="molecular mixture: share of a real cell's molecules replaced by molecules of an anchor nucleus of its cell type (round 8)")
    parser.add_argument("--mix-low", type=float, default=None, help="confine the mixture to genes the cell type has below this many counts per 10k at E10.5")
    parser.add_argument("--mix-k", default="20", help="anchor nuclei pooled into the partner profile, or 'auto' (measured molecules only)")
    parser.add_argument("--thin", type=float, default=1.0, help="share of its molecules every cell of the heart library keeps before the recipe runs (binomial thinning; 1 = no thinning)")
    parser.add_argument("--name", default="")
    parser.add_argument("--selfcheck", default="", help="scenario table of round7_eval (development) holding this configuration")
    parser.add_argument("--selfcheck-config", default="")
    args = parser.parse_args()
    name = args.name or (f"round7_heartlib_f{args.f:g}_pos{args.pos:g}_{args.level}_k{args.k_prof}_dev{args.dev:g}_trend{args.trend:g}"
                         + (f"_real{args.real_share:g}" if args.real_share else "") + (f"_shared{args.shared:g}" if args.shared else "")
                         + ("_devw" if args.dev_kind == "devw" else "") + ("_counts" if args.shift_mode == "counts" else "")
                         + (f"_mix{args.mix:g}" + (f"low{args.mix_low:g}" if args.mix_low is not None else "") + (f"k{args.mix_k}" if args.mix_k != "20" else "") if args.mix else "")
                         + (f"_thin{args.thin:g}" if args.thin < 1.0 else ""))
    out = REAL / "predictions" / name
    out.mkdir(parents=True, exist_ok=False)
    spec, genes = board_spec("T1:test")
    assert genes == board_spec("T1:val")[1]
    ax, anchor_info = load_anchor(genes)
    source = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    source, thin_info = thinned_source(source, args.thin)
    previous = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes) if args.dev else None
    prev = (previous, np.arange(previous.n_cells)) if args.dev else None
    trend, bulk_info = (None, None)
    if args.trend:
        from experiments.bulk_trend import load_trend
        trend, bulk_info = load_trend(genes)
    kw = dict(level=args.level, dev=args.dev, prev=prev, days=4.0, trend=trend, trend_share=args.trend, k_prof=args.k_prof, real_share=args.real_share,
              shared=args.shared, dev_kind=args.dev_kind, shift_mode=args.shift_mode, mix=args.mix, mix_low=args.mix_low,
              mix_k=args.mix_k if args.mix_k == "auto" else int(args.mix_k))
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
    n_base = len(rows)                       # all real E10.5 cells: the drawn base and the real stand-ins
    n_drawn = int(info["n_drawn_base"])      # the base drawn in the natural composition
    stand_for = info["stand_in_for"]         # anchor nuclei whose assigned cell type a real stand-in cell represents
    ckw = {k: v for k, v in dkw.items() if k != "dev"}
    ctx = anchor.context(source, main, "min", args.level, **ckw)
    genes_detected = (pred > 0).sum(axis=1)
    mixture = None
    if args.mix:
        from experiments.round7_method import mixed_cells
        _, minfo = mixed_cells(source, rows, anchor, ctx, args.mix, SEED, k=kw["mix_k"], guard=True, return_info=True, low=args.mix_low)
        lab = source.labels[rows]
        low_counts = {}
        if args.mix_low is not None:
            for t in np.unique(lab[minfo["partner"] >= 0]):
                m = ctx["source_means"].get("type:" + t, ctx["source_means"].get(lineage_of(np.array([t]))[0]))
                o = ctx["own"].get("type:" + t, ctx["own"].get(lineage_of(np.array([t]))[0]))
                low_counts[str(t)] = int(((m < args.mix_low) & o).sum())
        mixture = {"cells_mixed": int(minfo["mixed_cells"]), "cells_left_unmixed": int((minfo["partner"] < 0).sum()),
                   "unmixed_by_lineage": pd.Series(lineage_of(lab[minfo["partner"] < 0])).value_counts().to_dict(),
                   "distinct_partner_nuclei": int(len(np.unique(minfo["partner"][minfo["partner"] >= 0]))),
                   "genes_in_the_mixture_by_cell_type": low_counts or "all genes the cell type expresses itself"}
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
                  + ("; a share real_share of the nuclei is not synthesised but represented by a real E10.5 cell of the assigned cell type "
                     "(same cell-type composition; with real_share = 1 the file holds real E10.5 cells only)" if args.real_share else "")
                  + ("; base additionally moved forward along the lineage log-ratio (pos > 0)" if args.pos else "")
                  + ("; all cells of the lineages present in the anchor additionally moved forward along the part of the anchor / E10.5 "
                     "log-ratio that all lineages share (share `shared` of it, only in genes the cell type expresses itself)" if args.shared else "")
                  + ("; molecular mixture: every real cell is paired with an anchor nucleus assigned to its cell type (to its lineage where fewer "
                     "than 50 anchor nuclei are assigned to the type; cells of the lineage 'Other', cells whose lineage also has fewer than 50 "
                     "anchor nuclei, and cells of a type and lineage with fewer than 50 E10.5 cells are left unmixed); "
                     + ("for the genes its cell type has below `mix_low` counts per 10k at E10.5, " if args.mix_low is not None else "")
                     + "the cell keeps each of its molecules with probability 1 - mix and receives anchor molecules at the rate mix x depth x "
                       "(share of the gene in the partner's profile); genes whose mean in the cell type is below one twentieth of the largest "
                       "lineage mean at E10.5 (mostly markers of other lineages) are removed from the profile first and are not received" if args.mix else "")
                  + ("; before all of this, every cell of the heart library is thinned: each of its molecules is kept with probability `thin` "
                     "(the E12.5 cells are assumed to be somewhat sparser than E10.5 library 3)" if args.thin < 1.0 else ""),
        "configuration": {"f": args.f, "pos": args.pos, "fwd": args.pos, "bwd": 1.0 - args.pos, "harmonise": 1.0, "mode": "min",
                          "level": args.level, "dev": args.dev, "days_to_anchor": 4.0, "previous_stage": "E9.5 (all cells)" if args.dev else None,
                          "bulk_trend_share": args.trend, "real_share": args.real_share, "shared_forward_share": args.shared,
                          "two_witness_on": {"dev": "lineage-specific part of the ratio", "devw": "whole ratio"}[args.dev_kind], "shift_mode": args.shift_mode,
                          "mixture_share": args.mix, "mixture_low_gene_threshold_per_10k": args.mix_low, "mixture_partner_profile": args.mix_k if args.mix else None,
                          "anchor_form": f"resample (k_prof = {args.k_prof}, multinomial at source-lineage depth)",
                          **({"thin": args.thin} if args.thin < 1.0 else {})},
        "seed": SEED, "cells_total": int(len(pred)), "base_cells": int(n_drawn), "stand_in_cells": int(n_base - n_drawn),
        "real_cells_total": int(n_base), "anchor_cells": int(len(picked)),
        "real_cells": {"drawn_in_natural_composition": n_drawn, "stand_ins_for_anchor_nuclei": int(n_base - n_drawn),
                       "distinct_E10.5_rows": int(len(np.unique(rows))), "synthetic_nuclei": int(len(picked)),
                       "stand_in_celltype_shares": pd.Series(source.labels[rows[n_drawn:]]).value_counts(normalize=True).round(4).to_dict(),
                       "requested_celltype_shares": pd.Series(ctx["near"][stand_for]).value_counts(normalize=True).round(4).to_dict() if len(stand_for) else {},
                       "stand_ins_of_another_type_than_requested": int((source.labels[rows[n_drawn:]] != ctx["near"][stand_for]).sum()) if len(stand_for) else 0,
                       "median_genes_detected_stand_ins": float(np.median(genes_detected[n_drawn:n_base])) if n_base > n_drawn else None,
                       "stand_in_lineage_shares": pd.Series(lin_base[n_drawn:]).value_counts(normalize=True).round(4).to_dict(),
                       "median_umi_of_stand_in_source_rows": float(np.median(depth_of(source.X[rows[n_drawn:]]))) if n_base > n_drawn else None,
                       "all_real_cells_celltype_shares": pd.Series(source.labels[rows]).value_counts(normalize=True).round(4).to_dict(),
                       "all_real_cells_lineage_shares": pd.Series(lin_base).value_counts(normalize=True).round(4).to_dict(),
                       "median_umi_of_all_real_source_rows": float(np.median(depth_of(source.X[rows])))},
        "heart_libraries": libs,
        # "base" = the cells drawn in the natural composition (rows 0 .. n_drawn - 1 of the file); the real stand-ins, if
        # any, are described under "real_cells" (rows n_drawn .. n_base - 1)
        "base": {"cells": n_drawn,
                 "library_shares": pd.Series(source.lib[rows[:n_drawn]]).value_counts(normalize=True).round(4).to_dict(),
                 "celltype_shares": pd.Series(source.labels[rows[:n_drawn]]).value_counts(normalize=True).round(4).to_dict(),
                 "lineage_shares": pd.Series(lin_base[:n_drawn]).value_counts(normalize=True).round(4).to_dict(),
                 "median_umi_of_source_rows": float(np.median(depth_of(source.X[rows[:n_drawn]])))},
        "anchor": {"lineage_shares": pd.Series(lin_anchor).value_counts(normalize=True).round(4).to_dict(),
                   "nearest_E10.5_celltype_shares": pd.Series(ctx["near"][picked]).value_counts(normalize=True).round(4).to_dict(),
                   "median_umi_raw_nuclei": float(np.median(anchor.depth[picked])) if len(picked) else None,
                   "median_genes_detected_resequenced": float(np.median(genes_detected[n_base:])) if len(picked) else None},
        "file": {"lineage_shares_total": total_lineage,
                 "median_genes_detected": float(np.median(genes_detected)),
                 "median_genes_detected_base": float(np.median(genes_detected[:n_drawn])),
                 "median_genes_detected_real_cells": float(np.median(genes_detected[:n_base])),
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
                              for k, v in ctx.get(args.dev_kind, {}).items()},
        "shared_forward_shift": ({"share": args.shared, "genes_in_shared_part": int((c != 0).sum()),
                                  "genes_moved_by_lineage_or_type": {k: int((v != 0).sum()) for k, v in ctx["dc"].items() if not k.startswith("type:")},
                                  "median_abs_log_factor_where_moved": float(np.median(np.abs(args.shared * c[c != 0])))} if args.shared else None),
        "molecular_mixture": mixture,
        **({"thinning": thin_info} if thin_info is not None else {}),
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
        if "seed" in stored.columns:                 # tables with several seeds: the export uses the project seed
            stored = stored[stored["seed"] == SEED]
        stored = stored[stored["config"] == args.selfcheck_config].set_index("scenario")["score"]
        e10, pool, _, evs = scenario_setup("development", genes)
        cfg = C(args.selfcheck_config, rho_f=args.f / .5, rho_t=(args.shared or args.pos) / .5, bwd=1.0 if args.shared else "interp", pool_k="resample",
                level=args.level, rho_dev=args.dev / .5, days_trend=args.trend * 5.0 / 2.0, k_prof=args.k_prof, real_share=args.real_share,
                fwd_kind="dc" if args.shared else "d", dev_kind=args.dev_kind, shift_mode=args.shift_mode, rho_mix=args.mix / .5, mix_low=args.mix_low,
                mix_k=args.mix_k if args.mix_k == "auto" else int(args.mix_k))
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
    # row bookkeeping for independent checks (indices only, no expression values): which E10.5 row every real cell of the
    # file is, which anchor nuclei the stand-ins represent, which nuclei were synthesised, and the mixture partners
    np.savez(out / "provenance.npz", real_rows=np.asarray(rows, dtype=np.int64), n_drawn=np.int64(n_drawn),
             stand_in_for=np.asarray(stand_for, dtype=np.int64), synthetic_from=np.asarray(picked, dtype=np.int64),
             mixture_partner=np.asarray(minfo["partner"], dtype=np.int64) if args.mix else np.array([], dtype=np.int64))
    manifest["provenance_sha256"] = file_hash(out / "provenance.npz")
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("name", "configuration", "cells_total", "base_cells", "stand_in_cells", "anchor_cells", "file", "validation",
                                               "prediction_cells_sha256", "file_sha256", "file_bytes")}, indent=1))


if __name__ == "__main__":
    main()
