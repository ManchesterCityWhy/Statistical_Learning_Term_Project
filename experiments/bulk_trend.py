"""Transfer a replicated, cross-assay E9.5 to E14.5 cardiac RNA trend."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from t1_pipeline.cli import (
    check_hash, load_config, make_prediction_file, open_stages, output_dir, run_manifest,
)
from t1_pipeline.data import ROOT, board_spec, file_hash, validate_prediction, write_json
from t1_pipeline.evaluation import calibrated_score, metric_panel, pseudobulk_diagnostics
from t1_pipeline.model import Extrapolator
from .geo_transport import renormalize

SOURCE = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223237"
ALLOWED = (
    ("GSM6943750", "E9_Rep1", "E9.5"),
    ("GSM6943751", "E9_Rep2", "E9.5"),
    ("GSM6943752", "E9_Rep3", "E9.5"),
    ("GSM6943761", "E14_Rep1", "E14.5"),
    ("GSM6943762", "E14_Rep2", "E14.5"),
    ("GSM6943763", "E14_Rep3", "E14.5"),
)
CANDIDATES = {
    "bulk_075": (.75, False, False),
    "bulk_150": (1.5, False, False),
    "bulk_support_100": (1.0, True, False),
    "bulk_type_075": (.75, False, True),
}
TYPE_SETTINGS = {"method": "pseudobulk_shift", "alpha": 1.0, "preserve_zeros": True, "renormalize": True}


def load_trend(genes):
    mapping_path = ROOT / "data/external/GSM6211349/GSM6211349_E14_genes.tsv.gz"
    mapping = pd.read_csv(mapping_path, sep="\t", header=None, names=["ensembl", "symbol"])
    mapping = mapping.drop_duplicates("ensembl").set_index("ensembl")["symbol"]
    source_dir = ROOT / "data/external/GSE223237"
    vectors = {"E9.5": [], "E14.5": []}
    records = []
    for accession, tag, stage in ALLOWED:
        path = source_dir / f"{accession}_{tag}_counts.txt.gz"
        table = pd.read_csv(path, sep="\t", index_col=0).iloc[:, 0]
        index = table.index.astype("string").str.split(".").str[0]
        symbols = mapping.reindex(index).to_numpy()
        counts = pd.Series(table.to_numpy(dtype=float), index=symbols).groupby(level=0).sum()
        cpm = counts.reindex(genes, fill_value=0).to_numpy() * (1e6 / counts.sum())
        vectors[stage].append(cpm)
        records.append({"accession": accession, "stage": stage, "file": path.name, "sha256": file_hash(path)})
    earlier = np.stack(vectors["E9.5"])
    later = np.stack(vectors["E14.5"])
    # Only transfer reasonably expressed genes with a consistent effect across
    # the independent hearts. Low counts and variable effects shrink to zero.
    differences = np.log(later[:, None] + 2) - np.log(earlier[None, :] + 2)
    typical = np.median(differences, axis=(0, 1))
    agreement = (np.sign(differences) == np.sign(typical)).mean(axis=(0, 1))
    expression = np.maximum(np.median(earlier, axis=0), np.median(later, axis=0))
    reliability = expression / (expression + 12) * np.clip(2 * agreement - 1, 0, 1)
    trend = np.clip(typical * reliability, -2.0, 2.0).astype(np.float32)
    return trend, {"series": "GSE223237", "source": SOURCE, "samples": records,
                   "symbol_map_source": "GSM6211349 E14.5 Cell Ranger gene table only",
                   "symbol_map_sha256": file_hash(mapping_path)}


def apply_trend(cells, trend, timescale, change_support, seed):
    result = np.empty_like(cells)
    ratio = np.exp(trend * timescale)
    rng = np.random.default_rng(seed)
    if change_support:
        detection = (cells > 0).mean(axis=0)
        expected = 1 - (1 - detection) ** ratio
        ratio_keep = np.clip(expected / np.maximum(detection, 1e-6), 0, 1)
        activate = np.clip((expected - detection) / np.maximum(1 - detection, 1e-6), 0, .3)
        activate[detection < .015] = 0
        conditional = cells.mean(axis=0) / np.maximum(detection, 1e-6)
    for start in range(0, len(cells), 128):
        base = cells[start:start + 128]
        scaled = np.log1p(np.expm1(base) * ratio)
        if change_support:
            draw = rng.random(base.shape, dtype=np.float32)
            scaled[(base > 0) & (draw >= ratio_keep)] = 0
            scaled = np.where((base == 0) & (draw < activate), conditional, scaled)
        result[start:start + len(base)] = scaled
    return renormalize(result)


def predict(name, baseline, typed, trend, h, seed):
    if name == "copy_last":
        return baseline
    if name == "internal_type_norm":
        return typed
    strength, support, type_shift = CANDIDATES[name]
    return apply_trend(typed if type_shift else baseline, trend, strength * h / 5.0, support, seed)


def benchmark(config, stages, out, metadata, trend, seeds):
    early, last, future = (stages[stage] for stage in ("E8.5", "E9.5", "E10.5"))
    model = Extrapolator(early, last, config["pca_cells_per_stage"], config["pca_seed"])
    records = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rows = rng.choice(last.n_cells, round(.1 * last.n_cells), replace=False)
        target_rows = rng.choice(future.n_cells, round(.1 * future.n_cells), replace=False)
        cut = len(target_rows) // 2
        ref, target, other = last.dense(rows), future.dense(target_rows[:cut]), future.dense(target_rows[cut:])
        floor = metric_panel(ref, target, ref, seed)
        ceiling = metric_panel(other, target, ref, seed)
        typed, _ = model.predict(TYPE_SETTINGS, len(rows), seed, h=1)
        for name in ["copy_last", "internal_type_norm", *CANDIDATES]:
            print(f"Scoring {name}, evaluation seed {seed}", flush=True)
            pred = predict(name, ref, typed, trend, 1, seed)
            metrics = metric_panel(pred, target, ref, seed)
            records.append(dict(candidate=name, seed=seed, **metrics,
                                local_score=calibrated_score(metrics, floor, ceiling)["local_score"],
                                **pseudobulk_diagnostics(pred, target, ref),
                                nonzero_fraction=float((pred > 0).mean())))
            pd.DataFrame(records).to_csv(out / "metrics.csv", index=False)
    result = pd.DataFrame(records).drop(columns="seed").groupby("candidate").mean()
    result = result.sort_values("local_score", ascending=False)
    result.to_csv(out / "summary.csv")
    write_json(out / "manifest.json", {**metadata, "status": "complete", "evaluation_seeds": seeds,
                                        "candidate_settings": CANDIDATES})
    print(result.to_string())


def export(config, out, stages, metadata, candidate, trend, seed, genes):
    past, last = stages["E9.5"], stages["E10.5"]
    model = Extrapolator(past, last, config["pca_cells_per_stage"], config["pca_seed"])
    baseline = last.dense(np.random.default_rng(seed).choice(last.n_cells, 2000, replace=False))
    typed, _ = model.predict(TYPE_SETTINGS, len(baseline), seed, h=2)
    pred = predict(candidate, baseline, typed, trend, 2, seed)
    path = out / "E12.5_prediction.h5ad"
    make_prediction_file(path, pred, genes)
    validation = validate_prediction(path, "T1:test")
    write_json(out / "manifest.json", {**metadata, "status": "prediction_ready", "candidate": candidate,
                                        "seed": seed, "h": 2, "candidate_settings": CANDIDATES[candidate],
                                        "nonzero_fraction": float((pred > 0).mean()), "file_validation": validation})
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("benchmark", "export"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/t1_competitive_verify.json")
    parser.add_argument("--candidate", choices=list(CANDIDATES))
    parser.add_argument("--seeds", nargs="+", type=int, choices=[42], default=[42])
    args = parser.parse_args()
    if args.mode == "export" and (args.candidate is None or len(args.seeds) != 1):
        parser.error("export requires a candidate and exactly one seed")
    config = load_config(args.config)
    _, genes = board_spec("T1:val" if args.mode == "benchmark" else "T1:test")
    trend, external = load_trend(genes)
    names = ["E8.5", "E9.5", "E10.5"] if args.mode == "benchmark" else ["E9.5", "E10.5"]
    with ExitStack() as stack:
        stages = open_stages(stack, config, names, genes)
        hashes = {name: file_hash(stage.path) for name, stage in stages.items()}
        for name, digest in hashes.items():
            check_hash(name, digest)
        out = output_dir(args.output)
        metadata = {**run_manifest(config), "experimental_code_sha256": file_hash(Path(__file__)),
                    "input_sha256": hashes, "external_data": external, "status": "running"}
        write_json(out / "manifest.json", metadata)
        if args.mode == "benchmark":
            benchmark(config, stages, out, metadata, trend, args.seeds)
        else:
            export(config, out, stages, metadata, args.candidate, trend, args.seeds[0], genes)


if __name__ == "__main__":
    main()
