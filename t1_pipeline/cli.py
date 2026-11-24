"""Run from the repository root: python -m t1_pipeline --help."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd

from .data import ROOT, TRAIN_SHA256, Stage, board_spec, file_hash, sync_specs, validate_prediction, write_json
from .evaluation import VECKIT_COMMIT, METRICS, calibrated_score, metric_panel, pseudobulk_diagnostics
from .model import Extrapolator


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def run_manifest(config):
    import common.core_metrics as core

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
            "git_head": git.stdout.strip() if git.returncode == 0 else None,
            "code_sha256": {p.name: file_hash(p) for p in sorted((ROOT / "t1_pipeline").glob("*.py"))},
            "python": platform.python_version(), "veckit_commit": VECKIT_COMMIT,
            "veckit_core_sha256": file_hash(Path(core.__file__)),
            "packages": {name: version(name) for name in ["numpy", "scipy", "pandas", "scikit-learn", "anndata", "h5py", "veckit"]}}


def open_stages(stack, config, names, genes):
    result = {}
    for name in names:
        path = resolve(config["data"][name])
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}. Set data paths in configs/t1.json.")
        stage = Stage.open(path, genes)
        stack.callback(stage.close)
        result[name] = stage
    return result


def check_hash(name, digest):
    if name in TRAIN_SHA256 and digest != TRAIN_SHA256[name]:
        raise ValueError(f"{name} SHA-256 differs from the official training release; check the download")


def output_dir(path):
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=False)
    return p


def doctor(args):
    config = load_config(args.config)
    files = [{"stage": stage, "path": str(resolve(path)), "exists": resolve(path).is_file()}
             for stage, path in config["data"].items()]
    spec_ready = (ROOT / "specs/t1/boards.json").is_file()
    result = {"data": files, "specification_ready": spec_ready,
              "ready_for_real_benchmark": spec_ready and all(item["exists"] for item in files),
              "note": "Presence checks only; inspect validates values, genes and training checksums."}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        write_json(resolve(args.output), result)
    return 0 if result["ready_for_real_benchmark"] else 2


def inspect_data(args):
    config = load_config(args.config)
    _, genes = board_spec("T1:val")
    with ExitStack() as stack:
        stages = open_stages(stack, config, ["E8.5", "E9.5", "E10.5"], genes)
        summaries = {}
        for name, stage in stages.items():
            summaries[name] = stage.inspect()
            check_hash(name, summaries[name]["sha256"])
        write_json(resolve(args.output), {"stages": summaries, "note": "E10.5 is development data; E12.5 truth is never used."})
    print(f"Data checks written: {resolve(args.output)}")


def make_prediction_file(path, prediction, genes):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    data = ad.AnnData(X=np.asarray(prediction, dtype=np.float32),
                      obs=pd.DataFrame(index=[f"pred_{i}" for i in range(len(prediction))]),
                      var=pd.DataFrame(index=genes))
    data.write_h5ad(path, compression="gzip")


def benchmark(args):
    config = load_config(args.config)
    spec, genes = board_spec("T1:val")
    n_cells = int(config["prediction_cells"])
    if not spec["min_cells"] <= n_cells <= spec["max_cells"]:
        raise ValueError("prediction_cells is outside the validation board limits")
    with ExitStack() as stack:
        stages = open_stages(stack, config, ["E8.5", "E9.5", "E10.5"], genes)
        fingerprints = {name: file_hash(stage.path) for name, stage in stages.items()}
        for name, digest in fingerprints.items():
            check_hash(name, digest)
        out = output_dir(args.output)
        manifest = run_manifest(config)
        manifest.update({"status": "running", "input_sha256": fingerprints, "split": "E8.5+E9.5 -> E10.5",
                         "score_scope": "Local development only; not an official leaderboard score."})
        write_json(out / "manifest.json", manifest)
        model = Extrapolator(stages["E8.5"], stages["E9.5"], config["pca_cells_per_stage"], config["pca_seed"])
        model.fit_pca([c["dimensions"] for c in config["candidates"] if c["method"] == "pca_shift"])
        fraction = config["evaluation_stage_fraction"]
        if not 0 < fraction <= 1:
            raise ValueError("evaluation_stage_fraction must lie in (0, 1]")
        rng = np.random.default_rng(config["evaluation_seed"])
        reference_indices = rng.choice(stages["E9.5"].n_cells, max(2, round(stages["E9.5"].n_cells * fraction)), replace=False)
        target_indices = rng.choice(stages["E10.5"].n_cells, max(4, round(stages["E10.5"].n_cells * fraction)), replace=False)
        cut = len(target_indices) // 2
        reference = stages["E9.5"].dense(reference_indices)
        target = stages["E10.5"].dense(target_indices[:cut])
        ceiling_sample = stages["E10.5"].dense(target_indices[cut:])
        np.savez_compressed(out / "evaluation_indices.npz", reference=reference_indices, truth=target_indices[:cut], ceiling=target_indices[cut:])
        seed = config["evaluation_seed"]
        floor = metric_panel(reference, target, reference, seed)
        ceiling = metric_panel(ceiling_sample, target, reference, seed)
        write_json(out / "local_anchors.json", {"floor": floor, "ceiling": ceiling,
                    "note": "Recomputed on this run's disjoint samples; not the organisers' hidden test anchors."})
        records = []
        for candidate in config["candidates"]:
            for prediction_seed in config["seeds"]:
                print(f"Evaluating {candidate['name']}, seed={prediction_seed}", flush=True)
                prediction, diagnostics = model.predict(candidate, n_cells, prediction_seed, h=1)
                raw = metric_panel(prediction, target, reference, seed)
                try:
                    local = calibrated_score(raw, floor, ceiling)
                except ValueError as error:
                    local = {"local_score": None, "calibration_error": str(error)}
                row = {"candidate": candidate["name"], "seed": prediction_seed, **raw,
                       "local_score": local["local_score"], **pseudobulk_diagnostics(prediction, target, reference),
                       "nonzero_fraction": diagnostics["prediction_nonzero_fraction"]}
                records.append(row)
                name = f"{candidate['name']}_seed{prediction_seed}"
                write_json(out / f"{name}.json", {"candidate": candidate, "seed": prediction_seed,
                           "metrics": raw, "calibration": local, "diagnostics": diagnostics})
                if args.keep_predictions:
                    make_prediction_file(out / f"{name}.h5ad", prediction, genes)
                pd.DataFrame(records).to_csv(out / "metrics.csv", index=False)
        columns = list(METRICS) + ["local_score", "mean_rmse", "change_cosine_diagnostic", "nonzero_fraction"]
        summary = pd.DataFrame(records).groupby("candidate")[columns].agg(["mean", "std"])
        summary.columns = [f"{key}_{stat}" for key, stat in summary.columns]
        summary = summary.sort_values("local_score_mean", ascending=False)
        summary.to_csv(out / "summary.csv")
        write_json(out / "pca_diagnostics.json", model.diagnostics)
        manifest["status"] = "complete"
        write_json(out / "manifest.json", manifest)
        print(summary.to_string())


def freeze(args):
    run = resolve(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Can only freeze a completed benchmark")
    config = manifest["config"]
    candidates = [c for c in config["candidates"] if c["name"] == args.candidate]
    if len(candidates) != 1:
        raise ValueError("Candidate must occur exactly once in the completed run")
    if args.seed not in config["seeds"]:
        raise ValueError("Choose a prediction seed included in the completed development run")
    path = resolve(args.output)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection: {path}")
    value = {"candidate": candidates[0], "data": config["data"],
             "pca_cells_per_stage": config["pca_cells_per_stage"], "pca_seed": config["pca_seed"],
             "prediction_cells": config["prediction_cells"], "prediction_seed": args.seed,
             "development_run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
             "development_manifest_sha256": file_hash(run / "manifest.json"),
             "input_sha256": manifest["input_sha256"], "frozen_at": datetime.now(timezone.utc).isoformat(),
             "development_code_sha256": manifest["code_sha256"], "veckit_commit": manifest["veckit_commit"],
             "development_packages": manifest["packages"], "veckit_core_sha256": manifest["veckit_core_sha256"]}
    write_json(path, value)
    print(f"Frozen selection: {path}")


def predict_final(args):
    frozen = load_config(args.selection)
    spec, genes = board_spec("T1:test")
    if not spec["min_cells"] <= frozen["prediction_cells"] <= spec["max_cells"]:
        raise ValueError("Frozen cell count does not satisfy the current T1:test specification")
    current = run_manifest(frozen)
    if current["code_sha256"] != frozen["development_code_sha256"]:
        raise ValueError("Pipeline code changed after development; rerun validation and freeze the new version")
    if current.get("packages", {}) != frozen.get("development_packages", {}):
        raise ValueError("Runtime package versions changed after development")
    if current.get("veckit_core_sha256") != frozen.get("veckit_core_sha256"):
        raise ValueError("Scorer source changed after development")
    if frozen["veckit_commit"] != VECKIT_COMMIT:
        raise ValueError("Scorer version changed after development")
    with ExitStack() as stack:
        stages = open_stages(stack, frozen, ["E9.5", "E10.5"], genes)
        for name, stage in stages.items():
            if file_hash(stage.path) != frozen["input_sha256"][name]:
                raise ValueError(f"{name} changed after model selection")
        out = output_dir(args.output)
        model = Extrapolator(stages["E9.5"], stages["E10.5"], frozen["pca_cells_per_stage"], frozen["pca_seed"])
        candidate = frozen["candidate"]
        model.fit_pca([candidate["dimensions"]] if candidate["method"] == "pca_shift" else [])
        prediction, diagnostics = model.predict(candidate, frozen["prediction_cells"], frozen["prediction_seed"], h=2)
        path = out / "E12.5_prediction.h5ad"
        make_prediction_file(path, prediction, genes)
        validation = validate_prediction(path, "T1:test")
        manifest = run_manifest(frozen)
        manifest.update({"status": "prediction_ready_not_submitted", "split": "E9.5+E10.5 -> E12.5", "h": 2,
                         "selection_sha256": file_hash(resolve(args.selection)), "diagnostics": diagnostics,
                         "file_validation": validation, "pca": model.diagnostics})
        write_json(out / "manifest.json", manifest)
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        print(f"Prediction ready: {path}. Upload and official submission are separate steps.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="T1: validate E10.5, freeze a method, extrapolate E12.5")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sync-specs", help="Fetch only public board specifications and gene names")
    p = commands.add_parser("doctor", help="Check whether real data and public specifications are present")
    p.add_argument("--config", default="configs/t1.json")
    p.add_argument("--output")
    p = commands.add_parser("inspect", help="Check official inputs and write data diagnostics")
    p.add_argument("--config", default="configs/t1.json")
    p.add_argument("--output", default="runs/data_inspection.json")
    p = commands.add_parser("benchmark", help="Fit E8.5/E9.5 only; score on E10.5")
    p.add_argument("--config", default="configs/t1.json")
    p.add_argument("--output", required=True)
    p.add_argument("--keep-predictions", action="store_true")
    p = commands.add_parser("freeze", help="Record a candidate selected from a completed development run")
    p.add_argument("--run", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="configs/frozen_selection.json")
    p = commands.add_parser("predict-final", help="Refit on E9.5/E10.5; export E12.5 at h=2")
    p.add_argument("--selection", default="configs/frozen_selection.json")
    p.add_argument("--output", required=True)
    p = commands.add_parser("validate", help="Check a prediction against a public board specification")
    p.add_argument("input")
    p.add_argument("--board", choices=["T1:val", "T1:test"], default="T1:test")
    args = parser.parse_args(argv)
    try:
        if args.command == "sync-specs":
            print(json.dumps(sync_specs(), ensure_ascii=False, indent=2))
        elif args.command == "validate":
            print(json.dumps(validate_prediction(resolve(args.input), args.board), ensure_ascii=False, indent=2))
        else:
            status = {"doctor": doctor, "inspect": inspect_data, "benchmark": benchmark,
                      "freeze": freeze, "predict-final": predict_final}[args.command](args)
            if status:
                raise SystemExit(status)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
