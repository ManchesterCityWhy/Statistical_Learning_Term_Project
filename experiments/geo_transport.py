"""Experimental T1 interpolation towards one permitted E14.5 sample.

Run with python -m experiments.geo_transport from the repository root.
Only the E14.5 cells of the documented GSE150817 CELLxGENE release are used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from t1_pipeline.cli import (
    check_hash, load_config, make_prediction_file, open_stages, output_dir, run_manifest,
)
from t1_pipeline.data import ROOT, board_spec, file_hash, validate_prediction, write_json
from t1_pipeline.evaluation import calibrated_score, metric_panel, pseudobulk_diagnostics
from t1_pipeline.model import Extrapolator

SAMPLE = "GSE150817"
PAPER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC13602375/"
COLLECTION = "https://cellxgene.cziscience.com/collections/613f5480-4957-4f80-b804-0e2b85ac454c"
EXTERNAL_AGE = 14.5
EXTERNAL_DIR = ROOT / "data" / "external" / SAMPLE
CANDIDATES = {
    "geo_transport_05": (0.5, True),
    "geo_transport_10": (1.0, True),
    "geo_transport_15": (1.5, True),
    "geo_keep_support_10": (1.0, False),
    "geo_type_075": (.75, True),
}


def load_external(genes):
    matrix_path = EXTERNAL_DIR / "perinatal_heart.h5ad"
    data = ad.read_h5ad(matrix_path, backed="r")
    try:
        eligible = ((data.obs["age"] == "E14.5") & (data.obs["tissue"] == "heart")
                    & (data.obs["disease"] == "normal"))
        rows = np.flatnonzero(eligible.to_numpy())
        if len(rows) != 8889 or data.raw is None:
            raise ValueError("Unexpected E14.5 subset in the CELLxGENE release")
        symbols = data.raw.var["feature_name"].copy()
        donors = sorted(data.obs.loc[eligible, "donor_id"].astype(str).unique())
    finally:
        data.file.close()
    with h5py.File(matrix_path, "r") as f:
        counts = ad.io.sparse_dataset(f["raw/X"])[rows, :].tocsr().astype(np.float32)
    if counts.shape[1] != len(symbols):
        raise ValueError("GEO matrix and symbols disagree")
    index = pd.Index(genes).get_indexer(symbols)
    keep = np.flatnonzero(index >= 0)
    available = np.unique(index[keep])
    projection = sparse.csr_matrix(
        (np.ones(len(keep), dtype=np.float32), (keep, index[keep])),
        shape=(len(symbols), len(genes)),
    )
    counts = counts @ projection
    library = np.asarray(counts.sum(axis=1)).ravel()
    detected = counts.getnnz(axis=1)
    mt = np.array([g.lower().startswith("mt-") for g in genes])
    mt_total = np.asarray(counts[:, mt].sum(axis=1)).ravel()
    valid = (library >= 500) & (detected >= 200) & (mt_total <= .2 * library)
    counts = counts[valid].tocsr()
    if not len(available) or counts.shape[0] < 500:
        raise ValueError("Unexpectedly few usable cells or genes in external sample")
    counts = counts.multiply((10000 / library[valid])[:, None]).tocsr()
    counts.data = np.log1p(counts.data)
    return counts, available, {
        "accession": SAMPLE, "source": PAPER, "collection": COLLECTION, "stage": EXTERNAL_AGE,
        "donors": donors, "cells_before_filtering": len(rows),
        "usable_cells": counts.shape[0], "shared_genes": len(available),
        "matrix_sha256": file_hash(matrix_path),
    }


def renormalize(expression):
    for start in range(0, len(expression), 128):
        x = np.expm1(expression[start:start + 128])
        library = x.sum(axis=1, keepdims=True, dtype=np.float64)
        scale = np.divide(10000.0, library, out=np.ones_like(library), where=library > 0)
        expression[start:start + 128] = np.log1p(x * scale)
    return expression


def fit_matching_space(source, external, available, seed):
    # Use genes varying in both assays and center each cohort separately in
    # the matching space. The prediction uses the original log profiles.
    source = source[:, available].tocsr()
    external = external[:, available].tocsr()
    moments = []
    for x in (source, external):
        mean = np.asarray(x.mean(axis=0)).ravel()
        var = np.maximum(np.asarray(x.power(2).mean(axis=0)).ravel() - mean**2, 0)
        moments.append((mean, var))
    variance = np.sqrt(moments[0][1] * moments[1][1])
    columns = np.argsort(variance)[-min(1600, len(variance)):]
    scale = np.maximum(np.sqrt((moments[0][1][columns] + moments[1][1][columns]) / 2), .15)
    a = (source[:, columns].toarray() - moments[0][0][columns]) / scale
    b = (external[:, columns].toarray() - moments[1][0][columns]) / scale
    pooled = np.vstack((a, b))
    projection = PCA(n_components=min(30, len(columns) - 1), svd_solver="randomized", random_state=seed)
    latent = projection.fit_transform(pooled).astype(np.float32)
    # Reduce domination by the first one or two broad assay components.
    latent /= np.maximum(projection.explained_variance_ ** .25, .1)
    return latent[:len(a)], latent[len(a):]


def pair_donors(a, b, rng):
    cost = cdist(a, b, metric="sqeuclidean").astype(np.float32)
    nearest = cost.min(axis=1)
    confidence = np.clip(np.quantile(nearest, .75) / np.maximum(nearest, 1e-6), .15, 1)
    cost -= nearest[:, None]
    local_radius = np.partition(cost, 20, axis=1)[:, 20]
    epsilon = max(float(np.median(local_radius)), 1e-3)
    kernel = np.exp(-cost / epsilon)
    # Softly enforce reference coverage without forcing all source states
    # onto dissimilar states in the new assay.
    u = np.ones(kernel.shape[0], dtype=np.float32)
    v = np.ones(kernel.shape[1], dtype=np.float32)
    for _ in range(24):
        u = np.maximum(kernel @ v, 1e-8) ** -.65
        v = np.maximum(kernel.T @ u, 1e-8) ** -.65
    donors = np.empty(len(a), dtype=int)
    for i, row in enumerate(kernel):
        probabilities = row.astype(np.float64) * v
        donors[i] = rng.choice(len(b), p=probabilities / probabilities.sum())
    return donors, confidence.astype(np.float32)


def interpolate(parents, donors, available, fraction, transfer_support, seed, confidence):
    if not 0 <= fraction <= 1:
        raise ValueError("Transport fraction must be in [0, 1]")
    result = parents.copy()
    rng = np.random.default_rng(seed)
    for start in range(0, len(result), 128):
        stop = min(start + 128, len(result))
        a = result[start:stop, available]
        b = donors[start:stop, :][:, available].toarray()
        observed_a, observed_b = a > 0, b > 0
        share = fraction * confidence[start:stop, None]
        mixed = np.where(observed_a & observed_b, (1 - share) * a + share * b, a)
        if transfer_support:
            swap = rng.random(a.shape, dtype=np.float32) < share
            mixed = np.where(swap & (observed_a != observed_b), b, mixed)
        result[start:stop, available] = mixed
    return renormalize(result)


def load_reference_parents(stage, external, overlap, rows, seed):
    source = stage.matrix(rows)
    source = sparse.csr_matrix(source)
    rng = np.random.default_rng(seed)
    positions = rng.choice(external.shape[0], min(external.shape[0], 4500), replace=False)
    subset = external[positions]
    a, b = fit_matching_space(source, subset, overlap, seed)
    selected, confidence = pair_donors(a, b, rng)
    return source.toarray().astype(np.float32), subset[selected], confidence


def benchmark(config, out, stages, ext, overlap, metadata, seeds):
    stage = stages["E9.5"]
    early = stages["E8.5"]
    target_stage = stages["E10.5"]
    internal = Extrapolator(early, stage, config["pca_cells_per_stage"], config["pca_seed"])
    candidate = {"method": "pseudobulk_shift", "alpha": 1.0, "preserve_zeros": True, "renormalize": True}
    weighted = {**candidate, "state_weight": 0.35}
    records = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rows = rng.choice(stage.n_cells, round(.1 * stage.n_cells), replace=False)
        targets = rng.choice(target_stage.n_cells, round(.1 * target_stage.n_cells), replace=False)
        cut = len(targets) // 2
        ref, pairs, confidence = load_reference_parents(stage, ext, overlap, rows, seed)
        target = target_stage.dense(targets[:cut])
        other = target_stage.dense(targets[cut:])
        floor = metric_panel(ref, target, ref, seed)
        ceiling = metric_panel(other, target, ref, seed)
        np.savez_compressed(out / f"indices_{seed}.npz", reference=rows, truth=targets[:cut], ceiling=targets[cut:])
        typed, _ = internal.predict(candidate, len(rows), seed, h=1)
        for name in ["copy_last", "internal_type_weighted", *CANDIDATES]:
            print(f"Scoring {name}, evaluation seed {seed}", flush=True)
            if name == "copy_last":
                pred = ref
            elif name == "internal_type_weighted":
                pred, _ = internal.predict(weighted, len(rows), seed, h=1)
            else:
                strength, support = CANDIDATES[name]
                parents = typed if name == "geo_type_075" else ref
                pred = interpolate(parents, pairs, overlap, strength / (EXTERNAL_AGE - 9.5),
                                   support, seed, confidence)
            metrics = metric_panel(pred, target, ref, seed)
            record = {
                "candidate": name, "seed": seed, **metrics,
                "local_score": calibrated_score(metrics, floor, ceiling)["local_score"],
                **pseudobulk_diagnostics(pred, target, ref),
                "nonzero_fraction": float((pred > 0).mean()),
            }
            records.append(record)
            pd.DataFrame(records).to_csv(out / "metrics.csv", index=False)
    result = pd.DataFrame(records).drop(columns="seed").groupby("candidate").mean()
    result = result.sort_values("local_score", ascending=False)
    result.to_csv(out / "summary.csv")
    write_json(out / "manifest.json", {**metadata, "status": "complete", "evaluation_seeds": seeds,
                                        "candidate_settings": CANDIDATES})
    print(result.to_string())


def export(config, out, stages, ext, overlap, metadata, candidate, seed, genes):
    source = stages["E10.5"]
    strength, support = CANDIDATES[candidate]
    rows = np.random.default_rng(seed).choice(source.n_cells, 2000, replace=False)
    parents, donors, confidence = load_reference_parents(source, ext, overlap, rows, seed)
    if candidate == "geo_type_075":
        internal = Extrapolator(stages["E9.5"], source, config["pca_cells_per_stage"], config["pca_seed"])
        typed = {"method": "pseudobulk_shift", "alpha": 1.0, "preserve_zeros": True, "renormalize": True}
        parents, _ = internal.predict(typed, len(rows), seed, h=2)
    fraction = strength * (12.5 - 10.5) / (EXTERNAL_AGE - 10.5)
    prediction = interpolate(parents, donors, overlap, fraction, support, seed, confidence)
    path = out / "E12.5_prediction.h5ad"
    make_prediction_file(path, prediction, genes)
    validation = validate_prediction(path, "T1:test")
    write_json(out / "manifest.json", {**metadata, "status": "prediction_ready", "candidate": candidate,
                                        "seed": seed, "transport_fraction": fraction,
                                        "nonzero_fraction": float((prediction > 0).mean()),
                                        "file_validation": validation})
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
    ext, overlap, external = load_external(genes)
    if args.mode == "benchmark":
        names = ["E8.5", "E9.5", "E10.5"]
    else:
        names = ["E9.5", "E10.5"] if args.candidate == "geo_type_075" else ["E10.5"]
    with ExitStack() as stack:
        stages = open_stages(stack, config, names, genes)
        hashes = {name: file_hash(stage.path) for name, stage in stages.items()}
        for name, digest in hashes.items():
            check_hash(name, digest)
        out = output_dir(args.output)
        metadata = {**run_manifest(config), "experimental_code_sha256": file_hash(Path(__file__)),
                    "input_sha256": hashes, "external_data": external, "status": "running"}
        write_json(out / "manifest.json", metadata)
        print(f"External sample {SAMPLE}: {ext.shape[0]} cells, {len(overlap)} common genes", flush=True)
        if args.mode == "benchmark":
            benchmark(config, out, stages, ext, overlap, metadata, args.seeds)
        else:
            export(config, out, stages, ext, overlap, metadata, args.candidate, args.seeds[0], genes)


if __name__ == "__main__":
    main()
