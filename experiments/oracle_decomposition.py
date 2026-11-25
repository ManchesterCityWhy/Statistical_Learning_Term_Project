"""Where does the E9.5 -> E10.5 error come from? An oracle decomposition (analysis only).

Every condition below deliberately uses E10.5 truth, so none of them is a prediction method and
none may be submitted. They answer the question the report needs: if a model got one component of
development exactly right, how much of the local score would it recover?

    copy_last                 E9.5 reference cells, unchanged (score 50 by construction)
    oracle_global_shift       copy_last + the true change of the overall mean, clipped at 0
    oracle_type_composition   E9.5 cells resampled to the true E10.5 cell-type proportions
    oracle_state_matching     E9.5 cells resampled to the true E10.5 cell *states*
                              (nearest E9.5 neighbour of each E10.5 cell in a joint PCA space):
                              the best that any re-weighting of E9.5 cells can do
    oracle_composition_shift  oracle_type_composition + each type's true mean change
    oracle_pool_sample        random E10.5 cells (sanity check, close to the ceiling)
    round2_best               the selected round-2 model (if its sampled rows are available)

"True" quantities come from an E10.5 pool that excludes the development, ceiling, confirmation
and confirmation-ceiling cells, so an oracle never copies the cells it is scored on.
Labels are harmonised by removing numbered sub-clusters (Endocardium-1 -> Endocardium); E10.5
types with no E9.5 counterpart are reported as "novel" mass that no E9.5 resampling can supply.

    python -m experiments.oracle_decomposition            # writes runs/oracle_decomposition_seed42/
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round3_common import (
    RUNS, Evaluator, fixed_split, harmonise_label, open_stage, weighted_sample_without_replacement,
)
from t1_pipeline.data import board_spec, write_json

SEED = 42
POOL_CELLS = 4000
ROUND2_ROWS = RUNS / "composition_refined_seed42" / "sampled_rows.npz"


def type_shares(labels):
    values, counts = np.unique(labels, return_counts=True)
    return pd.Series(counts / counts.sum(), index=values)


def resample_by_type(source_labels, target_shares, size, rng):
    present = type_shares(source_labels)
    weights = np.zeros(len(source_labels))
    for label, share in target_shares.items():
        if label in present.index:
            weights[source_labels == label] = share / (source_labels == label).sum()
    return weighted_sample_without_replacement(weights, size, rng)


def joint_space(stage, pool, seed):
    """Shared 30-PC space of E9.5 and pool cells on the 2,000 most variable genes.

    Fitted on 3,000 E9.5 cells plus the pool; all E9.5 cells are then projected in batches,
    so the full E9.5 matrix is never held densely.
    """
    rng = np.random.default_rng(seed)
    fit_rows = np.sort(rng.choice(stage.n_cells, min(3000, stage.n_cells), replace=False))
    fit = np.vstack([stage.dense(fit_rows), pool])
    genes = np.argsort(fit.var(axis=0))[-2000:]
    pca = PCA(n_components=30, svd_solver="randomized", random_state=seed).fit(fit[:, genes])
    batches = np.array_split(np.arange(stage.n_cells), max(1, stage.n_cells // 1024))
    source_latent = np.vstack([pca.transform(stage.dense(rows)[:, genes]) for rows in batches])
    return source_latent, pca.transform(pool[:, genes])


def main():
    parser = argparse.ArgumentParser(description="Oracle decomposition on the round-2 split")
    parser.add_argument("--output", default="oracle_decomposition_seed42")
    parser.add_argument("--with-confirmation", action="store_true")
    args = parser.parse_args()
    out = RUNS / args.output
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)

    early, source, future = open_stage("E8.5"), open_stage("E9.5"), open_stage("E10.5")
    split, split_note = fixed_split(source, future)
    used = np.concatenate([split[k] for k in ("development", "ceiling", "confirmation", "confirmation_ceiling")])
    unused = np.setdiff1d(np.arange(future.n_cells), used)
    # At most half of the unused cells form the "truth" pool; the rest feed the pool-sample check.
    pool_rows = np.sort(rng.choice(unused, min(POOL_CELLS, len(unused) // 2), replace=False))

    reference = source.dense(split["reference"])
    size = len(reference)
    evaluators = {"development": Evaluator(reference, future.dense(split["development"]),
                                           future.dense(split["ceiling"]))}
    if args.with_confirmation:
        evaluators["confirmation"] = Evaluator(reference, future.dense(split["confirmation"]),
                                               future.dense(split["confirmation_ceiling"]))

    pool = future.dense(pool_rows)
    source_mean, pool_mean = source.mean(), pool.mean(axis=0)
    predictions = {"copy_last": reference,
                   "oracle_global_shift": np.maximum(reference + (pool_mean - source_mean), 0),
                   "oracle_pool_sample": future.dense(np.sort(rng.choice(
                       np.setdiff1d(unused, pool_rows), min(size, len(unused) - len(pool_rows)), replace=False)))}

    # Cell-state oracle: how many E10.5 pool cells pick each E9.5 cell as nearest neighbour.
    source_latent, pool_latent = joint_space(source, pool, SEED)
    nearest = NearestNeighbors(n_neighbors=1).fit(source_latent).kneighbors(pool_latent, return_distance=False)
    matches = np.bincount(nearest.ravel(), minlength=source.n_cells).astype(float)
    # A floor of 1e-3 keeps enough distinct rows when fewer than `size` cells are ever matched.
    predictions["oracle_state_matching"] = source.dense(
        weighted_sample_without_replacement(matches + 1e-3, size, rng))

    composition = {}
    labels = {"E8.5": early.labels, "E9.5": source.labels, "E10.5": future.labels}
    if all(v is not None for v in labels.values()):
        harmonised = {k: np.array([harmonise_label(x) for x in v]) for k, v in labels.items()}
        src, pool_labels = harmonised["E9.5"], harmonised["E10.5"][pool_rows]
        pool_share = type_shares(pool_labels)
        novel = pool_share[~pool_share.index.isin(np.unique(src))]
        rows = resample_by_type(src, pool_share, size, rng)
        predictions["oracle_type_composition"] = source.dense(rows)
        shifted = predictions["oracle_type_composition"].copy()
        for label in np.unique(src[rows]):
            mask, truth = src[rows] == label, pool_labels == label
            if truth.sum() >= 20:
                delta = pool[truth].mean(axis=0) - source.mean(np.flatnonzero(src == label))
                shifted[mask] = np.maximum(shifted[mask] + delta, 0)
        predictions["oracle_composition_shift"] = shifted
        composition = {
            "stage_type_shares": {k: type_shares(v).round(4).to_dict() for k, v in harmonised.items()},
            "e105_pool_share_in_types_absent_at_e95": float(novel.sum()),
            "e105_types_absent_at_e95": novel.round(4).to_dict(),
            "raw_e105_labels": sorted(set(labels["E10.5"])),
        }
    else:
        composition = {"note": "celltype labels missing in at least one stage; type oracles skipped"}

    if ROUND2_ROWS.exists():
        _, genes = board_spec("T1:val")
        trend, _ = load_trend(genes)
        old = np.load(ROUND2_ROWS)["deconv3_mix1_bulk3"]
        predictions["round2_best"] = apply_trend(source.dense(old), trend, 3 / 5, False, SEED)

    records = []
    for name, prediction in predictions.items():
        for split_name, evaluator in evaluators.items():
            records.append({"condition": name, "split": split_name, **evaluator(prediction),
                            "nonzero_fraction": float((prediction > 0).mean())})
            pd.DataFrame(records).to_csv(out / "oracle_scores.csv", index=False)
            print(f"{split_name:>12}  {name:<26} {records[-1]['score']:.2f}", flush=True)
    write_json(out / "summary.json", {
        "split": split_note, "seed": SEED, "pool_cells": int(len(pool_rows)), "composition": composition,
        "detected_genes_per_cell_median": {
            "E9.5_reference": float(np.median((reference > 0).sum(axis=1))),
            "E10.5_pool": float(np.median((pool > 0).sum(axis=1)))},
        "note": "Analysis only: every oracle uses E10.5 truth and must never be submitted.",
    })
    print(pd.DataFrame(records)[["condition", "split", "score", "de_score", "de_direction", "mmd_u",
                                 "variogram"]].to_string())
    for stage in (early, source, future):
        stage.close()


if __name__ == "__main__":
    main()
