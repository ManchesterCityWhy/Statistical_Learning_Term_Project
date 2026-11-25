"""Shared pieces for the round-3 scripts (entropy balancing and the oracle analysis).

Both scripts reuse the evaluation split of round 2 (``runs/expanded_seed42/indices.npz``) so that
their numbers are directly comparable with ``experiments/round2_seed42``. If that file is missing,
``fixed_split`` regenerates it with exactly the random calls of ``experiments/expanded_search.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from t1_pipeline.data import Stage, board_spec
from t1_pipeline.evaluation import calibrated_score, metric_panel

DATA = Path("data/T1")
RUNS = Path("runs")
ROUND2_SPLIT = RUNS / "expanded_seed42" / "indices.npz"
SPLIT_KEYS = ("reference", "development", "ceiling", "confirmation", "confirmation_ceiling")


def open_stage(name: str, board: str = "T1:val") -> Stage:
    _, genes = board_spec(board)
    return Stage.open(DATA / f"{name}_RNA.h5ad", genes)


def fixed_split(source: Stage, target: Stage) -> tuple[dict, str]:
    """Return the round-2 reference/development/confirmation rows.

    Order of random calls copied from expanded_search.py: permutation of the target stage first,
    then the reference draw from the source stage, both from default_rng(42).
    """
    if ROUND2_SPLIT.exists():
        saved = np.load(ROUND2_SPLIT)
        return {key: saved[key] for key in SPLIT_KEYS}, f"reused {ROUND2_SPLIT}"
    rng = np.random.default_rng(42)
    order = rng.permutation(target.n_cells)
    n = round(.1 * target.n_cells) // 2
    reference = rng.choice(source.n_cells, round(.1 * source.n_cells), replace=False)
    split = {"reference": reference, "development": order[:n], "ceiling": order[n:2 * n],
             "confirmation": order[2 * n:3 * n], "confirmation_ceiling": order[3 * n:4 * n]}
    return split, "regenerated with the round-2 random calls"


class Evaluator:
    """Scores predictions on one target half against the matching ceiling half.

    Floor = the reference cells themselves (copy_last), ceiling = the disjoint target half,
    both recomputed locally exactly as in round 2. These are local development scores, not
    leaderboard scores.
    """

    def __init__(self, reference: np.ndarray, target: np.ndarray, ceiling_half: np.ndarray):
        self.reference, self.target = reference, target
        self.floor = metric_panel(reference, target, reference)
        self.ceiling = metric_panel(ceiling_half, target, reference)

    def __call__(self, prediction: np.ndarray) -> dict:
        panel = metric_panel(prediction, self.target, self.reference)
        scored = calibrated_score(panel, self.floor, self.ceiling)
        return {**panel, "score": scored["local_score"],
                **{f"skill_{k}": v for k, v in scored["skills"].items()}}


def weighted_sample_without_replacement(weights: np.ndarray, size: int, rng) -> np.ndarray:
    """Efraimidis-Spirakis draw: distinct rows, inclusion favouring large weights."""
    weights = np.asarray(weights, dtype=np.float64)
    if size > np.count_nonzero(weights > 0):
        raise ValueError("Fewer positive-weight cells than requested")
    keys = rng.exponential(size=len(weights)) / np.maximum(weights, 1e-300)
    keys[weights <= 0] = np.inf
    return np.sort(np.argpartition(keys, size - 1)[:size])


def count_scale_mean(stage: Stage, rows=None, batch: int = 512) -> np.ndarray:
    """Mean of expm1(log1p-normalised X), i.e. the pseudobulk in counts per 10,000."""
    rows = np.arange(stage.n_cells) if rows is None else np.asarray(rows)
    total = np.zeros(len(stage.order), dtype=np.float64)
    for start in range(0, len(rows), batch):
        total += np.expm1(stage.dense(rows[start:start + batch])).sum(axis=0, dtype=np.float64)
    return total / len(rows)


def harmonise_label(label: str) -> str:
    """Strip numbered sub-clusters (``Endocardium-1`` -> ``Endocardium``), per the data page caution."""
    head, _, tail = label.rpartition("-")
    return head if head and tail.isdigit() else label
