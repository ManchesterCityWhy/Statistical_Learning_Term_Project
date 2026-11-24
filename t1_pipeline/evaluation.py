"""Use pinned veckit functions, with explicitly local floor/ceiling anchors."""
from __future__ import annotations

import numpy as np

METRICS = {"de_score": (.25, True), "de_direction": (.25, True),
           "mmd_u": (.30, False), "variogram": (.20, False)}
VECKIT_COMMIT = "cc3859ab2a2305efc4544815d3f44f660efa77d4"


def metric_panel(pred, target, reference, seed=0):
    # Import core functions, not the obsolete rbf_mmd/deg_recovery helpers.
    from common.core_metrics import de_score, de_direction, mmd_unbiased, variogram_score

    values = {"de_score": de_score(pred, target, reference)["score"],
              "de_direction": de_direction(pred, target, reference),
              "mmd_u": mmd_unbiased(pred, target, seed=seed),
              "variogram": variogram_score(pred, target, seed=seed)}
    return {k: float(v) if np.isfinite(v) else None for k, v in values.items()}


def calibrated_score(panel, floor, ceiling):
    """Official transform with locally recomputed anchors, NOT a leaderboard score."""
    skills = {}
    for key, (_, higher) in METRICS.items():
        m, f, c = panel.get(key), floor.get(key), ceiling.get(key)
        if f is None or c is None or abs(f - c) < 1e-12:
            raise ValueError(f"No usable local calibration anchors for {key}; retain raw metrics")
        if m is None:
            skills[key] = 0.0
            continue
        distance = c - m if higher else m - c
        skills[key] = 1.0 if distance <= 0 else abs(f - c) / (abs(f - c) + distance)
    return {"local_score": 100 * sum(METRICS[k][0] * value for k, value in skills.items()), "skills": skills}


def pseudobulk_diagnostics(pred, target, reference):
    predicted_mean = pred.mean(axis=0, dtype=np.float64)
    true_mean = target.mean(axis=0, dtype=np.float64)
    ref_mean = reference.mean(axis=0, dtype=np.float64)
    delta_p, delta_t = predicted_mean - ref_mean, true_mean - ref_mean
    denominator = np.linalg.norm(delta_p) * np.linalg.norm(delta_t)
    return {"mean_rmse": float(np.sqrt(np.mean((predicted_mean - true_mean) ** 2))),
            "change_cosine_diagnostic": float(delta_p @ delta_t / denominator) if denominator else 0.0}
