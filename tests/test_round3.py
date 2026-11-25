"""Fast synthetic checks for the round-3 helpers (no competition data involved)."""
import numpy as np

from experiments.round3_balance import balance_to_ess, entropy_balance, gumbel_top_k
from experiments.round3_common import harmonise_label, weighted_sample_without_replacement


def test_entropy_balance_hits_a_reachable_target_and_stays_uniform_at_zero_shift():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4000, 5))
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    logits, info = entropy_balance(z, np.zeros(5), ridge=1e-6)
    np.testing.assert_allclose(logits - logits.mean(), 0, atol=1e-6)
    target = np.array([.4, -.2, 0, .1, 0])
    logits, info = entropy_balance(z, target, ridge=1e-6)
    weights = np.exp(logits - logits.max())
    np.testing.assert_allclose(weights / weights.sum() @ z, target, atol=1e-3)
    assert info["converged"] and 0 < info["relative_ess"] < 1


def test_ridge_trades_balance_for_weight_dispersion():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(3000, 3))
    target = np.array([1.5, 0, 0])
    loose = entropy_balance(z, target, ridge=10.0)[1]
    tight = entropy_balance(z, target, ridge=1e-4)[1]
    assert loose["remaining_imbalance"] > tight["remaining_imbalance"]
    assert loose["relative_ess"] > tight["relative_ess"]


def test_weighted_sampling_returns_distinct_rows_and_skips_zero_weights():
    rng = np.random.default_rng(2)
    weights = np.r_[np.zeros(50), np.ones(100)]
    rows = weighted_sample_without_replacement(weights, 60, rng)
    assert len(set(rows)) == 60 and rows.min() >= 50


def test_label_harmonisation_only_strips_numbered_suffixes():
    assert harmonise_label("Endocardium-1") == "Endocardium"
    assert harmonise_label("OFT/RV-CM") == "OFT/RV-CM"
    assert harmonise_label("NCC-derived") == "NCC-derived"


def test_ess_budget_is_respected_even_for_an_unreachable_target():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(5000, 4))
    _, info = balance_to_ess(z, np.array([25.0, 0, 0, 0]), ess=.3)
    assert .28 <= info["relative_ess"] and info["achieved_shift_fraction"] > 0


def test_gumbel_top_k_handles_extreme_logits():
    rng = np.random.default_rng(4)
    logits = np.r_[np.full(10, 800.0), np.full(90, -800.0)]
    rows = gumbel_top_k(logits, 20, rng)
    assert len(set(rows)) == 20 and set(range(10)) <= set(rows)
