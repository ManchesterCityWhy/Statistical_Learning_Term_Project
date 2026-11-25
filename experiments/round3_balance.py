"""Round 3: entropy-balanced resampling towards a bulk-guided target pseudobulk.

Round 2 moved the population by re-weighting ~20 annotated cell types with a ridge-penalised NNLS
deconvolution. Its best solution put 42% of the weight on septum transversum and almost none on
ventricular cardiomyocytes, i.e. the label-level fit was extreme. This script asks the same
question with a classical survey-statistics tool instead:

    find cell weights w that are as close to uniform as possible (in KL divergence) while the
    weighted pseudobulk moves towards the bulk-predicted target m* = m_src * exp(s * h * trend).

That is calibration / entropy balancing (Deville and Saerndal 1992; Hainmueller 2012). The solution
is an exponential tilt w_i ∝ exp(theta' z_i), where z_i are K summary features of cell i. A ridge
term lambda makes the balance approximate, so weights stay finite when the target lies outside
what the observed cells can reach. Weights act on individual cell states, need no labels, and the
target direction is split into a part explained by cell-state composition (the K features) and a
residual that is applied as the within-cell bulk expression trend (as in round 2).

How far the weights may move is set by the share of cells that must stay effectively in use
(relative effective sample size 1/(N sum w^2) in {0.5, 0.3, 0.15}); the ridge is solved for.

Usage (from the repository root, with the round-2 inputs in place):
    python -m experiments.round3_balance benchmark            # ~24 configs, dev + confirmation
    python -m experiments.round3_balance benchmark --quick    # 6 configs, smoke test
    python -m experiments.round3_balance export --config-name s3_ess0.3_b3 --h 1

Selection uses the development split only; the confirmation split is scored once for the top
three, never searched. E10.5 truth is used only for scoring, never as an input to the weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round3_common import RUNS, Evaluator, count_scale_mean, fixed_split, open_stage
from t1_pipeline.cli import make_prediction_file
from t1_pipeline.data import board_spec, file_hash, validate_prediction, write_json

SEED = 42
ROUND2_BEST = "deconv3_mix1_bulk3"
ROUND2_ROWS = RUNS / "composition_refined_seed42" / "sampled_rows.npz"
ROUND2_EXPECTED = {"development": 60.5004, "confirmation": 59.7893}


# --------------------------------------------------------------------------- balancing
class CellFeatures:
    """Variance-stabilised count-scale features of every source cell, projected to K cell-state axes.

    Only genes that are expressed and carry a non-zero bulk trend enter (capped at ``max_genes`` by
    variance). The K axes are principal components of the source cells, so the balance can only
    move weight along directions in which cells really differ; the part of the bulk target that
    lies outside these axes is left to the within-cell expression trend. (Balancing on the raw
    target direction instead was tried on synthetic data and chased single outlier cells.)
    """

    def __init__(self, stage, trend, components=30, max_genes=4000, fit_cells=5000, seed=SEED):
        self.stage, self.trend = stage, trend
        self.mean = count_scale_mean(stage)
        informative = np.flatnonzero((self.mean >= .2) & (np.abs(trend) > .05))
        if len(informative) < 50:
            raise ValueError(f"Only {len(informative)} expressed genes carry a bulk trend; check the GEO files "
                             "and the GSM6211349 Ensembl-to-symbol table")
        rng = np.random.default_rng(seed)
        fit_rows = np.sort(rng.choice(stage.n_cells, min(fit_cells, stage.n_cells), replace=False))
        sample = self._scaled(fit_rows, informative)
        keep = np.argsort(sample.var(axis=0))[-min(max_genes, len(informative)):]
        self.genes = informative[np.sort(keep)]
        # Scaled values of every cell on the selected genes (N x <=4000 float32, ~270 MB at E9.5).
        self.scaled = np.vstack([self._scaled(rows, self.genes)
                                 for rows in np.array_split(np.arange(stage.n_cells),
                                                            max(1, stage.n_cells // 1024))])
        components = min(components, len(fit_rows) - 1, len(self.genes) - 1)
        self.pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
        self.pca.fit(self.scaled[fit_rows])
        self.matrix = self.pca.transform(self.scaled)

    def _scaled(self, rows, genes):
        counts = np.expm1(self.stage.dense(rows))[:, genes]
        return ((counts - self.mean[genes]) / np.sqrt(self.mean[genes] + 1)).astype(np.float32)

    def target_direction(self, strength):
        """Required change of the (scaled) mean so that the pseudobulk becomes m_src*exp(strength*trend)."""
        desired = self.mean * np.exp(np.clip(self.trend * strength, -2, 2))
        desired *= self.mean.sum() / desired.sum()
        return ((desired - self.mean)[self.genes] / np.sqrt(self.mean[self.genes] + 1)).astype(np.float32)

    def design(self, direction):
        """Return standardised cell features Z (N x K) and the target moment z*.

        Uniform weights give mean(Z) = 0 after centring; the target is the required mean shift
        expressed on the same (linear) axes.
        """
        z = self.matrix.astype(np.float64)
        shift = direction @ self.pca.components_.T
        centre, scale = z.mean(axis=0), np.maximum(z.std(axis=0), 1e-8)
        return (z - centre) / scale, shift / scale


def entropy_balance(z, target, ridge):
    """Solve min_theta log mean exp(z theta) - theta'target + ridge/2 |theta|^2 (convex dual).

    Returns the log-weights (logits, defined up to a constant) and balance diagnostics.
    """

    def objective(theta):
        logits = z @ theta
        top = logits.max()
        weights = np.exp(logits - top)
        total = weights.sum()
        weights /= total
        value = top + np.log(total / len(z)) - theta @ target + .5 * ridge * theta @ theta
        gradient = weights @ z - target + ridge * theta
        return value, gradient

    result = minimize(objective, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "gtol": 1e-8})
    logits = z @ result.x
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    reached = weights @ z
    return logits, {
        "ridge": float(ridge),
        "converged": bool(result.success),
        "relative_ess": float(1 / np.sum(weights ** 2) / len(weights)),
        "max_weight_times_n": float(weights.max() * len(weights)),
        "target_norm": float(np.linalg.norm(target)),
        "remaining_imbalance": float(np.linalg.norm(reached - target)),
        "achieved_shift_fraction": float(reached @ target / max(target @ target, 1e-12)),
    }


def balance_to_ess(z, target, ess, low=1e-4, high=1e4, steps=30):
    """Smallest ridge whose weights keep at least ``ess`` x N effective cells.

    The relative effective sample size 1/(N sum w^2) grows with the ridge, so a bisection on
    log(ridge) finds the most balanced solution that still uses a fixed share of the cells.
    This replaces a hand-set ridge, whose effect depends on how far the target is.
    """
    logits, info = entropy_balance(z, target, low)
    if info["relative_ess"] >= ess:
        return logits, info
    a, b = np.log(low), np.log(high)
    best = entropy_balance(z, target, high)
    for _ in range(steps):
        mid = (a + b) / 2
        candidate = entropy_balance(z, target, np.exp(mid))
        if candidate[1]["relative_ess"] >= ess:
            best, b = candidate, mid
        else:
            a = mid
    return best


def gumbel_top_k(logits, size, rng):
    """Distinct rows drawn with probability proportional to exp(logits), without underflow.

    Equivalent to successive sampling without replacement (Efraimidis-Spirakis keys).
    """
    if size > len(logits):
        raise ValueError("Requested more cells than the source stage holds")
    keys = logits + rng.gumbel(size=len(logits))
    return np.sort(np.argpartition(-keys, size - 1)[:size])


def balanced_prediction(stage, features, trend, config, n_cells, h, seed=SEED):
    rng = np.random.default_rng(seed)
    z, target = features.design(features.target_direction(config["strength"] * h))
    logits, diagnostics = balance_to_ess(z, target, config["ess"])
    rows = gumbel_top_k(logits, min(n_cells, stage.n_cells), rng)
    prediction = stage.dense(rows)
    if config["bulk"]:
        prediction = apply_trend(prediction, trend, config["bulk"] * h / 5, False, seed)
    weights = np.exp(logits - logits.max())
    return prediction, rows, weights / weights.sum(), diagnostics


def grid(quick):
    strengths, shares, bulks = ([2, 3], [.3], [0, 3]) if quick else ([1, 2, 3, 5], [.5, .3, .15], [0, 3])
    return [{"name": f"s{s}_ess{e}_b{b}", "strength": s, "ess": e, "bulk": b}
            for s in strengths for e in shares for b in bulks]


def type_mass(labels, rows_or_weights, weights=False):
    if labels is None:
        return {}
    if weights:
        return {str(k): float(rows_or_weights[labels == k].sum()) for k in np.unique(labels)}
    values, counts = np.unique(labels[rows_or_weights], return_counts=True)
    return {str(k): float(c / len(rows_or_weights)) for k, c in zip(values, counts)}


# --------------------------------------------------------------------------- commands
def benchmark(args):
    out = RUNS / args.output
    out.mkdir(parents=True, exist_ok=False)
    source, future = open_stage("E9.5"), open_stage("E10.5")
    _, genes = board_spec("T1:val")
    trend, external = load_trend(genes)
    split, split_note = fixed_split(source, future)
    reference = source.dense(split["reference"])
    evaluators = {
        "development": Evaluator(reference, future.dense(split["development"]), future.dense(split["ceiling"])),
        "confirmation": Evaluator(reference, future.dense(split["confirmation"]),
                                  future.dense(split["confirmation_ceiling"])),
    }
    n_cells = len(split["reference"])
    features = CellFeatures(source, trend)
    labels = source.labels
    records, cache = [], {}

    def log(row):
        records.append(row)
        pd.DataFrame(records).to_csv(out / "scores.csv", index=False)
        print(f"{row['split']:>12}  {row['name']:<24} {row['score']:.3f}", flush=True)

    log({"name": "copy_last", "split": "development", **evaluators["development"](reference)})
    if ROUND2_ROWS.exists():  # same split, same metric code: must reproduce the round-2 number
        rows = np.load(ROUND2_ROWS)[ROUND2_BEST]
        old = apply_trend(source.dense(rows), trend, 3 / 5, False, SEED)
        for split_name, evaluator in evaluators.items():
            log({"name": f"round2_{ROUND2_BEST}", "split": split_name, **evaluator(old),
                 "expected_round2_score": ROUND2_EXPECTED[split_name]})
    for config in grid(args.quick):
        prediction, rows, weights, diagnostics = balanced_prediction(source, features, trend, config, n_cells, h=1)
        cache[config["name"]] = (prediction, rows, weights)
        log({**config, "split": "development", **diagnostics, **evaluators["development"](prediction)})
    development = pd.DataFrame([r for r in records if r["split"] == "development" and "strength" in r])
    top = development.sort_values("score", ascending=False).head(3)
    for _, row in top.iterrows():
        prediction = cache[row["name"]][0]
        log({**{k: row[k] for k in ("name", "strength", "ess", "bulk")},
             "split": "confirmation", **evaluators["confirmation"](prediction)})
    best = top.iloc[0]["name"]
    _, rows, weights = cache[best]
    write_json(out / "summary.json", {
        "selected_on_development": best, "split": split_note, "seed": SEED, "external": external,
        "selected_type_mass_of_weights": type_mass(labels, weights, weights=True),
        "selected_type_share_of_sampled_rows": type_mass(labels, rows),
        "source_type_share": type_mass(labels, np.arange(source.n_cells)),
        "adoption_rule": "use for E12.5 only if its confirmation score beats round 2's 59.79 by >= 0.5",
        "note": "E10.5 truth scores the predictions only; weights use E9.5 cells and the bulk trend.",
    })
    np.savez_compressed(out / "selected_rows.npz", rows=rows, weights=weights)
    print(pd.DataFrame(records).to_string(max_cols=12))
    source.close(), future.close()


def export(args):
    config = next(c for c in grid(False) + grid(True) if c["name"] == args.config_name)
    out = Path("predictions") / f"round3_{config['name']}_h{args.h}"
    out.mkdir(parents=True, exist_ok=False)
    _, genes = board_spec("T1:test")
    source = open_stage("E10.5", "T1:test")
    trend, external = load_trend(genes)
    features = CellFeatures(source, trend)
    prediction, rows, weights, diagnostics = balanced_prediction(source, features, trend, config, 2000, h=args.h)
    path = out / "E12.5_prediction.h5ad"
    make_prediction_file(path, prediction, genes)
    library = np.expm1(prediction.astype(np.float64)).sum(axis=1)
    write_json(out / "manifest.json", {
        "config": config, "h": args.h, "seed": SEED, "input_E10_5_sha256": file_hash(source.path),
        "external": external, "balance": diagnostics, "validation": validate_prediction(path, "T1:test"),
        "library_min": float(library.min()), "library_max": float(library.max()),
        "type_share_of_sampled_rows": type_mass(source.labels, rows), "status": "not_submitted",
    })
    print(path)
    source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--output", default="round3_balance_seed42")
    bench.add_argument("--quick", action="store_true")
    exp = sub.add_parser("export")
    exp.add_argument("--config-name", required=True)
    exp.add_argument("--h", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    benchmark(args) if args.mode == "benchmark" else export(args)


if __name__ == "__main__":
    main()
