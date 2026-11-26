"""Round 4: estimate *which way each gene moves* better, then move the population that way.

The oracle analysis showed that knowing the true pseudobulk change alone is worth ~74 local points,
while the round-2 model reaches DCS 0.41 against 0.76 for the oracle. This script treats the gene-wise
change as a regression target across genes:

    Delta_true(E9.5 -> E10.5)_g  ~  w1 * bulk_g + w2 * internal_g + w3 * deconv_g + w4 * level_g

where every feature is computable without E10.5 (bulk trend applied to the source cells, the observed
change between the two previous stages, the change implied by the round-2 deconvolution, and the
source expression level). Weights are fitted on the development split only and judged on the
confirmation split. The fitted change is then imposed on the round-2 cells by an additive shift on
non-zero entries, scaled by each gene's detection rate so the pseudobulk moves by the intended amount.

    python -m experiments.round4_direction benchmark
    python -m experiments.round4_direction export --name <best> [--internal-scale 0.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round3_common import RUNS, Evaluator, fixed_split, open_stage
from t1_pipeline.data import board_spec, file_hash, validate_prediction, write_json

SEED = 42
ROUND2_ROWS = RUNS / "composition_refined_seed42" / "sampled_rows.npz"


def pb(x):
    return np.asarray(x, dtype=np.float64).mean(axis=0)


class DirectionScorer:
    """Vector-level DES and DCS, identical to veckit's de_score/de_direction given a pseudobulk change."""

    def __init__(self, target, reference):
        from common.core_metrics import de_genes, _signed_overlap
        self.overlap = _signed_overlap
        self.up, self.dn, _ = de_genes(target, reference)
        self.ref = pb(reference)
        self.true = pb(target) - self.ref
        c_up, _ = _signed_overlap(self.ref, self.up, self.dn)
        c_dn, _ = _signed_overlap(-self.ref, self.up, self.dn)
        self.chance = max(c_up, c_dn)
        self.rt, self.rr = rankdata(self.true), rankdata(self.ref)

    def __call__(self, delta):
        raw, _ = self.overlap(delta, self.up, self.dn)
        des = (raw - self.chance) / (1 - self.chance)
        c = np.corrcoef(np.vstack([rankdata(delta), self.rt, self.rr]))
        dcs = (c[0, 1] - c[0, 2] * c[1, 2]) / np.sqrt((1 - c[0, 2] ** 2) * (1 - c[1, 2] ** 2))
        return {"des": float(des), "dcs": float(dcs)}


def deconv_rows(stage, trend, strength, n_cells, rng):
    """Round-2 composition deconvolution (same code path as composition_search.py)."""
    from scipy.optimize import nnls
    labels = stage.labels
    types = np.unique(labels)
    means, p = [], []
    for lab in types:
        ids = np.flatnonzero(labels == lab)
        p.append(len(ids) / stage.n_cells)
        total = np.zeros(len(stage.order))
        for chunk in np.array_split(ids, max(1, len(ids) // 128)):
            total += np.expm1(stage.dense(chunk)).sum(axis=0, dtype=np.float64)
        means.append(total / len(ids))
    profiles, p = np.stack(means, axis=1), np.array(p)
    base = profiles @ p
    keep = (base > .1) & (np.std(profiles, axis=1) > .1)
    design = profiles[keep] / np.sqrt(base[keep, None] + 1)
    desired = base * np.exp(np.clip(trend * strength, -2, 2))
    desired *= base.sum() / desired.sum()
    q = nnls(np.vstack([design, 10 * np.eye(len(p))]), np.r_[desired[keep] / np.sqrt(base[keep] + 1), 10 * p])[0]
    q /= q.sum()
    weights = np.zeros(stage.n_cells)
    for lab, prob in zip(types, q):
        weights[labels == lab] = prob / (labels == lab).sum()
    return rng.choice(stage.n_cells, n_cells, replace=False, p=weights / weights.sum())


def impose(cells, target_level, strength=1.0, floor=.05):
    """Shift non-zero entries so the pseudobulk moves towards target_level (keep zeros, clip at 0)."""
    current = pb(cells)
    detect = np.maximum((cells > 0).mean(axis=0), floor)
    shift = strength * (target_level - current) / detect
    return np.maximum(cells + shift * (cells > 0), 0).astype(np.float32)


def fit_weights(features, target, ridge=1e-3):
    """Least squares across genes, features standardised; returns weights on the raw features."""
    X = np.column_stack(features)
    scale = X.std(axis=0)
    Z = X / scale
    w = np.linalg.solve(Z.T @ Z + ridge * len(Z) * np.eye(Z.shape[1]), Z.T @ target)
    return w / scale


def benchmark(args):
    out = RUNS / args.output
    out.mkdir(parents=True, exist_ok=False)
    early, source, future = open_stage("E8.5"), open_stage("E9.5"), open_stage("E10.5")
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    split, note = fixed_split(source, future)
    ref = source.dense(split["reference"])
    dev_t, conf_t = future.dense(split["development"]), future.dense(split["confirmation"])
    scorers = {"development": DirectionScorer(dev_t, ref), "confirmation": DirectionScorer(conf_t, ref)}
    evaluators = {"development": Evaluator(ref, dev_t, future.dense(split["ceiling"])),
                  "confirmation": Evaluator(ref, conf_t, future.dense(split["confirmation_ceiling"]))}
    ref_pb = pb(ref)

    # Round-2 cells (reproduced) as the starting population.
    rows = np.load(ROUND2_ROWS)["deconv3_mix1_bulk3"] if ROUND2_ROWS.exists() else \
        deconv_rows(source, trend, 3, len(ref), np.random.default_rng(SEED))
    base_cells = apply_trend(source.dense(rows), trend, 3 / 5, False, SEED)

    feats = {
        "bulk": pb(apply_trend(ref.copy(), trend, 3 / 5, False, SEED)) - ref_pb,
        "internal": source.mean() - early.mean(),
        "deconv": pb(base_cells) - ref_pb,
        "level": ref_pb,
    }
    records = []
    for name, vec in feats.items():
        for split_name, scorer in scorers.items():
            records.append({"candidate": f"feature_{name}", "split": split_name, **scorer(vec)})
    combos = {"bulk+deconv": ["bulk", "deconv"], "bulk+internal": ["bulk", "internal"],
              "bulk+deconv+internal": ["bulk", "deconv", "internal"],
              "all4": ["bulk", "deconv", "internal", "level"]}
    fitted = {}
    for name, keys in combos.items():
        w = fit_weights([feats[k] for k in keys], scorers["development"].true)
        vec = np.column_stack([feats[k] for k in keys]) @ w
        fitted[name] = (keys, w, vec)
        for split_name, scorer in scorers.items():
            records.append({"candidate": f"fit_{name}", "split": split_name, **scorer(vec),
                            "weights": json.dumps(dict(zip(keys, map(float, w))))})
    frame = pd.DataFrame(records)
    frame.to_csv(out / "direction_scores.csv", index=False)
    print(frame.to_string(), flush=True)

    # Full predictions: impose the best fitted directions on the round-2 cells.
    full = []
    log = lambda r: (full.append(r), pd.DataFrame(full).to_csv(out / "full_scores.csv", index=False),
                     print(f"{r['split']:>12} {r['candidate']:<34} {r['score']:.3f}", flush=True))
    for split_name, ev in evaluators.items():
        log({"candidate": "round2_base", "split": split_name, **ev(base_cells)})
    for name in combos:
        _, _, vec = fitted[name]
        for strength in (.5, 1.0):
            pred = impose(base_cells, ref_pb + vec, strength)
            for split_name, ev in evaluators.items():
                log({"candidate": f"{name}_s{strength}", "split": split_name, **ev(pred),
                     "nonzero_fraction": float((pred > 0).mean())})
    write_json(out / "summary.json", {"split": note, "seed": SEED,
                                      "weights": {k: dict(zip(v[0], map(float, v[1]))) for k, v in fitted.items()},
                                      "note": "weights fitted on development genes only; confirmation is held out"})
    for s in (early, source, future):
        s.close()


def export(args):
    """E12.5 from E10.5 with weights fitted on the (depth-matched) validation proxy.

    Features are rebuilt at test time exactly as in validation: bulk trend and deconvolution at
    multiplier h, the observed change from the depth-matched E9.5 to E10.5, and the E10.5 level.
    """
    from t1_pipeline.cli import make_prediction_file
    summary = json.loads(Path(args.weights).read_text())
    weights = summary["weights"][args.combo]
    out = Path("predictions") / f"round4_{args.combo}_s{args.strength}_h{args.h}"
    out.mkdir(parents=True, exist_ok=False)
    _, genes = board_spec("T1:test")
    from t1_pipeline.data import Stage
    source = Stage.open(Path("data/T1/E10.5_RNA.h5ad"), genes)
    past = Stage.open(Path(args.past), genes)
    trend, external = load_trend(genes)
    rng = np.random.default_rng(SEED)
    ref = source.dense(np.sort(rng.choice(source.n_cells, 2000, replace=False)))
    ref_pb = pb(ref)
    rows = deconv_rows(source, trend, 3 * args.h, 2000, np.random.default_rng(SEED))
    base = apply_trend(source.dense(rows), trend, 3 * args.h / 5, False, SEED)
    feats = {"bulk": pb(apply_trend(ref.copy(), trend, 3 * args.h / 5, False, SEED)) - ref_pb,
             "internal": (source.mean() - past.mean()) * args.h,
             "deconv": pb(base) - ref_pb, "level": ref_pb}
    delta = sum(w * feats[k] for k, w in weights.items())
    pred = impose(base, ref_pb + delta, args.strength)
    path = out / "E12.5_prediction.h5ad"
    make_prediction_file(path, pred, genes)
    library = np.expm1(pred.astype(np.float64)).sum(axis=1)
    write_json(out / "manifest.json", {
        "combo": args.combo, "weights": weights, "strength": args.strength, "h": args.h, "seed": SEED,
        "weights_from": args.weights, "past_stage_file": args.past, "input_E10_5_sha256": file_hash(source.path),
        "external": external, "validation": validate_prediction(path, "T1:test"),
        "nonzero_fraction": float((pred > 0).mean()), "library_min": float(library.min()),
        "library_max": float(library.max()), "status": "not_submitted"})
    print(path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("--output", default="round4_direction_seed42")
    e = sub.add_parser("export")
    e.add_argument("--weights", required=True, help="summary.json of a benchmark run")
    e.add_argument("--combo", default="all4")
    e.add_argument("--strength", type=float, default=1.0)
    e.add_argument("--h", type=int, choices=(1, 2), default=1)
    e.add_argument("--past", default="data/proxy/E9.5_RNA.h5ad", help="E9.5 thinned to the E10.5 depth")
    args = parser.parse_args()
    benchmark(args) if args.mode == "benchmark" else export(args)


if __name__ == "__main__":
    main()
