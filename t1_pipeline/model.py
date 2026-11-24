"""Train-only extrapolation, preserving sampled cell-level residuals."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA

from .data import Stage
from .distribution import DistributionEvolution


def extrapolate(cells, delta, *, alpha, h, preserve_zeros=False):
    if not 0 <= alpha <= 1 or h < 0:
        raise ValueError("Require alpha in [0, 1] and non-negative time multiplier")
    cells = np.asarray(cells, dtype=np.float32)
    offset = np.asarray(delta, dtype=np.float32) * np.float32(alpha * h)
    result = cells + (offset * (cells > 0) if preserve_zeros else offset)
    negative_fraction = float(np.mean(result < 0))
    return np.maximum(result, 0).astype(np.float32), negative_fraction


@dataclass
class Extrapolator:
    earlier: Stage
    last: Stage
    pca_cells_per_stage: int = 2500
    pca_seed: int = 42
    delta: np.ndarray = field(init=False)
    projections: dict = field(default_factory=dict, init=False)
    diagnostics: dict = field(default_factory=dict, init=False)
    type_deltas: dict | None = field(default=None, init=False)
    evolution: DistributionEvolution | None = field(default=None, init=False)

    def __post_init__(self):
        self.delta = self.last.mean() - self.earlier.mean()

    def fit_pca(self, dimensions):
        dimensions = sorted(set(dimensions))
        if not dimensions:
            return
        rng = np.random.default_rng(self.pca_seed)
        samples = [stage.dense(rng.choice(stage.n_cells, min(self.pca_cells_per_stage, stage.n_cells), replace=False))
                   for stage in (self.earlier, self.last)]
        x = np.vstack(samples)
        if dimensions[0] < 1 or dimensions[-1] >= min(x.shape):
            raise ValueError("PCA dimensions must be positive and smaller than the fitting matrix dimensions")
        pca = PCA(n_components=dimensions[-1], svd_solver="randomized", random_state=self.pca_seed)
        pca.fit(x)
        denominator = float(self.delta @ self.delta)
        for d in dimensions:
            basis = pca.components_[:d].astype(np.float64)
            projection = (self.delta @ basis.T) @ basis
            self.projections[d] = projection
            self.diagnostics[str(d)] = {
                "explained_variance_ratio": float(pca.explained_variance_ratio_[:d].sum()),
                "retained_delta_squared_norm": float(projection @ projection / denominator) if denominator else 0.0,
            }

    def fit_types(self):
        a, b = self.earlier.labels, self.last.labels
        if a is None or b is None:
            raise ValueError("pseudobulk_shift requires observed training celltype annotations")
        self.type_deltas = {}
        for label in sorted(set(a) & set(b)):
            self.type_deltas[label] = self.last.mean(np.flatnonzero(b == label)) - self.earlier.mean(np.flatnonzero(a == label))

    def predict(self, candidate, n_cells, seed, h):
        if n_cells < 1:
            raise ValueError("Prediction requires at least one cell")
        if n_cells > self.last.n_cells:
            raise ValueError("Prediction cell count exceeds available source cells; choose a smaller sample")
        rng = np.random.default_rng(seed)
        method = candidate["method"]
        weight = candidate.get("state_weight", 0.0)
        if weight or method == "hurdle":
            if self.evolution is None:
                self.evolution = DistributionEvolution(self.earlier, self.last, self.pca_seed)
            indices = self.evolution.sample(n_cells, rng, weight, h)
        else:
            indices = rng.choice(self.last.n_cells, n_cells, replace=False)
        cells = self.last.dense(indices)
        if method == "copy_last":
            pred, clipped = cells.copy(), 0.0
        elif method == "hurdle":
            pred = self.evolution.shift(cells, indices, rng, candidate, h)
            clipped = 0.0
        elif method in {"full_shift", "pca_shift"}:
            delta = self.delta if method == "full_shift" else self.projections[candidate["dimensions"]]
            pred, clipped = extrapolate(cells, delta, alpha=candidate["alpha"], h=h,
                                        preserve_zeros=candidate.get("preserve_zeros", False))
        elif method == "pseudobulk_shift":
            if self.type_deltas is None:
                self.fit_types()
            labels = self.last.labels[indices]
            pred = cells.copy()
            negative = 0.0
            for label, delta in self.type_deltas.items():
                mask = labels == label
                if mask.any():
                    pred[mask], fraction = extrapolate(cells[mask], delta, alpha=candidate.get("alpha", 1.0),
                                                       h=h, preserve_zeros=candidate.get("preserve_zeros", False))
                    negative += fraction * int(mask.sum())
            clipped = negative / n_cells
        else:
            raise ValueError(f"Unknown method: {method}")
        if candidate.get("renormalize", False):
            for start in range(0, len(pred), 256):
                counts = np.expm1(pred[start:start + 256])
                sums = counts.sum(axis=1, keepdims=True, dtype=np.float64)
                factors = np.divide(10000.0, sums, out=np.ones_like(sums), where=sums > 0)
                pred[start:start + 256] = np.log1p(counts * factors)
        diagnostics = {"negative_fraction_before_clipping": clipped,
                       "source_nonzero_fraction": float(np.mean(cells > 0)),
                       "prediction_nonzero_fraction": float(np.mean(pred > 0)),
                       "detected_genes_quantiles": np.quantile((pred > 0).sum(axis=1), [0, .25, .5, .75, 1]).tolist()}
        return pred, diagnostics
