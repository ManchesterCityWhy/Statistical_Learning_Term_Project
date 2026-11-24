"""Extrapolate cell-state frequencies and sparse expression distributions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


@dataclass
class Moments:
    n: int
    detections: np.ndarray
    sums: np.ndarray

    @classmethod
    def of(cls, matrix):
        return cls(
            matrix.shape[0],
            np.asarray((matrix > 0).sum(axis=0)).ravel(),
            np.asarray(matrix.sum(axis=0, dtype=np.float64)).ravel(),
        )

    def smooth(self, prior_p=0.0, prior_mu=0.0, count=0):
        detections = self.detections + count * prior_p
        p = detections / (self.n + count)
        sums = self.sums + count * prior_p * prior_mu
        mu = np.divide(sums, detections, out=np.zeros_like(sums), where=detections > 0)
        return p.astype(np.float32), mu.astype(np.float32)


class DistributionEvolution:
    """Fit the changes between two observed distributions, using cells from both stages."""

    def __init__(self, earlier, last, seed):
        self.earlier, self.last, self.seed = earlier, last, seed
        self.global_moments = None
        self.type_moments = {}
        self.state_odds = None

    def fit_expression(self):
        # Sampling is capped independently of the evaluation population.
        rng = np.random.default_rng(self.seed)
        stages = (self.earlier, self.last)
        indices = [rng.choice(stage.n_cells, min(6500, stage.n_cells), replace=False) for stage in stages]
        samples = [sparse.csr_matrix(stage.matrix(rows)) for stage, rows in zip(stages, indices)]
        self.global_moments = tuple(Moments.of(sample).smooth() for sample in samples)
        if any(stage.labels is None for stage in stages):
            return
        labels = [stage.labels[rows] for stage, rows in zip(stages, indices)]
        for label in sorted(set(labels[0]) & set(labels[1])):
            masks = [values == label for values in labels]
            if min(mask.sum() for mask in masks) < 35:
                continue
            self.type_moments[label] = tuple(
                Moments.of(sample[mask]).smooth(prior_p, prior_mu, count=80)
                for sample, mask, (prior_p, prior_mu) in zip(samples, masks, self.global_moments)
            )

    def fit_state_weights(self):
        # Identify a broad temporal direction from variable genes. Fit the classifier
        # on equally sized samples, so the intercept does not reflect stage size.
        rng = np.random.default_rng(self.seed)
        count = min(3000, self.earlier.n_cells, self.last.n_cells)
        stages = (self.earlier, self.last)
        matrices = [sparse.csr_matrix(stage.matrix(rng.choice(stage.n_cells, count, replace=False)))
                    for stage in stages]
        pooled = sparse.vstack(matrices, format="csr")
        mean = np.asarray(pooled.mean(axis=0)).ravel()
        var = np.asarray(pooled.power(2).mean(axis=0)).ravel() - mean**2
        columns = np.argsort(var)[-min(2048, len(var)):]
        dimensions = min(32, pooled.shape[0] - 1, len(columns) - 1)
        pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=self.seed)
        latent = pca.fit_transform(pooled[:, columns].toarray().astype(np.float32))
        scale = np.maximum(np.sqrt(pca.explained_variance_), 1e-3)
        classifier = LogisticRegression(C=.1, max_iter=250)
        classifier.fit(latent / scale, np.repeat([0, 1], count))
        odds = []
        for start in range(0, self.last.n_cells, 1024):
            rows = np.arange(start, min(start + 1024, self.last.n_cells))
            selected = self.last.matrix(rows)[:, columns]
            dense = selected.toarray() if sparse.issparse(selected) else selected
            odds.append(classifier.decision_function(pca.transform(dense.astype(np.float32)) / scale))
        values = np.concatenate(odds)
        self.state_odds = np.clip(values - np.median(values), -2.5, 2.5)

    def sample(self, count, rng, strength, h):
        if strength == 0:
            return rng.choice(self.last.n_cells, count, replace=False)
        if self.state_odds is None:
            self.fit_state_weights()
        # Exponential-race sampling keeps distinct parent cells and damps
        # the influence of states with nearly perfect stage separation.
        log_weights = np.clip(strength * h * self.state_odds, -1.6, 1.6)
        race = rng.exponential(size=self.last.n_cells) * np.exp(-log_weights)
        return np.argpartition(race, count - 1)[:count]

    def shift(self, cells, indices, rng, candidate, h):
        if self.global_moments is None:
            self.fit_expression()
        labels = self.last.labels
        labels = None if labels is None else labels[indices]
        local_weight = float(candidate.get("local", 0.0))
        strength = float(candidate.get("alpha", 0.5)) * h
        gain = float(candidate.get("gain", 1.0))
        out = cells.copy()
        if local_weight and labels is not None:
            for label in np.unique(labels):
                rows = np.flatnonzero(labels == label)
                _shift_rows(out, rows, self.global_moments, self.type_moments.get(label),
                            local_weight, strength, gain, rng)
        else:
            _shift_rows(out, np.arange(len(out)), self.global_moments, None,
                        0.0, strength, gain, rng)
        return out


def _shift_rows(out, rows, global_moments, group_moments, mixing, strength, gain, rng):
    (p0, mu0), (p1, mu1) = global_moments
    delta_p, delta_mu = p1 - p0, mu1 - mu0
    if group_moments is not None:
        (gp0, gm0), (gp1, gm1) = group_moments
        delta_p = (1 - mixing) * delta_p + mixing * (gp1 - gp0)
        delta_mu = (1 - mixing) * delta_mu + mixing * (gm1 - gm0)
        p1, mu1 = gp1, gm1

    # Low-support positive-expression estimates shrink towards no change.
    supported_delta = np.clip(delta_mu, -1.0, 1.0) * np.minimum(1.0, p1 / .05)
    target_p = np.clip(p1 + strength * delta_p, 0.0, 1.0)
    loss = np.clip((p1 - target_p) / np.maximum(p1, 1e-5), 0, .7)
    activation = gain * np.clip((target_p - p1) / np.maximum(1.0 - p1, 1e-5), 0, .5)
    activation = np.where(p1 >= .005, activation, 0)
    # The activation values are in the same observed log-expression scale.
    intensity = np.clip(mu1 + strength * supported_delta, .02, None)
    for start in range(0, len(rows), 128):
        selected = rows[start:start + 128]
        batch = out[selected]
        present = batch > 0
        draw = rng.random(batch.shape, dtype=np.float32)
        batch[present & (draw < loss)] = 0.0
        retained = present & (batch > 0)
        batch = np.where(retained, np.maximum(batch + strength * supported_delta, 0.0), batch)
        batch = np.where(~present & (draw < activation), intensity, batch)
        out[selected] = batch
