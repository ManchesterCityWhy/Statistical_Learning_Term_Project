"""Round 5 shared pieces: in-memory stages, library ids, a cached replica of the official metrics, proxies.

Three local proxies, all with reference = the 1,706 E9.5 cells of the round-2 split:
    dm   depth-matched: E9.5/E8.5 thinned to the E10.5 depth distribution, target = E10.5 (all libraries)
    raw  released E9.5/E8.5, target = E10.5 (all libraries)
    h3   released E9.5/E8.5, target = E10.5 library 3 only (the deep, heart-proper library)
The E10.5 release is three 10x libraries (barcode suffix _1/_2/_3). Libraries 1 and 2 are shallow
(median 3.8k / 6.4k UMI) and mostly pharyngeal / non-cardiac tissue; library 3 has the depth of every
E8.5/E9.5 library (10.4k) and a heart-proper composition. `h3` asks: what if the target is a cleanly
dissected, normally sequenced heart, which is what the data page suggests for later stages.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from t1_pipeline.data import board_spec
from t1_pipeline.evaluation import METRICS, calibrated_score

REAL = Path.home() / "statlearn_real"
SEED = 42
CM = {"OFT/RV-CM", "OFT-CM", "RV-CM", "LV-CM", "V-CM", "AVC-CM", "IFT-CM", "SV-CM"}
ENDO = {"Endocardium", "Endocardium-1", "Endocardium-2", "EndoMT Endocardium", "Early VIC", "Endothelium", "BEC"}
EPI = {"Proepicardium", "Epicardium"}
HEME = {"Blood", "Macrophage"}
CARDIAC = CM | ENDO | EPI | HEME
COARSE = {**{k: "CM" for k in CM}, **{k: "Endo" for k in ENDO}, **{k: "Epi" for k in EPI},
          "Blood": "Blood", "Macrophage": "Macrophage"}


def coarse(labels):
    return np.array([COARSE.get(x, "Other") for x in labels])


class Mem:
    """A released stage held in memory as CSR in official gene order (duck-types t1_pipeline.data.Stage)."""

    def __init__(self, path, genes=None):
        genes = genes or board_spec("T1:val")[1]
        data = ad.read_h5ad(path)
        order = data.var_names.astype(str).get_indexer(genes)
        assert (order >= 0).all() and len(order) == data.n_vars
        x = data.X.tocsr() if sparse.issparse(data.X) else sparse.csr_matrix(data.X)
        self.X = x[:, order].astype(np.float32).tocsr()
        self.X.eliminate_zeros()
        self.labels = data.obs["celltype"].astype(str).to_numpy()
        self.lib = np.array([name.rsplit("_", 1)[-1] for name in data.obs_names])
        self.n_cells, self.order, self.path = self.X.shape[0], np.arange(len(genes)), Path(path)

    def dense(self, rows):
        return np.asarray(self.X[np.asarray(rows, dtype=int)].toarray(), dtype=np.float32)

    def mean(self, rows=None):
        x = self.X if rows is None else self.X[np.asarray(rows, dtype=int)]
        return np.asarray(x.mean(axis=0), dtype=np.float64).ravel()

    def close(self):
        pass


def pb(x):
    return np.asarray(x, dtype=np.float64).mean(axis=0)


class FastEval:
    """Official DES / DCS / MMD / CSS with everything that depends only on (reference, target) cached.

    Same random draws as common.core_metrics called through t1_pipeline.evaluation.metric_panel (seed 42);
    `check()` compares against the official functions.
    """

    def __init__(self, reference, target, ceiling_half, seed=SEED):
        from common.core_metrics import _ranks, _signed_overlap, de_genes, pseudobulk
        from sklearn.decomposition import PCA
        self.seed, self.reference, self.target = seed, reference, target
        self._overlap, self._ranks, self._pb = _signed_overlap, _ranks, pseudobulk
        self.up, self.dn, _ = de_genes(target, reference)
        self.pb_ref = pseudobulk(reference)
        self.dt = pseudobulk(target) - self.pb_ref
        self.chance = max(_signed_overlap(self.pb_ref, self.up, self.dn)[0],
                          _signed_overlap(-self.pb_ref, self.up, self.dn)[0])
        self.rt, self.rr = _ranks(self.dt), _ranks(self.pb_ref)
        self.pca = PCA(n_components=30, random_state=0).fit(target)
        self.latent = self.pca.transform(target)
        self._vario = {}
        self.floor = self.panel(reference)
        self.ceiling = self.panel(ceiling_half)

    def direction(self, dp):
        """DES and DCS of a predicted pseudobulk change vector."""
        if np.std(dp) < 1e-2 * np.std(self.dt):
            return 0.0, 0.0
        raw, _ = self._overlap(dp, self.up, self.dn)
        des = (raw - self.chance) / (1 - self.chance)
        c = np.corrcoef(np.vstack([self._ranks(dp), self.rt, self.rr]))
        if 1.0 - abs(c[0, 2]) < 1e-6:
            return float(des), 0.0
        dcs = (c[0, 1] - c[0, 2] * c[1, 2]) / np.sqrt(max((1 - c[0, 2] ** 2) * (1 - c[1, 2] ** 2), 1e-12))
        return float(des), float(np.clip(dcs, -1, 1))

    def mmd(self, pred, n=2000, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
        from sklearn.metrics.pairwise import rbf_kernel
        rng = np.random.default_rng(self.seed)
        a = self.pca.transform(pred[rng.choice(pred.shape[0], min(n, pred.shape[0]), replace=False)])
        b = self.latent[rng.choice(self.target.shape[0], min(n, self.target.shape[0]), replace=False)]
        d2 = np.sum((b[:, None] - b[None, :]) ** 2, -1)
        gamma0 = 1.0 / (np.median(d2[d2 > 0]) + 1e-9)
        na, nb, total = a.shape[0], b.shape[0], 0.0
        for s in scales:
            g = gamma0 * s
            kaa, kbb, kab = rbf_kernel(a, a, g), rbf_kernel(b, b, g), rbf_kernel(a, b, g)
            np.fill_diagonal(kaa, 0.0), np.fill_diagonal(kbb, 0.0)
            total += kaa.sum() / (na * (na - 1)) + kbb.sum() / (nb * (nb - 1)) - 2 * kab.mean()
        return float(total / len(scales))

    def variogram(self, pred, n_pairs=20000, n_cells=1500, p=0.5):
        rng = np.random.default_rng(self.seed)
        rows_a = rng.choice(pred.shape[0], min(n_cells, pred.shape[0]), replace=False)
        key = pred.shape[0]
        if key not in self._vario:
            b = self.target[rng.choice(self.target.shape[0], min(n_cells, self.target.shape[0]), replace=False)]
            i, j = rng.integers(0, b.shape[1], n_pairs), rng.integers(0, b.shape[1], n_pairs)
            keep = i != j
            i, j = i[keep], j[keep]
            self._vario[key] = (i, j, (np.abs(b[:, i] - b[:, j]) ** p).mean(0))
        i, j, vb = self._vario[key]
        a = pred[rows_a]
        va = (np.abs(a[:, i] - a[:, j]) ** p).mean(0)
        return float(((va - vb) ** 2).mean())

    def panel(self, pred):
        des, dcs = self.direction(self._pb(pred) - self.pb_ref)
        return {"de_score": des, "de_direction": dcs, "mmd_u": self.mmd(pred), "variogram": self.variogram(pred)}

    def __call__(self, pred):
        panel = self.panel(pred)
        scored = calibrated_score(panel, self.floor, self.ceiling)
        return {**panel, "score": scored["local_score"], **{f"skill_{k}": v for k, v in scored["skills"].items()}}

    def check(self, pred):
        from t1_pipeline.evaluation import metric_panel
        official, fast = metric_panel(pred, self.target, self.reference), self.panel(pred)
        return {k: (official[k], fast[k]) for k in METRICS}


def split_rows(source, future):
    """Round-2 split (reference from E9.5; development/confirmation and their ceiling halves from E10.5)."""
    saved = np.load(REAL / "runs" / "expanded_seed42" / "indices.npz")
    split = {k: saved[k] for k in ("reference", "development", "ceiling", "confirmation", "confirmation_ceiling")}
    assert split["reference"].max() < source.n_cells and split["development"].max() < future.n_cells
    return split


def h3_rows(future, n=1249, seed=SEED):
    """Development / confirmation targets drawn from E10.5 library 3 only."""
    lib3 = np.flatnonzero(future.lib == "3")
    order = np.random.default_rng(seed).permutation(lib3)
    return {"development": order[:n], "ceiling": order[n:2 * n],
            "confirmation": order[2 * n:3 * n], "confirmation_ceiling": order[3 * n:4 * n]}


def renormalize(x):
    out = np.empty_like(x, dtype=np.float32)
    for s in range(0, len(x), 256):
        c = np.expm1(x[s:s + 256].astype(np.float64))
        tot = c.sum(axis=1, keepdims=True)
        out[s:s + 256] = np.log1p(c * np.divide(1e4, tot, out=np.ones_like(tot), where=tot > 0))
    return out
