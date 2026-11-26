"""Round 6 shared pieces: lineages, label transfer, the E14.5 target split, thinning helpers, result logging.

Everything here is computed from the inputs a prediction is allowed to see (the source stage, earlier stages,
the external anchor); nothing reads a proxy target.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from experiments.round5_common import REAL, SEED, FastEval, Mem, pb
from experiments.round5_exp4 import depth_of, thin_rows

CM = {"OFT/RV-CM", "OFT-CM", "RV-CM", "LV-CM", "V-CM", "AVC-CM", "IFT-CM", "SV-CM"}
ENDO = {"Endocardium", "Endocardium-1", "Endocardium-2", "Endothelium", "BEC"}
CUSHION = {"EndoMT Endocardium", "Early VIC"}
EPI = {"Proepicardium", "Epicardium", "JCF"}
HEART = CM | ENDO | CUSHION | EPI | {"Blood", "Macrophage"}
LINEAGE = {**{k: "CM" for k in CM}, **{k: "Endo" for k in ENDO}, **{k: "Cushion" for k in CUSHION},
           **{k: "Epi" for k in EPI}, "Blood": "Blood", "Macrophage": "Mac"}
LINEAGES = ("CM", "Endo", "Cushion", "Epi", "Blood", "Mac", "Other")


def lineage_of(labels):
    return np.array([LINEAGE.get(x, "Other") for x in labels])


def centroid_transfer(ref_x, labels, query, n_top=60):
    """Nearest reference cell type of every query cell (Pearson correlation on the union of type markers)."""
    names = np.unique(labels)
    cent = np.stack([np.asarray(ref_x[np.flatnonzero(labels == g)].mean(axis=0)).ravel() for g in names])
    overall = np.asarray(ref_x.mean(axis=0)).ravel()
    genes = np.unique(np.concatenate([np.argsort(-(c - overall))[:n_top] for c in cent]))
    c = cent[:, genes]
    c = (c - c.mean(axis=1, keepdims=True)) / (c.std(axis=1, keepdims=True) + 1e-9)
    out = np.empty(query.shape[0], dtype=object)
    for s in range(0, query.shape[0], 2048):
        q = query[s:s + 2048][:, genes].toarray()
        q = (q - q.mean(axis=1, keepdims=True)) / (q.std(axis=1, keepdims=True) + 1e-9)
        out[s:s + 2048] = names[(q @ c.T).argmax(axis=1)]
    return out.astype(str)


def thinned(stage, target_depths, seed=SEED):
    """A copy of a stage whose cells are binomially thinned to quantile-matched target depths."""
    out = copy.copy(stage)
    out.X = thin_rows(stage.X, target_depths, np.random.default_rng(seed))
    return out


def anchor_split(n, size=1000, seed=SEED):
    """Disjoint rows of the E14.5 sample: two target halves + their ceiling halves, the rest is a mixing pool."""
    order = np.random.default_rng(seed).permutation(n)
    keys = ("development", "ceiling", "confirmation", "confirmation_ceiling")
    out = {k: np.sort(order[i * size:(i + 1) * size]) for i, k in enumerate(keys)}
    out["pool"] = np.sort(order[4 * size:])
    return out


def take(stage, rows, n, rng, weights=None):
    """n distinct rows of `rows` (optionally weighted)."""
    rows = np.asarray(rows)
    p = None if weights is None else np.asarray(weights, float) / np.sum(weights)
    return np.sort(rows[rng.choice(len(rows), n, replace=False, p=p)])


class Log:
    def __init__(self, path):
        self.path, self.records = path, []
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, ev, cells, **meta):
        r = {**meta, **ev(cells), "cells": int(len(cells)), "nonzero": float((cells > 0).mean())}
        self.records.append(r)
        pd.DataFrame(self.records).to_csv(self.path, index=False)
        tag = " ".join(f"{k}={v}" for k, v in meta.items())
        print(f"{tag:<70} {r['score']:6.2f}  DES {r['skill_de_score']:.3f} DCS {r['skill_de_direction']:.3f} "
              f"MMD {r['skill_mmd_u']:.3f} CSS {r['skill_variogram']:.3f}", flush=True)
        return r


__all__ = ["REAL", "SEED", "FastEval", "Mem", "pb", "depth_of", "thin_rows", "CM", "ENDO", "CUSHION", "EPI", "HEART",
           "LINEAGE", "LINEAGES", "lineage_of", "centroid_transfer", "thinned", "anchor_split", "take", "Log"]
