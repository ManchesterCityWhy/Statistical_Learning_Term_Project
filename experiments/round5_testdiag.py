"""Round 5 diagnostics on the test-time inputs (no truth involved): what do the candidate change vectors look like?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round5_common import REAL, SEED, Mem, coarse, pb
from t1_pipeline.data import board_spec


def deconv_weights(stage, trend, strength, rows=None):
    rows = np.arange(stage.n_cells) if rows is None else rows
    labels = stage.labels[rows]
    types = np.unique(labels)
    profiles = np.stack([np.asarray(stage.X[rows[labels == t]].expm1().mean(axis=0)).ravel() for t in types], axis=1)
    p = np.array([(labels == t).mean() for t in types])
    base = profiles @ p
    keep = (base > .1) & (np.std(profiles, axis=1) > .1)
    design = profiles[keep] / np.sqrt(base[keep, None] + 1)
    desired = base * np.exp(np.clip(trend * strength, -2, 2))
    desired *= base.sum() / desired.sum()
    q = nnls(np.vstack([design, 10 * np.eye(len(p))]), np.r_[desired[keep] / np.sqrt(base[keep] + 1), 10 * p])[0]
    return pd.DataFrame({"source": p, "deconv": q / q.sum()}, index=types)


def main():
    pd.set_option("display.width", 250)
    _, genes = board_spec("T1:test")
    trend, _ = load_trend(genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    e9t = Mem(REAL / "data/proxy/E9.5_RNA.h5ad", genes)
    lib3 = np.flatnonzero(e10.lib == "3")
    for s in (3, 6):
        w = deconv_weights(e10, trend, s)
        w3 = deconv_weights(e10, trend, s, lib3)
        print(f"deconv strength {s}: all cells / library 3 only")
        print(pd.concat([w, w3.add_suffix("_lib3")], axis=1).round(3).sort_values("deconv", ascending=False).to_string())
    rng = np.random.default_rng(SEED)
    ref = e10.dense(np.sort(rng.choice(e10.n_cells, 2498, replace=False)))
    ref_pb = pb(ref)
    m10, m3, m9, m9t = e10.mean(), e10.mean(lib3), e9.mean(), e9t.mean()
    vec = {"lib3-all": m3 - m10, "bulk2/5": pb(apply_trend(ref.copy(), trend, 2 / 5, False, SEED)) - ref_pb,
           "E10all-E9raw": m10 - m9, "E10all-E9thin": m10 - m9t, "E10lib3-E9raw": m3 - m9,
           "trend": trend.astype(float), "level": m10}
    for lab in ("CM", "Endo", "Epi", "Blood", "Other"):
        rows = np.flatnonzero(coarse(e10.labels) == lab)
        vec[f"type:{lab}-all"] = e10.mean(rows) - m10
    print(pd.DataFrame({k: {"norm": np.linalg.norm(v), "n|d|>=.25": int((np.abs(v) >= .25).sum()),
                            "up": int((v >= .25).sum()), "down": int((v <= -.25).sum())} for k, v in vec.items()}).T)
    names = list(vec)
    rho = spearmanr(np.column_stack([vec[k] for k in names])).statistic
    print(pd.DataFrame(rho, index=names, columns=names).round(2).to_string())
    expressed = m10 > .05
    rho = spearmanr(np.column_stack([vec[k][expressed] for k in names])).statistic
    print("expressed genes only", int(expressed.sum()))
    print(pd.DataFrame(rho, index=names, columns=names).round(2).to_string())
    top = lambda v, n=25: ", ".join(np.array(genes)[np.argsort(-v)[:n]])
    for k in ("lib3-all", "E10lib3-E9raw", "bulk2/5"):
        print(k, "UP:", top(vec[k]), "\n   DOWN:", top(-vec[k]))


if __name__ == "__main__":
    main()
