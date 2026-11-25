"""Round 5: does the within-lineage change of one day persist into the next day? (depth-clean: E8.5, E9.5, E10.5 library 3)"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.bulk_trend import load_trend
from experiments.round5_common import REAL, Mem, coarse
from t1_pipeline.data import board_spec


def overlap(pred, truth, thr=.25):
    up, dn = np.flatnonzero(truth >= thr), np.flatnonzero(truth <= -thr)
    order = np.argsort(-pred)
    hit = len(np.intersect1d(order[:len(up)], up)) + len(np.intersect1d(order[len(order) - len(dn):], dn))
    return hit / max(len(up) + len(dn), 1), len(up), len(dn)


def main():
    pd.set_option("display.width", 250)
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    e8, e9, e10 = (Mem(REAL / f"data/T1/{s}_RNA.h5ad", genes) for s in ("E8.5", "E9.5", "E10.5"))
    c8, c9, c10 = coarse(e8.labels), coarse(e9.labels), coarse(e10.labels)
    c8[e8.labels == "JCF"] = "Epi"
    lib3 = e10.lib == "3"
    rows = []
    groups = [("ALL", np.ones(e8.n_cells, bool), np.ones(e9.n_cells, bool), lib3)]
    groups += [(k, c8 == k, c9 == k, (c10 == k) & lib3) for k in ("CM", "Endo", "Epi", "Blood", "Other")]
    for name, a, b, c in groups:
        m8, m9, m10 = e8.mean(np.flatnonzero(a)), e9.mean(np.flatnonzero(b)), e10.mean(np.flatnonzero(c))
        d1, d2 = m9 - m8, m10 - m9
        ex = (m9 > .05) | (m10 > .05)
        o1, nu, nd = overlap(d1, d2)
        ot, _, _ = overlap(trend, d2)
        o_both, _, _ = overlap(d1 / d1[ex].std() + trend / trend[ex].std(), d2)
        o_lvl, _, _ = overlap(-m9, d2)
        o_lvl2, _, _ = overlap(m9, d2)
        rows.append({"lineage": name, "cells": (int(a.sum()), int(b.sum()), int(c.sum())), "n_up": nu, "n_dn": nd,
                     "rho(d1,d2)": spearmanr(d1[ex], d2[ex]).statistic, "rho(trend,d2)": spearmanr(trend[ex], d2[ex]).statistic,
                     "rho(trend,d1)": spearmanr(trend[ex], d1[ex]).statistic,
                     "ovl_d1": o1, "ovl_trend": ot, "ovl_d1+trend": o_both, "ovl_level_null": max(o_lvl, o_lvl2)})
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
