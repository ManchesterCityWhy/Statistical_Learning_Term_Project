"""Cell-type / lineage composition and depth of every released library (E8.5, E9.5, E10.5). No prediction involved."""
import numpy as np, pandas as pd
from experiments.round5_common import REAL, Mem
from experiments.round6_common import lineage_of, depth_of, LINEAGES, HEART
from t1_pipeline.data import board_spec
_, genes = board_spec("T1:val")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
rows = []
for st in ("E8.5", "E9.5", "E10.5"):
    m = Mem(REAL / f"data/T1/{st}_RNA.h5ad", genes)
    lib = np.asarray(m.lib).astype(str)
    lab = np.asarray(m.labels).astype(str)
    lin = lineage_of(lab)
    dep = depth_of(m.X)
    for l in np.unique(lib):
        s = lib == l
        heart = np.isin(lab[s], list(HEART))
        r = {"stage": st, "lib": l, "cells": int(s.sum()), "umi_med": float(np.median(dep[s])), "heart_types": round(float(heart.mean()), 3)}
        for k in LINEAGES:
            r[k] = round(float((lin[s] == k).mean()), 3)
        hp = np.isin(lin[s], ["CM", "Endo", "Cushion", "Epi"])
        r["CM_of_heart_proper"] = round(float((lin[s] == "CM").sum() / max(hp.sum(), 1)), 3)
        for k in ("CM", "Endo", "Epi", "Blood"):
            sel = s & (lin == k)
            r["umi_" + k] = float(np.median(dep[sel])) if sel.sum() else np.nan
        rows.append(r)
    if st == "E10.5":
        for l in np.unique(lib):
            s = lib == l
            vc = pd.Series(lab[s]).value_counts(normalize=True).round(3)
            print(f"E10.5 lib {l} cell types:", dict(vc.head(24)))
print(pd.DataFrame(rows).to_string(index=False))
