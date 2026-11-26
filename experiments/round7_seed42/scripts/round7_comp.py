"""Lineage composition and depth by lineage of every stage / library, and of the anchor (as assigned to lib-3 types)."""
import numpy as np, pandas as pd
from experiments.round5_anchor import load_anchor
from experiments.round6_common import REAL, Mem, depth_of, lineage_of, LINEAGES, centroid_transfer
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250)
_, genes = board_spec("T1:val")
rows = []
stages = {}
for st in ("E8.5", "E9.5", "E10.5"):
    m = Mem(REAL / f"data/T1/{st}_RNA.h5ad", genes)
    stages[st] = m
    lin = lineage_of(m.labels); d = depth_of(m.X)
    for lib in list(np.unique(m.lib)) + ["all"]:
        sel = np.ones(m.n_cells, bool) if lib == "all" else m.lib == lib
        r = {"sample": f"{st} lib {lib}", "cells": int(sel.sum())}
        for k in LINEAGES:
            mk = sel & (lin == k)
            r[k] = round(100 * mk.sum() / sel.sum(), 1)
            r[k + "_umi"] = int(np.median(d[mk])) if mk.sum() >= 20 else -1
        heart = sel & (lin != "Other")
        r["CM_of_heart"] = round(100 * (sel & (lin == "CM")).sum() / max(heart.sum(), 1), 1)
        rows.append(r)
ax, _ = load_anchor(genes, verbose=False)
e10 = stages["E10.5"]; lib3 = np.flatnonzero(e10.lib == "3")
near = centroid_transfer(e10.X[lib3], e10.labels[lib3], ax)
lin = lineage_of(near); d = depth_of(ax)
r = {"sample": "E14.5 anchor", "cells": ax.shape[0]}
for k in LINEAGES:
    mk = lin == k
    r[k] = round(100 * mk.mean(), 1); r[k + "_umi"] = int(np.median(d[mk])) if mk.sum() >= 20 else -1
r["CM_of_heart"] = round(100 * (lin == "CM").sum() / max((lin != "Other").sum(), 1), 1)
rows.append(r)
f = pd.DataFrame(rows).set_index("sample")
print(f[[k for k in LINEAGES] + ["CM_of_heart", "cells"]].to_string())
print(f[[k + "_umi" for k in LINEAGES]].to_string())
print("anchor nuclei by nearest lib-3 type:"); print(pd.Series(near).value_counts().to_string())
print("lib3 types:"); print(pd.Series(e10.labels[lib3]).value_counts().to_string())
