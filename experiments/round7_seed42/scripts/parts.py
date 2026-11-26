"""Lineage shares of the three parts of the half-real file (drawn base, real stand-ins, synthetic nuclei)."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round6_common import REAL, LINEAGES, Mem, lineage_of
from experiments.round7_export import build, cells_hash
from t1_pipeline.data import board_spec
_, genes = board_spec("T1:test")
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes); e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
ax, _ = load_anchor(genes, verbose=False); trend, _ = load_trend(genes)
prev = (e9, np.arange(e9.n_cells))
cells, info, anchor, main, libs, dkw = build(e10, ax, 0.45, 0.0, 5000, return_info=True, level="type", dev=0.5, prev=prev, days=4.0, trend=trend, trend_share=0.4, k_prof=20, real_share=0.5)
print("cells sha256", cells_hash(cells))
rows, nd = info["base_rows"], info["n_drawn_base"]
ctx = anchor.context(e10, main, "min", "type", prev=prev, days_to_anchor=4.0)
parts = {"drawn base": lineage_of(e10.labels[rows[:nd]]), "real stand-ins": lineage_of(e10.labels[rows[nd:]]), "synthetic nuclei": ctx["lineage"][info["anchor_rows"]]}
tab = pd.DataFrame({k: pd.Series(v).value_counts(normalize=True) for k, v in parts.items()}).reindex(list(LINEAGES)).fillna(0)
tab["file"] = pd.Series(np.concatenate(list(parts.values()))).value_counts(normalize=True)
print((100 * tab).round(1).to_string())
print({k: len(v) for k, v in parts.items()})
g = (cells > 0).sum(1)
print("median genes detected:", float(np.median(g[:nd])), float(np.median(g[nd:len(rows)])), float(np.median(g[len(rows):])), float(np.median(g)))
from experiments.round6_common import depth_of
print("median UMI of source rows: drawn base", float(np.median(depth_of(e10.X[rows[:nd]]))), "stand-ins", float(np.median(depth_of(e10.X[rows[nd:]]))), "| raw nuclei picked (synthetic)", float(np.median(anchor.depth[info["anchor_rows"]])), "| nuclei represented by stand-ins", float(np.median(anchor.depth[info["stand_in_for"]])))
print("stand-in types:", pd.Series(e10.labels[rows[nd:]]).value_counts().to_dict())
