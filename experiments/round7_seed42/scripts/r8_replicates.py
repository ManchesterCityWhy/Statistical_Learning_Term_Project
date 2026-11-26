"""Is the shared part of the anchor / E10.5 ratio a developmental axis or something that varies between libraries
of one stage?  Score-free. For pairs of libraries (replicates of a stage, consecutive stages), per lineage:
log-ratio of lineage means, its rank correlation with the shared part and the regression slope on it."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from experiments.round5_anchor import load_anchor
from experiments.round5_common import REAL, Mem
from experiments.round6_common import lineage_of
from experiments.round7_method import Anchor7
from experiments.round6_anchor import lineage_means
from t1_pipeline.data import board_spec
_, genes = board_spec("T1:val")
ax, _ = load_anchor(list(genes), verbose=False)
anchor = Anchor7(ax)
st = {s: Mem(REAL / f"data/T1/{s}_RNA.h5ad", list(genes)) for s in ("E8.5", "E9.5", "E10.5")}
e10 = st["E10.5"]
lib3 = np.flatnonzero(np.asarray(e10.lib).astype(str) == "3")
ctx = anchor.context(e10, lib3, "min", "lineage")
sh = -ctx["c"].astype(float)
ms3 = lineage_means(e10.X[lib3], lineage_of(e10.labels[lib3]), 50)
best = np.max(np.stack(list(ms3.values())), axis=0)
sel0 = (best >= 1.0) & (sh != 0)
def means(stage, lib):
    m = st[stage]
    rows = np.arange(m.n_cells) if lib is None else np.flatnonzero(np.asarray(m.lib).astype(str) == lib)
    return lineage_means(m.X[rows], lineage_of(m.labels[rows]), 50)
pairs = [("E8.5 lib1 -> E8.5 lib2 (replicates)", ("E8.5", "1"), ("E8.5", "2")),
         ("E9.5 lib1 -> E9.5 lib2 (replicates)", ("E9.5", "1"), ("E9.5", "2")),
         ("E10.5 lib1 -> lib2", ("E10.5", "1"), ("E10.5", "2")),
         ("E10.5 lib2 -> lib3", ("E10.5", "2"), ("E10.5", "3")),
         ("E10.5 lib1 -> lib3", ("E10.5", "1"), ("E10.5", "3")),
         ("E8.5 -> E9.5 (all)", ("E8.5", None), ("E9.5", None)),
         ("E9.5 lib1 -> E10.5 lib3", ("E9.5", "1"), ("E10.5", "3")),
         ("E9.5 lib2 -> E10.5 lib3", ("E9.5", "2"), ("E10.5", "3")),
         ("E9.5 (all) -> E10.5 lib1", ("E9.5", None), ("E10.5", "1")),
         ("E9.5 (all) -> E10.5 lib2", ("E9.5", None), ("E10.5", "2")),
         ("E8.5 (all) -> E10.5 lib3", ("E8.5", None), ("E10.5", "3"))]
rows = []
g = np.asarray(genes)
ribo = np.char.startswith(g, "Rpl") | np.char.startswith(g, "Rps")
for name, a, b in pairs:
    ma, mb = means(*a), means(*b)
    r = {"pair": name}
    for k in ("CM", "Endo", "Epi"):
        if k in ma and k in mb:
            v = np.log((mb[k] + .1) / (ma[k] + .1))
            r[f"rho {k}"] = round(float(spearmanr(v[sel0], sh[sel0]).correlation), 3)
            r[f"slope {k}"] = round(float((v[sel0] * sh[sel0]).sum() / (sh[sel0] ** 2).sum()), 3)
            r[f"ribo {k}"] = round(float(v[ribo & (best >= 1)].mean()), 3)
    rows.append(r)
pd.set_option("display.width", 250)
print("shared part: ribosomal protein genes mean", round(float(sh[ribo & (best >= 1)].mean()), 3))
print(pd.DataFrame(rows).to_string(index=False))
