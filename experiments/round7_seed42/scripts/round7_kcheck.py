"""How many genes does a re-sequenced anchor nucleus show, compared with a real E10.5 heart-library cell of the same
lineage and UMI depth? (Calibrates the neighbourhood size k_prof without any score.)"""
import numpy as np, pandas as pd
from experiments.round5_anchor import load_anchor
from experiments.round6_common import REAL, SEED, Mem, depth_of, lineage_of
from experiments.round7_method import Anchor7, _factors, MIN_CELLS
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250)
_, genes = board_spec("T1:test")
ax, _ = load_anchor(genes, verbose=False)
anchor = Anchor7(ax)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
lib3 = np.flatnonzero(e10.lib == "3")
ctx = anchor.context(e10, lib3, "min", "type")
rng = np.random.default_rng(SEED)
depth_s, lin_s = depth_of(e10.X[lib3]), lineage_of(e10.labels[lib3])
genes_s = np.diff(e10.X[lib3].indptr)
sq_s = np.array([np.sqrt(e10.X[int(r)].data).sum() for r in lib3[:0]])  # placeholder
picked = np.sort(rng.choice(ax.shape[0], 1500, replace=False))
lin_a = ctx["lineage"][picked]
n_genes = ax.shape[1]
lf = np.tile(ctx["c"], (len(picked), 1)) + _factors(ctx["near"][picked], lin_a, ctx["d"], -1.0, n_genes)
depths = np.empty(len(picked))
for lin in np.unique(lin_a):
    sel = np.flatnonzero(lin_a == lin)
    cand = depth_s[lin_s == lin] if (lin_s == lin).sum() >= MIN_CELLS and lin != "Other" else depth_s[lin_s != "Blood"]
    depths[sel] = rng.choice(cand, len(sel), replace=True)
rows = []
def summarise(name, x, d, lin):
    g = (x > 0).sum(axis=1) if not hasattr(x, "indptr") else np.diff(x.indptr)
    sq = np.asarray(np.sqrt(x).sum(axis=1)).ravel() if not hasattr(x, "indptr") else np.asarray(x.sqrt().sum(axis=1)).ravel()
    for k in ("CM", "Endo", "Cushion", "Epi", "all"):
        m = np.ones(len(lin), bool) if k == "all" else lin == k
        # compare at matched depth: cells between 9k and 13k UMI
        w = m & (d >= 9000) & (d <= 13000)
        if w.sum() < 30:
            continue
        rows.append(dict(sample=name, lineage=k, cells=int(w.sum()), genes=float(np.median(g[w])), sum_sqrt=float(np.median(sq[w])), umi=float(np.median(d[w]))))
summarise("real E10.5 library 3", e10.X[lib3], depth_s, lin_s)
for k in (1, 2, 3, 4, 5, 7, 10, 20):
    cells = anchor.resampled(picked, lf, depths, k_prof=k, seed=SEED)
    summarise(f"re-sequenced, k_prof {k}", cells, depths, lin_a)
f = pd.DataFrame(rows)
print("median genes detected, cells with 9,000-13,000 UMI:")
print(f.pivot_table(index="sample", columns="lineage", values="genes", sort=False).round(0).to_string())
print("median sum over genes of sqrt(log1p value):")
print(f.pivot_table(index="sample", columns="lineage", values="sum_sqrt", sort=False).round(0).to_string())
print(f.pivot_table(index="sample", columns="lineage", values="umi", sort=False).round(0).to_string())
