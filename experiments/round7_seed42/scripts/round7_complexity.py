"""Genes detected at matched UMI depth, by stage / library / lineage (is the later heart library less complex?)."""
import numpy as np, pandas as pd
from experiments.round5_anchor import load_anchor
from experiments.round6_common import REAL, Mem, depth_of, lineage_of, centroid_transfer
from experiments.round5_exp4 import thin_rows
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250)
_, genes = board_spec("T1:val")
rng = np.random.default_rng(0)
TARGET = 6000.0
rows = []
def stats(name, x, lin):
    d = depth_of(x)
    for k in ["CM", "Endo", "Epi", "Cushion", "Blood", "Other", "all"]:
        m = (lin == k) if k != "all" else np.ones(len(lin), bool)
        m &= d >= TARGET
        if m.sum() < 80:
            continue
        idx = np.flatnonzero(m)
        idx = idx[rng.choice(len(idx), min(1500, len(idx)), replace=False)]
        t = thin_rows(x[idx], np.full(len(idx), TARGET), rng)      # every cell thinned to exactly 6,000 UMI
        g = np.diff(t.indptr)
        top = np.array([np.sort(np.expm1(t[i].data))[-50:].sum() / 1e4 for i in range(min(300, t.shape[0]))])
        rows.append(dict(sample=name, lineage=k, cells=int(m.sum()), genes_at_6k=float(np.median(g)), top50_share=float(np.median(top)),
                         mean_sqrt=float(np.sqrt(t.data).sum() / t.shape[0]), median_umi_raw=float(np.median(d[m]))))
for st in ("E8.5", "E9.5", "E10.5"):
    m = Mem(REAL / f"data/T1/{st}_RNA.h5ad", genes)
    lin = lineage_of(m.labels)
    for lib in np.unique(m.lib):
        sel = np.flatnonzero(m.lib == lib)
        stats(f"{st} lib {lib}", m.X[sel], lin[sel])
    if st == "E10.5":
        e10 = m
ax, _ = load_anchor(genes, verbose=False)
lib3 = np.flatnonzero(e10.lib == "3")
near = centroid_transfer(e10.X[lib3], e10.labels[lib3], ax)
d = depth_of(ax)
la = lineage_of(near)
# anchor nuclei are mostly below 6k UMI: thin everything (cells and anchor) to 2,500 for a second comparison
def stats2(name, x, lin, target=2500.0):
    d = depth_of(x)
    for k in ["CM", "Endo", "Epi", "all"]:
        m = ((lin == k) if k != "all" else np.ones(len(lin), bool)) & (d >= target)
        if m.sum() < 80:
            continue
        idx = np.flatnonzero(m); idx = idx[rng.choice(len(idx), min(1500, len(idx)), replace=False)]
        t = thin_rows(x[idx], np.full(len(idx), target), rng)
        rows2.append(dict(sample=name, lineage=k, genes_at_2500=float(np.median(np.diff(t.indptr))), mean_sqrt=float(np.sqrt(t.data).sum() / t.shape[0])))
rows2 = []
stats2("E14.5 anchor", ax, la)
stats2("E10.5 lib 3", e10.X[lib3], lineage_of(e10.labels[lib3]))
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
stats2("E9.5", e9.X, lineage_of(e9.labels))
f = pd.DataFrame(rows)
print(f.pivot_table(index="sample", columns="lineage", values="genes_at_6k", sort=False).round(0).to_string())
print("share of counts in the 50 most abundant genes of a cell (at 6k):")
print(f.pivot_table(index="sample", columns="lineage", values="top50_share", sort=False).round(3).to_string())
print("mean over cells of sum_g sqrt(log1p value) at 6k (the quantity the variogram is built from):")
print(f.pivot_table(index="sample", columns="lineage", values="mean_sqrt", sort=False).round(0).to_string())
print(pd.DataFrame(rows2).pivot_table(index="sample", columns="lineage", values="genes_at_2500", sort=False).round(0).to_string())
