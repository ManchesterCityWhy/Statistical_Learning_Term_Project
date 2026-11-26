"""Sanity statistics of the exported file: base part vs synthetic part, compared with real library-3 cells."""
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from experiments.round6_common import REAL, Mem, lineage_of
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250, "display.max_rows", 100)
_, genes = board_spec("T1:test")
g = pd.Index(genes)
p = REAL / "predictions/round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4/E12.5_prediction.h5ad"
a = ad.read_h5ad(p)
x = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float32)
base, syn = x[:2750], x[2750:]
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
lib3 = e10.X[np.flatnonzero(e10.lib == "3")].toarray()
def desc(name, m):
    top = np.expm1(m).max(axis=1) / 1e4
    return dict(part=name, cells=len(m), genes_med=float(np.median((m > 0).sum(1))), genes_q05=float(np.quantile((m > 0).sum(1), .05)),
                genes_q95=float(np.quantile((m > 0).sum(1), .95)), top_gene_share_med=float(np.median(top)), top_gene_share_q99=float(np.quantile(top, .99)),
                cells_top_gene_over_30pct=int((top > .3).sum()), max_value=float(m.max()))
print(pd.DataFrame([desc("real library 3", lib3), desc("file: base part", base), desc("file: synthetic part", syn)]).round(3).to_string(index=False))
markers = ["Tnnt2", "Myl2", "Myl3", "Myl7", "Myl4", "Myh6", "Myh7", "Nppa", "Ttn", "Actc1", "Top2a", "Hba-x", "Hbb-y", "Hbb-bh1", "Hba-a1", "Hbb-bs",
           "Col1a1", "Col3a1", "Postn", "Wt1", "Upk3b", "Pecam1", "Npr3", "Cdh5", "Fabp4", "Sox9", "Twist1", "Malat1", "Xist", "Rps2", "Rpl4", "mt-Co1", "C1qb"]
idx = g.get_indexer(markers)
tab = pd.DataFrame({"lib3_mean": lib3[:, idx].mean(0), "base_mean": base[:, idx].mean(0), "syn_mean": syn[:, idx].mean(0), "file_mean": x[:, idx].mean(0),
                    "lib3_det": (lib3[:, idx] > 0).mean(0), "base_det": (base[:, idx] > 0).mean(0), "syn_det": (syn[:, idx] > 0).mean(0)}, index=markers)
print(tab.round(3).to_string())
# genome-wide: pseudobulk and detection of the two parts against library 3
pb3, pbb, pbs = lib3.mean(0), base.mean(0), syn.mean(0)
d3, dbb, ds = (lib3 > 0).mean(0), (base > 0).mean(0), (syn > 0).mean(0)
expr = pb3 > .01
print("genes with mean > 0.01 in library 3:", int(expr.sum()))
print("corr pseudobulk lib3 vs base part %.4f | vs synthetic part %.4f" % (np.corrcoef(pb3[expr], pbb[expr])[0, 1], np.corrcoef(pb3[expr], pbs[expr])[0, 1]))
print("corr detection  lib3 vs base part %.4f | vs synthetic part %.4f" % (np.corrcoef(d3[expr], dbb[expr])[0, 1], np.corrcoef(d3[expr], ds[expr])[0, 1]))
big = np.argsort(-(pbs - pb3))[:25]; small = np.argsort(pbs - pb3)[:25]
print("synthetic part higher than library 3 (mean log1p):", ", ".join(f"{genes[i]} {pbs[i]-pb3[i]:+.2f}" for i in big))
print("synthetic part lower  than library 3 (mean log1p):", ", ".join(f"{genes[i]} {pbs[i]-pb3[i]:+.2f}" for i in small))
