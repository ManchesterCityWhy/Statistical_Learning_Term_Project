"""Sanity statistics of the half-real file: drawn base, real stand-ins, synthetic nuclei; against real library-3 cells."""
import json
import numpy as np, pandas as pd, anndata as ad
from experiments.round6_common import REAL, Mem, lineage_of
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250, "display.max_rows", 100)
_, genes = board_spec("T1:test")
g = pd.Index(genes)
name = "round7_heartlib_f0.45_pos0_type_k20_dev0.5_trend0.4_real0.5"
m = json.loads((REAL / "predictions" / name / "manifest.json").read_text())
a = ad.read_h5ad(REAL / "predictions" / name / "E12.5_prediction.h5ad")
x = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float32)
n_drawn, n_real = m["real_cells"]["drawn_in_natural_composition"], m["base_cells"]
parts = {"file: drawn base": x[:n_drawn], "file: real stand-ins": x[n_drawn:n_real], "file: synthetic nuclei": x[n_real:]}
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
lib3 = e10.X[np.flatnonzero(e10.lib == "3")].toarray()
def desc(nm, mat):
    top = np.expm1(mat).max(axis=1) / 1e4
    gd = (mat > 0).sum(1)
    return dict(part=nm, cells=len(mat), genes_med=float(np.median(gd)), genes_q05=float(np.quantile(gd, .05)), genes_q95=float(np.quantile(gd, .95)),
                top_gene_share_med=float(np.median(top)), top_gene_share_q99=float(np.quantile(top, .99)), max_value=float(mat.max()))
print(pd.DataFrame([desc("real library 3", lib3)] + [desc(k, v) for k, v in parts.items()] + [desc("file: all", x)]).round(3).to_string(index=False))
print("distinct rows in the file:", len(np.unique(x, axis=0)), "| expm1 row sums", float(np.expm1(x.astype(np.float64)).sum(1).min()), float(np.expm1(x.astype(np.float64)).sum(1).max()))
markers = ["Tnnt2", "Myl2", "Myl7", "Myh7", "Nppa", "Ttn", "Top2a", "Hba-x", "Hbb-y", "Hbb-bh1", "Col1a1", "Col3a1", "Postn", "Wt1", "Upk3b", "Pecam1", "Npr3", "Sox9", "Twist1", "Xist", "Rps2", "mt-Co1"]
idx = g.get_indexer(markers)
tab = pd.DataFrame({"lib3": lib3[:, idx].mean(0), **{k.replace("file: ", ""): v[:, idx].mean(0) for k, v in parts.items()}, "file": x[:, idx].mean(0)}, index=markers)
print(tab.round(3).to_string())
print("manifest real_cells:", {k: v for k, v in m["real_cells"].items() if not isinstance(v, dict)})
print("lineage shares of the file:", m["file"]["lineage_shares_total"])
req, got = m["real_cells"]["requested_celltype_shares"], m["real_cells"]["stand_in_celltype_shares"]
print("stand-in cell types, requested vs drawn (shares):", {k: (req.get(k, 0), got.get(k, 0)) for k in sorted(set(req) | set(got), key=lambda k: -req.get(k, 0))})
