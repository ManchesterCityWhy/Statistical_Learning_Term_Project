"""Read-only sanity statistics of an exported prediction (no scoring)."""
import sys, hashlib, numpy as np, anndata as ad
from scipy import sparse
from pathlib import Path
sys.path.insert(0, str(Path.home() / "statlearn_real"))
from t1_pipeline.data import board_spec, validate_prediction
path = Path(sys.argv[1])
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2500   # number of base cells (they come first)
print("validate_prediction:", validate_prediction(path, "T1:test"))
a = ad.read_h5ad(path)
X = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
spec, genes = board_spec("T1:test")
print("shape", X.shape, "dtype", X.dtype, "var order ok", list(a.var_names) == genes, "obs cols", list(a.obs.columns), "obsm", list(a.obsm.keys()))
print("finite", bool(np.isfinite(X).all()), "min", float(X.min()), "max", float(X.max()))
lib = np.expm1(X.astype(np.float64)).sum(1)
print("expm1 row sums: min %.4f max %.4f" % (lib.min(), lib.max()))
det = (X > 0).sum(1)
print(f"detected genes per cell: first {NB} (base) median", np.median(det[:NB]), f"last {len(det) - NB} (anchor) median", np.median(det[NB:]), "all quantiles", np.quantile(det, [.05, .25, .5, .75, .95]))
_, idx = np.unique(X, axis=0, return_index=True)
print("distinct rows", len(idx), "of", len(X))
ref = ad.read_h5ad(Path.home() / "statlearn_real/data/T1/E10.5_RNA.h5ad")
order = ref.var_names.get_indexer(genes)
R = ref.X.tocsr()[:, order]
frac_ref = np.asarray((R > 0).mean(0)).ravel()
frac_pred = (X > 0).mean(0)
print("genes detected in >=1% of E10.5 cells:", int((frac_ref >= .01).sum()), "of which never in prediction:", int(((frac_ref >= .01) & (frac_pred == 0)).sum()))
pb_ref, pb_pred = np.asarray(R.mean(0)).ravel(), X.mean(0)
print("corr of gene means with E10.5 (all libraries):", float(np.corrcoef(pb_ref, pb_pred)[0, 1]))
g = {n: i for i, n in enumerate(genes)}
for name in ("Tnnt2", "Myh6", "Nppa", "Col1a1", "Postn", "Pecam1", "Wt1", "Hbb-y", "Hba-x", "Hbb-bh1", "Hbb-bs", "Hba-a1", "Xist", "Rpl13", "Rps28", "Malat1", "Actb", "Top2a", "Ckm", "Mb"):
    i = g[name]
    print(f"{name:8s} E10.5 all {pb_ref[i]:.3f} | prediction {pb_pred[i]:.3f} | base part {X[:NB, i].mean():.3f} | anchor part {X[NB:, i].mean():.3f}")
