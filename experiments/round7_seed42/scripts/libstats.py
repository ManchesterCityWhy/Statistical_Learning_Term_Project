import numpy as np, anndata as ad, pandas as pd
from pathlib import Path
from scipy import sparse
REAL = Path.home() / "statlearn_real"
pd.set_option("display.width", 250)
rows = []
for st in ("E8.5", "E9.5", "E10.5"):
    a = ad.read_h5ad(REAL / f"data/T1/{st}_RNA.h5ad")
    x = a.X.tocsr(); x.eliminate_zeros()
    depth = 1e4 / np.expm1(np.minimum.reduceat(x.data, x.indptr[:-1]))
    genes = np.diff(x.indptr)
    lib = np.array([n.rsplit("_", 1)[-1] for n in a.obs_names])
    lab = a.obs["celltype"].astype(str).to_numpy()
    for l in np.unique(lib):
        m = lib == l
        q = np.quantile(depth[m], [.1, .25, .5, .75, .9])
        rows.append(dict(stage=st, lib=l, cells=int(m.sum()), total_M=round(depth[m].sum() / 1e6, 1), mean=round(depth[m].mean()),
                         q10=round(q[0]), q25=round(q[1]), median=round(q[2]), q75=round(q[3]), q90=round(q[4]), genes_med=int(np.median(genes[m]))))
    if st == "E10.5":
        m3 = lib == "3"
        t = pd.DataFrame({"type": lab[m3], "depth": depth[m3]}).groupby("type")["depth"].agg(["size", "median", "mean"]).round(0).sort_values("size", ascending=False)
        print("E10.5 library 3 depth by cell type"); print(t.to_string())
print(pd.DataFrame(rows).to_string(index=False))
