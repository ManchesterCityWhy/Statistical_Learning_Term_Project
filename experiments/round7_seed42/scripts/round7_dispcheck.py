"""Round 7: are the re-sequenced nuclei as different from one another as real cells of the same type are?

Score-free check. PCA (30 components) is fitted on real E10.5 library-3 cells; within every large cell type the
spread of real cells and of synthetic nuclei of that type is compared in this space:
    trace      sum of the variances along the 30 components
    nn         median distance of a cell to its nearest neighbour of the same group (clumping)
Synthetic nuclei: re-sequenced and moved back completely (the exported form, no developmental shifts), for
several neighbourhood sizes k_prof; and raw nuclei (harmonised, not pooled) for comparison.
"""
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from experiments.round6_common import REAL, Mem, lineage_of, depth_of
from experiments.round5_anchor import load_anchor
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250, "display.max_rows", 200)
_, genes = board_spec("T1:test")
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
ax_raw, _ = load_anchor(genes, verbose=False)
lib3 = np.flatnonzero(e10.lib == "3")
rng = np.random.default_rng(0)
fit_rows = np.sort(rng.choice(lib3, 5000, replace=False))
x_fit = e10.dense(fit_rows)
keep = x_fit.mean(0) >= 0.05
pca = PCA(n_components=30, random_state=0).fit(x_fit[:, keep])
print("genes used", int(keep.sum()), "explained variance of 30 PCs %.3f" % pca.explained_variance_ratio_.sum())
z_real = pca.transform(x_fit[:, keep])
lab_real = e10.labels[fit_rows]

def spread(z):
    nn = NearestNeighbors(n_neighbors=2).fit(z).kneighbors(z)[0][:, 1]
    return float(z.var(axis=0).sum()), float(np.median(nn))

def run(name, **kw):
    anchor = Anchor7(ax_raw)
    cells, info = predict7(e10, lib3, anchor, 5000, 0.45, seed=42, return_info=True, **kw)
    syn = cells[len(info["base_rows"]):]
    ctx = anchor.context(e10, lib3, "min", "type")
    near = ctx["near"][info["anchor_rows"]]
    z = pca.transform(syn[:, keep])
    out = []
    for t in ("LV-CM", "Endocardium-2", "Epicardium", "EndoMT Endocardium", "BEC", "OFT-CM"):
        a, r = near == t, lab_real == t
        if a.sum() < 60 or r.sum() < 60:
            continue
        n = min(a.sum(), r.sum(), 300)           # same group size for both, so that nn distances are comparable
        ia, ir = rng.choice(np.flatnonzero(a), n, replace=False), rng.choice(np.flatnonzero(r), n, replace=False)
        tr_s, nn_s = spread(z[ia]); tr_r, nn_r = spread(z_real[ir])
        d_cent = float(np.linalg.norm(z[ia].mean(0) - z_real[ir].mean(0)))
        out.append(dict(form=name, type=t, n=int(n), trace_real=tr_r, trace_syn=tr_s, trace_ratio=tr_s / tr_r, nn_real=nn_r, nn_syn=nn_s, nn_ratio=nn_s / nn_r,
                        centroid_distance=d_cent, genes_real=float(np.median((x_fit[np.flatnonzero(r)] > 0).sum(1))), genes_syn=float(np.median((syn[a] > 0).sum(1)))))
    frame = pd.DataFrame(out)
    print("\n===", name); print(frame.drop(columns="form").round(2).to_string(index=False))
    print("median trace ratio %.2f, median nn ratio %.2f" % (frame.trace_ratio.median(), frame.nn_ratio.median()), flush=True)
    return frame

import sys
res = []
if len(sys.argv) > 1 and sys.argv[1] == "tilt":
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        res.append(run(f"re-sequenced, moved back 1, k_prof 20, tilt {t:g}", bwd=1.0, pool_k="resample", level="type", k_prof=20, tilt=t))
else:
    for k in (20, 10, 5, 3, 1):
        res.append(run(f"re-sequenced, moved back 1, k_prof {k}", bwd=1.0, pool_k="resample", level="type", k_prof=k))
    res.append(run("re-sequenced, not moved back, k_prof 20", bwd=0.0, pool_k="resample", level="type", k_prof=20))
    res.append(run("raw nuclei (round 6 form)"))
allr = pd.concat(res)
print("\nsummary (median over cell types):")
print(allr.groupby("form", sort=False)[["trace_ratio", "nn_ratio", "centroid_distance", "genes_syn"]].median().round(2).to_string())
