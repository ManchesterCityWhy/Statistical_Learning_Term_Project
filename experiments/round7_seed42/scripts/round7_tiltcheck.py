"""Round 7: score-free checks of the tilted re-sequenced nuclei (run before any adoption).

1. genes detected at matched depth (9,000-13,000 UMI) against real E10.5 library-3 cells, tilt 0 vs tilt T (k_prof 20)
2. alignment: type means of the nuclei against the library-3 means of the same type (counts per 10k), tilt 0 vs T
3. extreme values: share of a cell's counts taken by its largest gene, largest log1p value, against real cells
4. how often tilt x delta hits the clip
"""
import sys
import numpy as np, pandas as pd
from experiments.round5_anchor import load_anchor
from experiments.round6_common import REAL, SEED, Mem, depth_of, lineage_of
from experiments.round7_method import Anchor7, _factors, MIN_CELLS, TILT_CLIP, predict7
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250, "display.max_rows", 200)
T = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
_, genes = board_spec("T1:test")
ax, _ = load_anchor(genes, verbose=False)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
lib3 = np.flatnonzero(e10.lib == "3")
depth_s, lin_s = depth_of(e10.X[lib3]), lineage_of(e10.labels[lib3])
real = e10.X[lib3]

# -- 1. detection at matched depth (same construction as round7_kcheck)
anchor = Anchor7(ax)
ctx = anchor.context(e10, lib3, "min", "type")
rng = np.random.default_rng(SEED)
picked = np.sort(rng.choice(ax.shape[0], 1500, replace=False))
lin_a = ctx["lineage"][picked]
lf = np.tile(ctx["c"], (len(picked), 1)) + _factors(ctx["near"][picked], lin_a, ctx["d"], -1.0, ax.shape[1])
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
        w = m & (d >= 9000) & (d <= 13000)
        if w.sum() >= 30:
            rows.append(dict(sample=name, lineage=k, genes=float(np.median(g[w])), sum_sqrt=float(np.median(sq[w]))))
summarise("real E10.5 library 3", real, depth_s, lin_s)
for t in (0.0, T):
    for k in (10, 20):
        summarise(f"re-sequenced, k_prof {k}, tilt {t:g}", anchor.resampled(picked, lf, depths, k_prof=k, seed=SEED, tilt=t), depths, lin_a)
f = pd.DataFrame(rows)
print("1. median genes detected, cells with 9,000-13,000 UMI:")
tab = f.pivot_table(index="sample", columns="lineage", values="genes", sort=False)
print(tab.round(0).to_string())
print("   relative to real cells (all lineages):", (tab["all"] / tab.loc["real E10.5 library 3", "all"] - 1).round(4).to_dict())
print("   median sum over genes of sqrt(log1p value):")
print(f.pivot_table(index="sample", columns="lineage", values="sum_sqrt", sort=False).round(0).to_string())

# -- 4. clip rate of tilt x delta on these nuclei
nn = anchor.neighbours(20)[:, :20]
comp = anchor._svd.components_.astype(np.float64)
zn = anchor._zraw[nn[picked[:300]]].astype(np.float64)
own = T * ((zn[:, 0, :] - zn.mean(axis=1)) @ comp)
expressed = np.asarray(ax[picked[:300]].mean(axis=0)).ravel() > 0.05
print("4. tilt x delta over expressed genes: sd %.3f, |.| > 1: %.4f, clipped (|.| >= %.1f): %.5f" % (
    own[:, expressed].std(), (np.abs(own[:, expressed]) > 1).mean(), TILT_CLIP, (np.abs(own[:, expressed]) >= TILT_CLIP).mean()))

# -- 2 and 3 on the nuclei of the exported configuration without developmental shifts
lib3_dense_rows = np.sort(np.random.default_rng(1).choice(lib3, 4000, replace=False))
real_d = e10.dense(lib3_dense_rows)
lab_real = e10.labels[lib3_dense_rows]
cp = lambda m: np.expm1(m.astype(np.float64))
for t in (0.0, T):
    a7 = Anchor7(ax)
    cells, info = predict7(e10, lib3, a7, 5000, 0.45, seed=42, return_info=True, bwd=1.0, pool_k="resample", level="type", k_prof=20, tilt=t)
    syn = cells[len(info["base_rows"]):]
    c7 = a7.context(e10, lib3, "min", "type")
    near = c7["near"][info["anchor_rows"]]
    out = []
    for ty in pd.Series(near).value_counts().index:
        a, s = near == ty, e10.labels[lib3] == ty
        if a.sum() < 30 or s.sum() < 50 or lineage_of([ty])[0] == "Other":
            continue
        ma, ms = cp(syn[a]).mean(0), np.asarray(np.expm1(e10.X[lib3[s]]).mean(axis=0)).ravel()
        ex = ms >= 0.5
        lr = np.log((ma[ex] + .1) / (ms[ex] + .1))
        out.append(dict(type=ty, nuclei=int(a.sum()), median_abs=float(np.median(np.abs(lr))), q90_abs=float(np.quantile(np.abs(lr), .9)), mean_signed=float(lr.mean())))
    fr = pd.DataFrame(out)
    print(f"\n2. alignment of type means, tilt {t:g}: median |ln ratio| per type from %.3f to %.3f (median %.3f); q90 from %.3f to %.3f; mean signed ln ratio from %.3f to %.3f" % (
        fr.median_abs.min(), fr.median_abs.max(), fr.median_abs.median(), fr.q90_abs.min(), fr.q90_abs.max(), fr.mean_signed.min(), fr.mean_signed.max()))
    top = cp(syn).max(axis=1) / 1e4
    print(f"3. tilt {t:g}: largest-gene share median %.4f q99 %.4f max %.4f; nuclei with a gene above 20%%: %d; largest log1p value %.3f; genes detected median %.0f (q05 %.0f, q95 %.0f)" % (
        np.median(top), np.quantile(top, .99), top.max(), int((top > .2).sum()), syn.max(), np.median((syn > 0).sum(1)), np.quantile((syn > 0).sum(1), .05), np.quantile((syn > 0).sum(1), .95)))
top = cp(real_d).max(axis=1) / 1e4
nb = lab_real != "Blood"
print("\n3. real library-3 cells: largest-gene share median %.4f q99 %.4f max %.4f; cells with a gene above 20%%: %d of %d; largest log1p value %.3f" % (
    np.median(top), np.quantile(top, .99), top.max(), int((top > .2).sum()), len(top), real_d.max()))
print("   real library-3 cells without blood: largest-gene share median %.4f q99 %.4f max %.4f" % (np.median(top[nb]), np.quantile(top[nb], .99), top[nb].max()))
