"""How closely do the re-sequenced anchor nuclei reproduce the library-3 means of their assigned cell type?

Builds the anchor part of the final configuration twice with the export's own `build`: (a) without the two
developmental shifts (dev = 0, trend share = 0), to check the alignment itself; (b) as exported. Compares
means of counts per 10k, type by type and lineage by lineage, with the library-3 means."""
import numpy as np, pandas as pd
from experiments.round6_common import REAL, Mem, lineage_of
from experiments.round5_anchor import load_anchor
from experiments.round7_method import Anchor7
from experiments.round7_export import build
from experiments.bulk_trend import load_trend
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250, "display.max_rows", 100)
_, genes = board_spec("T1:test")
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
ax_raw, _ = load_anchor(genes)
trend, _ = load_trend(genes)
from experiments.round7_export import cells_hash
pool = np.flatnonzero(e10.lib == "3")
prev = (e9, np.arange(e9.n_cells))
def cp10k(m):
    return np.expm1(m.astype(np.float64))
lib3 = e10.X[pool].toarray()
labels3 = e10.labels[pool]
for tag, dev, ts in (("alignment only (no developmental shifts)", 0.0, 0.0), ("as exported (dev 0.5, trend share 0.4)", 0.5, 0.4)):
    cells, info, ax, main, libs, dkw = build(e10, ax_raw, 0.45, 0.0, 5000, 42, True, "type", dev, prev if dev else None, 4.0, trend if ts else None, ts, 20)
    assert np.array_equal(main, pool), "heart-library rows differ"
    print("cells sha256", cells_hash(cells))
    syn = cells[len(info["base_rows"]):]
    ctx = ax.context(e10, main, "min", "type", prev=prev, days_to_anchor=4.0) if dev else ax.context(e10, main, "min", "type")
    near = ctx["near"][info["anchor_rows"]]
    lin = ctx["lineage"][info["anchor_rows"]]
    rows = []
    for t in pd.Series(near).value_counts().index:
        a = near == t
        s = labels3 == t
        if a.sum() < 30 or s.sum() < 50:
            continue
        ma, ms = cp10k(syn[a]).mean(0), cp10k(lib3[s]).mean(0)
        ex = ms >= 0.5
        lr = np.log((ma[ex] + 0.1) / (ms[ex] + 0.1))
        rows.append(dict(type=t, lineage=lineage_of([t])[0], nuclei=int(a.sum()), lib3_cells=int(s.sum()), genes=int(ex.sum()),
                         corr_log=float(np.corrcoef(np.log(ma[ex] + 0.1), np.log(ms[ex] + 0.1))[0, 1]),
                         median_abs_logratio=float(np.median(np.abs(lr))), q90_abs_logratio=float(np.quantile(np.abs(lr), .9)),
                         share_within_20pct=float((np.abs(lr) < np.log(1.2)).mean())))
    print("\n===", tag)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # all nuclei against library 3 at lineage level
    lrows = []
    lin3 = lineage_of(labels3)
    for L in pd.Series(lin).value_counts().index:
        a, s = lin == L, lin3 == L
        if a.sum() < 30 or s.sum() < 50:
            continue
        ma, ms = cp10k(syn[a]).mean(0), cp10k(lib3[s]).mean(0)
        ex = ms >= 0.5
        lr = np.log((ma[ex] + 0.1) / (ms[ex] + 0.1))
        lrows.append(dict(lineage=L, nuclei=int(a.sum()), genes=int(ex.sum()), corr_log=float(np.corrcoef(np.log(ma[ex] + 0.1), np.log(ms[ex] + 0.1))[0, 1]),
                          median_abs_logratio=float(np.median(np.abs(lr))), q90_abs_logratio=float(np.quantile(np.abs(lr), .9))))
    print(pd.DataFrame(lrows).round(3).to_string(index=False))
    if not dev:
        # raw anchor (no alignment) for comparison: same nuclei, single-nucleus values
        raw = ax.x[info["anchor_rows"]].toarray()
        rr = []
        for L in pd.Series(lin).value_counts().index:
            a, s = lin == L, lin3 == L
            if a.sum() < 30 or s.sum() < 50:
                continue
            ma, ms = cp10k(raw[a]).mean(0), cp10k(lib3[s]).mean(0)
            ex = ms >= 0.5
            lr = np.log((ma[ex] + 0.1) / (ms[ex] + 0.1))
            rr.append(dict(lineage=L, corr_log=float(np.corrcoef(np.log(ma[ex] + 0.1), np.log(ms[ex] + 0.1))[0, 1]),
                           median_abs_logratio=float(np.median(np.abs(lr))), q90_abs_logratio=float(np.quantile(np.abs(lr), .9))))
        print("raw anchor nuclei (no alignment), lineage level:")
        print(pd.DataFrame(rr).round(3).to_string(index=False))

# -- contamination markers: cardiomyocyte genes in the non-cardiomyocyte lineages, before and after alignment
markers = ["Myl2", "Myl3", "Myh7", "Myh6", "Tnnt2", "Actc1", "Ttn", "Ryr2", "Myl7", "Myl4", "Slc8a1", "Nppa"]
gi = pd.Index(genes).get_indexer(markers)
cells0, info0, ax0, main0, _, _ = build(e10, ax_raw, 0.45, 0.0, 5000, 42, True, "type", 0.0, None, 4.0, None, 0.0, 20)
syn0 = cells0[len(info0["base_rows"]):]
ctx0 = ax0.context(e10, main0, "min", "type")
lin0 = ctx0["lineage"][info0["anchor_rows"]]
raw0 = ax0.x[info0["anchor_rows"]].toarray()
lin3 = lineage_of(labels3)
out = []
for L in ("Endo", "Cushion", "Epi", "Mac", "Other", "CM"):
    a, s3 = lin0 == L, lin3 == L
    if a.sum() < 30 or s3.sum() < 50:
        continue
    m3, mr, ms = cp10k(lib3[s3][:, gi]).mean(0), cp10k(raw0[a][:, gi]).mean(0), cp10k(syn0[a][:, gi]).mean(0)
    for g, x3, xr, xs in zip(markers, m3, mr, ms):
        out.append(dict(lineage=L, gene=g, lib3=x3, raw_anchor=xr, resampled=xs, ln_raw_over_lib3=np.log((xr + .1) / (x3 + .1)), ln_resampled_over_lib3=np.log((xs + .1) / (x3 + .1))))
frame = pd.DataFrame(out)
print("\n=== cardiomyocyte genes, counts per 10k (alignment only, no developmental shifts)")
print(frame.round(3).to_string(index=False))
non = frame[frame.lineage != "CM"]
print("non-CM lineages: median ln(raw/lib3) %.2f, median ln(resampled/lib3) %.2f, max ln(resampled/lib3) %.2f" % (
    non.ln_raw_over_lib3.median(), non.ln_resampled_over_lib3.median(), non.ln_resampled_over_lib3.max()))
# reference cells of the test: how many
import json
from experiments.round5_common import split_rows
try:
    sp = split_rows("E9.5") if callable(split_rows) else None
    print("split_rows(E9.5) keys:", {k: len(v) for k, v in sp.items()} if isinstance(sp, dict) else type(sp))
except Exception as exc:
    print("split_rows call failed:", repr(exc)[:200])
