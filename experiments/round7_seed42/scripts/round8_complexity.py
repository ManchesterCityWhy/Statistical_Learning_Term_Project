"""Score-free check: do cells of one lineage get sparser from stage to stage in the released data?

Per stage, library and lineage: cells, median UMI, median detected genes, median of sum_g sqrt(x_g) (the per-cell quantity
the variogram metric is built from), and the same two after thinning every cell to a common depth (5,000 UMI; cells below it
are left out), which removes the sequencing depth and leaves the complexity of the transcriptome."""
import numpy as np
import pandas as pd

from experiments.round6_common import REAL, Mem, depth_of, lineage_of, thin_rows
from t1_pipeline.data import board_spec

pd.set_option("display.width", 250)
_, genes = board_spec("T1:test")
rows = []
COMMON = 5000.0
for stage in ("E8.5", "E9.5", "E10.5"):
    m = Mem(REAL / f"data/T1/{stage}_RNA.h5ad", genes)
    lin = lineage_of(m.labels)
    depth = depth_of(m.X)
    for lib in np.unique(m.lib):
        for L in ("CM", "Endo", "Cushion", "Epi", "Blood", "Mac", "Other", "ALL"):
            sel = np.flatnonzero((m.lib == lib) & ((lin == L) if L != "ALL" else True))
            if len(sel) < 80:
                continue
            x = m.X[sel]
            d = depth[sel]
            g = np.diff(x.indptr)
            srt = x.copy()
            srt.data = np.sqrt(srt.data)
            s = np.asarray(srt.sum(axis=1)).ravel()
            ok = np.flatnonzero(d >= COMMON)
            rng = np.random.default_rng(1)
            if len(ok) > 1500:
                ok = np.sort(rng.choice(ok, 1500, replace=False))
            g5 = s5 = np.nan
            if len(ok) >= 50:
                t = thin_rows(x[ok], np.full(len(ok), COMMON), rng)
                g5 = float(np.median(np.diff(t.indptr)))
                t.data = np.sqrt(t.data)
                s5 = float(np.median(np.asarray(t.sum(axis=1)).ravel()))
            rows.append({"stage": stage, "lib": lib, "lineage": L, "cells": len(sel), "umi_median": float(np.median(d)), "genes_median": float(np.median(g)),
                         "sum_sqrt_median": float(np.median(s)), "share_ge_5k": float((d >= COMMON).mean()), "genes_at_5k": g5, "sum_sqrt_at_5k": s5})
frame = pd.DataFrame(rows)
print(frame.round(3).to_string())
frame.to_csv(REAL / "runs" / "round8_probe2" / "complexity.csv", index=False)
