"""How much of the anchor / source log-ratio of a lineage is persistent development?  (no score, no target)

For every lineage: d = lineage-specific part of log(anchor mean / E10.5 library-3 mean), c = part shared by all lineages,
vel = log(E10.5 mean / E9.5 mean) inside the challenge data (one day), bulk = same-laboratory bulk log-ratio E9.5 -> E14.5.
If d were pure development at a constant rate, vel = d / 4 (slope 0.25, correlation near 1)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from experiments.round5_anchor import load_anchor
from experiments.round5_common import REAL, Mem
from experiments.round6_common import lineage_of
from experiments.round7_method import Anchor7, OWN_SHARE, EPS
from experiments.round6_anchor import lineage_means
from experiments.bulk_trend import load_trend
from t1_pipeline.data import board_spec
_, genes = board_spec("T1:val")
ax, _ = load_anchor(genes, verbose=False)
anchor = Anchor7(ax)
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
lib3 = np.flatnonzero(np.asarray(e10.lib).astype(str) == "3")
ctx = anchor.context(e10, lib3, "min", "lineage", prev=(e9, np.arange(e9.n_cells)), days_to_anchor=4.0)
trend = load_trend(genes)[0]
trend = np.asarray(trend, dtype=float)
ms = lineage_means(e10.X[lib3], lineage_of(e10.labels[lib3]), 50)
best = np.max(np.stack(list(ms.values())), axis=0)
pd.set_option("display.width", 250)
rows = []
for k in ("CM", "Endo", "Epi", "Cushion", "Mac"):
    if k not in ctx["d"]:
        continue
    d, c = ctx["d"][k].astype(float), ctx["c"].astype(float)
    raw = d - c
    vel = ctx["vel"].get(k)
    dev = ctx["dev"].get(k)
    own = ms[k] >= OWN_SHARE * best
    for name, sel in (("expressed>=1", ms[k] >= 1.0), ("expressed>=1 & own", (ms[k] >= 1.0) & own), ("expressed>=0.2 & own", (ms[k] >= 0.2) & own)):
        r = {"lineage": k, "genes": name, "n": int(sel.sum())}
        if vel is not None:
            v = vel.astype(float)[sel]
            for nm, x in (("d", d[sel]), ("raw", raw[sel])):
                r[f"rho(vel,{nm})"] = round(float(spearmanr(v, x).correlation), 3)
                r[f"slope vel~{nm}"] = round(float((v * x).sum() / (x * x).sum()), 3)
            r["rho(vel,bulk)"] = round(float(spearmanr(v, trend[sel]).correlation), 3)
            r["slope vel~bulk"] = round(float((v * trend[sel]).sum() / (trend[sel] ** 2).sum()), 3)
            r["same sign d,vel"] = round(float((np.sign(v) == np.sign(d[sel]))[(v != 0) & (d[sel] != 0)].mean()), 3)
        r["rho(bulk,d)"] = round(float(spearmanr(trend[sel], d[sel]).correlation), 3)
        r["rho(bulk,raw)"] = round(float(spearmanr(trend[sel], raw[sel]).correlation), 3)
        r["slope bulk~d"] = round(float((trend[sel] * d[sel]).sum() / (d[sel] ** 2).sum()), 3)
        r["sd d"] = round(float(d[sel].std()), 3); r["sd raw"] = round(float(raw[sel].std()), 3); r["sd c"] = round(float(c[sel].std()), 3)
        if vel is not None:
            r["sd vel"] = round(float(v.std()), 3)
        if dev is not None:
            r["|dev|^2/|d|^2"] = round(float((dev.astype(float)[sel] ** 2).sum() / (d[sel] ** 2).sum()), 3)
            r["moved by dev"] = round(float((dev[sel] != 0).mean()), 3)
        rows.append(r)
print(pd.DataFrame(rows).T.to_string())
print("bulk trend: sd over expressed genes", round(float(trend[best >= 1].std()), 3), "| nonzero share", round(float((trend[best >= 1] != 0).mean()), 3))
