"""Score-free look at the genes a cell type barely expresses at E10.5 but the E14.5 anchor does ("new genes"),
and at the size of the official DE sets for library 3 against the test reference."""
import numpy as np, pandas as pd
import common.core_metrics as cm
from experiments.round5_anchor import load_anchor
from experiments.round5_common import REAL, Mem, split_rows
from experiments.round6_common import lineage_of
from experiments.round7_method import Anchor7
from experiments.round6_anchor import lineage_means
from experiments.bulk_trend import load_trend
from t1_pipeline.data import board_spec
_, genes = board_spec("T1:val")
g = np.asarray(genes)
print("MIN_LFC", cm.MIN_LFC)
ax, _ = load_anchor(list(genes), verbose=False)
anchor = Anchor7(ax)
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", list(genes))
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", list(genes))
lib3 = np.flatnonzero(np.asarray(e10.lib).astype(str) == "3")
ctx = anchor.context(e10, lib3, "min", "type")
trend = np.asarray(load_trend(list(genes))[0], dtype=float)
ms = lineage_means(e10.X[lib3], lineage_of(e10.labels[lib3]), 50)
mp = lineage_means(e9.X, lineage_of(e9.labels), 50)
ma = lineage_means(anchor.x, ctx["lineage"], 50)
best = np.max(np.stack(list(ms.values())), axis=0)
rows = []
for k in ("CM", "Endo", "Epi", "Cushion"):
    own = ms[k] >= 0.05 * best
    for low in (0.1, 0.3, 1.0):
        new = (ms[k] < low) & own & (ma[k] >= 2 * np.maximum(ms[k], 0.02))
        share_anchor = ma[k][(ms[k] < low) & own].sum() / 1e4
        r = {"lineage": k, "low<": low, "genes below": int(((ms[k] < low) & own).sum()), "share of anchor UMIs in them": round(float(share_anchor), 4),
             "share of E10.5 UMIs in them": round(float(ms[k][(ms[k] < low) & own].sum() / 1e4), 4),
             "new genes (anchor >= 2x)": int(new.sum()), "of them bulk up": round(float((trend[new] > 0).mean()), 3), "bulk down": round(float((trend[new] < 0).mean()), 3),
             "bulk zero": round(float((trend[new] == 0).mean()), 3)}
        if k in mp:
            up = mp[k][new] < ms[k][new]; dn = mp[k][new] > ms[k][new]
            r["E9.5 -> E10.5 up"] = round(float(up.mean()), 3); r["down"] = round(float(dn.mean()), 3)
            r["mean E9.5 / E10.5 / anchor"] = f"{mp[k][new].mean():.3f} / {ms[k][new].mean():.3f} / {ma[k][new].mean():.3f}"
        rows.append(r)
pd.set_option("display.width", 280)
print(pd.DataFrame(rows).to_string(index=False))
k = "CM"
own = ms[k] >= 0.05 * best
new = np.flatnonzero((ms[k] < 0.3) & own & (ma[k] >= 2 * np.maximum(ms[k], 0.02)))
top = new[np.argsort(-ma[k][new])[:40]]
print("CM, largest new genes:", [(g[i], round(float(ms[k][i]), 2), round(float(ma[k][i]), 2), round(float(trend[i]), 2)) for i in top])
# official DE sets: library 3 (638-cell half) against the test reference
ref = e9.dense(split_rows(e9, e10)["reference"])
rng = np.random.default_rng(1)
t = e10.dense(np.sort(rng.choice(lib3, 638, replace=False)))
up, dn, lfc = cm.de_genes(t, ref)
pbr = ref.mean(axis=0)
print("library 3 vs reference: n_up", len(up), "n_dn", len(dn), "| reference mean log1p of DE genes: median", round(float(np.median(pbr[np.concatenate([up, dn])])), 3),
      "| genes with reference or target mean log1p >= 0.05:", int(((pbr >= .05) | (t.mean(axis=0) >= .05)).sum()), "| >= 0.01:", int(((pbr >= .01) | (t.mean(axis=0) >= .01)).sum()))
print("quantiles of |lfc| over all genes:", np.round(np.quantile(np.abs(lfc), [.5, .75, .9, .95, .99]), 4))
