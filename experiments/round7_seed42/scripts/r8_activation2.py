"""Same real-data question as r8_activation, for genes a lineage already EXPRESSES at the earlier stage (mean >= 0.3 per 10k):
do those the anchor shows much higher (>= 2x) or much lower (<= 0.5x) move towards the anchor level by the next stage?
Score-free; lineage means in counts per 10k; aggregate share of the way covered."""
import numpy as np, pandas as pd
from experiments.round5_anchor import load_anchor
from experiments.round5_common import REAL, Mem
from experiments.round6_common import lineage_of, centroid_transfer
from experiments.round6_anchor import lineage_means
from t1_pipeline.data import board_spec
_, genes = board_spec("T1:val")
g = np.asarray(genes)
ax, _ = load_anchor(list(genes), verbose=False)
st = {s: Mem(REAL / f"data/T1/{s}_RNA.h5ad", list(genes)) for s in ("E8.5", "E9.5", "E10.5")}
def rows_of(stage, lib):
    m = st[stage]
    return np.arange(m.n_cells) if lib is None else np.flatnonzero(np.asarray(m.lib).astype(str) == lib)
pairs = [("E8.5 -> E9.5", ("E8.5", None), ("E9.5", None)),
         ("E9.5 -> E10.5 lib 3", ("E9.5", None), ("E10.5", "3")), ("E9.5 -> E10.5 lib 1", ("E9.5", None), ("E10.5", "1")), ("E9.5 -> E10.5 lib 2", ("E9.5", None), ("E10.5", "2")),
         ("E9.5 lib 1 -> E9.5 lib 2 (same stage)", ("E9.5", "1"), ("E9.5", "2")), ("E8.5 lib 1 -> E8.5 lib 2 (same stage)", ("E8.5", "1"), ("E8.5", "2")),
         ("E10.5 lib 1 -> E10.5 lib 3 (same stage)", ("E10.5", "1"), ("E10.5", "3"))]
ribo = np.char.startswith(g, "Rpl") | np.char.startswith(g, "Rps")
out, cache = [], {}
for name, (s, sl), (t, tl) in pairs:
    src, tgt = st[s], st[t]
    rs, rt = rows_of(s, sl), rows_of(t, tl)
    if (s, sl) not in cache:
        near = centroid_transfer(src.X[rs], src.labels[rs], ax)
        cache[(s, sl)] = lineage_means(ax, lineage_of(near), 50)
    ma = cache[(s, sl)]
    ms = lineage_means(src.X[rs], lineage_of(src.labels[rs]), 50)
    mt = lineage_means(tgt.X[rt], lineage_of(tgt.labels[rt]), 50)
    best = np.max(np.stack(list(ms.values())), axis=0)
    for k in ("CM", "Endo", "Epi"):
        if k not in ms or k not in mt or k not in ma:
            continue
        expr = (ms[k] >= 0.3) & (ms[k] >= 0.05 * best)
        r = {"pair": name, "lineage": k}
        for nm, sel in (("up 2x", expr & (ma[k] >= 2 * ms[k])), ("up 1.5-2x", expr & (ma[k] >= 1.5 * ms[k]) & (ma[k] < 2 * ms[k])),
                        ("down 0.5x", expr & (ma[k] <= 0.5 * ms[k]) & ~ribo), ("down 0.5x ribo", expr & (ma[k] <= 0.5 * ms[k]) & ribo), ("similar", expr & (ma[k] > 0.8 * ms[k]) & (ma[k] < 1.25 * ms[k]))):
            gap = (ma[k][sel] - ms[k][sel]).sum()
            moved = (mt[k][sel] - ms[k][sel]).sum()
            # log-scale version: mean of log-ratio covered, robust to a few huge genes
            lg = np.log((ma[k][sel] + .1) / (ms[k][sel] + .1)); lm = np.log((mt[k][sel] + .1) / (ms[k][sel] + .1))
            r[nm + " n"] = int(sel.sum())
            r[nm + " frac"] = round(float(moved / gap), 3) if nm != "similar" and abs(gap) > 0 else None
            r[nm + " log frac"] = round(float(lm.sum() / lg.sum()), 3) if nm != "similar" else round(float(lm.mean()), 3)
        out.append(r)
pd.set_option("display.width", 330); pd.set_option("display.max_colwidth", 40)
print(pd.DataFrame(out).to_string(index=False))
