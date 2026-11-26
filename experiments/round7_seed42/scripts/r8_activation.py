"""Do the genes a lineage barely expresses at an earlier stage, but the E14.5 anchor does, move towards the anchor level
by the next released stage?  Score-free, lineage means in counts per 10k (linear).

For source stage S, next stage T and lineage L: genes with mean_S < 0.3, expressed by the lineage itself in the sense of the
guard (mean_S >= 5% of the largest lineage mean at S), and anchor mean >= 2 x max(mean_S, 0.02).
    fraction = sum(mean_T - mean_S) / sum(anchor - mean_S)       (aggregate share of the way covered)
Control: genes with mean_S < 0.3 that the anchor does not show either (anchor < 2 x max(mean_S, 0.02)): mean change."""
import numpy as np, pandas as pd
from experiments.round5_anchor import load_anchor
from experiments.round5_common import REAL, Mem
from experiments.round6_common import lineage_of, centroid_transfer
from experiments.round6_anchor import lineage_means
from t1_pipeline.data import board_spec
_, genes = board_spec("T1:val")
ax, _ = load_anchor(list(genes), verbose=False)
st = {s: Mem(REAL / f"data/T1/{s}_RNA.h5ad", list(genes)) for s in ("E8.5", "E9.5", "E10.5")}
def rows_of(stage, lib):
    m = st[stage]
    return np.arange(m.n_cells) if lib is None else np.flatnonzero(np.asarray(m.lib).astype(str) == lib)
pairs = [("E8.5 -> E9.5", ("E8.5", None), ("E9.5", None), 6.0), ("E8.5 -> E9.5 lib 1", ("E8.5", None), ("E9.5", "1"), 6.0), ("E8.5 -> E9.5 lib 2", ("E8.5", None), ("E9.5", "2"), 6.0),
         ("E9.5 -> E10.5 lib 3", ("E9.5", None), ("E10.5", "3"), 5.0), ("E9.5 -> E10.5 lib 1", ("E9.5", None), ("E10.5", "1"), 5.0), ("E9.5 -> E10.5 lib 2", ("E9.5", None), ("E10.5", "2"), 5.0),
         ("E9.5 lib 1 -> E9.5 lib 2 (same stage)", ("E9.5", "1"), ("E9.5", "2"), None), ("E8.5 lib 1 -> E8.5 lib 2 (same stage)", ("E8.5", "1"), ("E8.5", "2"), None),
         ("E10.5 lib 1 -> E10.5 lib 3 (same stage)", ("E10.5", "1"), ("E10.5", "3"), None)]
out = []
cache = {}
for name, (s, sl), (t, tl), days in pairs:
    src, tgt = st[s], st[t]
    rs, rt = rows_of(s, sl), rows_of(t, tl)
    key = (s, sl)
    if key not in cache:      # anchor nuclei assigned to the source stage's cell types, then lineages
        near = centroid_transfer(src.X[rs], src.labels[rs], ax)
        cache[key] = lineage_means(ax, lineage_of(near), 50)
    ma = cache[key]
    ms = lineage_means(src.X[rs], lineage_of(src.labels[rs]), 50)
    mt = lineage_means(tgt.X[rt], lineage_of(tgt.labels[rt]), 50)
    best = np.max(np.stack(list(ms.values())), axis=0)
    for k in ("CM", "Endo", "Epi"):
        if k not in ms or k not in mt or k not in ma:
            continue
        low = (ms[k] < 0.3) & (ms[k] >= 0.05 * best)
        new = low & (ma[k] >= 2 * np.maximum(ms[k], 0.02))
        ctl = low & ~new
        gap = (ma[k][new] - ms[k][new]).sum()
        moved = (mt[k][new] - ms[k][new]).sum()
        per_gene = (mt[k][new] - ms[k][new]) / (ma[k][new] - ms[k][new])
        strong = new & (ma[k] >= 1.0)
        out.append({"pair": name, "lineage": k, "new genes": int(new.sum()), "fraction of the way (aggregate)": round(float(moved / gap), 3),
                    "median per gene": round(float(np.median(per_gene)), 3), "share of genes that rose": round(float((mt[k][new] > ms[k][new]).mean()), 3),
                    "anchor >= 1: n": int(strong.sum()), "anchor >= 1: fraction": round(float((mt[k][strong] - ms[k][strong]).sum() / (ma[k][strong] - ms[k][strong]).sum()), 3),
                    "control genes": int(ctl.sum()), "control: mean change": round(float((mt[k][ctl] - ms[k][ctl]).mean()), 4),
                    "control: mean level": round(float(ms[k][ctl].mean()), 4), "new: mean source / target / anchor": f"{ms[k][new].mean():.3f} / {mt[k][new].mean():.3f} / {ma[k][new].mean():.3f}",
                    "constant-rate share": None if days is None else round(1.0 / days, 3)})
pd.set_option("display.width", 320); pd.set_option("display.max_colwidth", 44)
print(pd.DataFrame(out).to_string(index=False))
