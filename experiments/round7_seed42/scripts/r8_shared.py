"""What is in the part of the anchor / E10.5 log-ratio that all lineages share (c)?  No score, no target."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from experiments.round5_anchor import load_anchor
from experiments.round5_common import REAL, Mem
from experiments.round6_common import lineage_of
from experiments.round7_method import Anchor7, OWN_SHARE
from experiments.round6_anchor import lineage_means, HARMONISE_LINEAGES
from experiments.bulk_trend import load_trend
from t1_pipeline.data import board_spec
_, genes = board_spec("T1:val")
genes = np.asarray(genes)
ax, _ = load_anchor(list(genes), verbose=False)
anchor = Anchor7(ax)
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", list(genes))
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", list(genes))
lib3 = np.flatnonzero(np.asarray(e10.lib).astype(str) == "3")
ctx = anchor.context(e10, lib3, "min", "type", prev=(e9, np.arange(e9.n_cells)), days_to_anchor=4.0)
trend = np.asarray(load_trend(list(genes))[0], dtype=float)
ms = lineage_means(e10.X[lib3], lineage_of(e10.labels[lib3]), 50)
mp = lineage_means(e9.X, lineage_of(e9.labels), 50)
best = np.max(np.stack(list(ms.values())), axis=0)
sh = -ctx["c"].astype(float)            # shared log(anchor / source)
print("harmonise lineages", HARMONISE_LINEAGES, "| used", ctx["used"])
expr = best >= 1.0
print("genes with a shared part != 0:", int((sh != 0).sum()), "| among expressed (best lineage mean >= 1):", int(((sh != 0) & expr).sum()), "of", int(expr.sum()))
print("share up / down among expressed:", round(float((sh[expr] > 0).mean()), 3), round(float((sh[expr] < 0).mean()), 3), "| sd", round(float(sh[expr].std()), 3), "| median |sh| where != 0", round(float(np.median(np.abs(sh[expr & (sh != 0)]))), 3))
# shared same-laboratory velocity E9.5 -> E10.5: median over lineages of log(E10.5 / E9.5), sign-consistent
ks = [k for k in ("CM", "Endo", "Epi") if k in ms and k in mp]
V = np.stack([np.log((ms[k] + .1) / (mp[k] + .1)) for k in ks])
vsh = np.where((np.sign(V) == np.sign(V[0])).all(axis=0), np.sign(V[0]) * np.abs(V).min(axis=0), 0.0)
sel = expr & (sh != 0)
print("rho(shared ratio, shared velocity) over expressed genes with shared ratio:", round(float(spearmanr(sh[sel], vsh[sel]).correlation), 3),
      "| same sign where both != 0:", round(float((np.sign(sh[sel]) == np.sign(vsh[sel]))[vsh[sel] != 0].mean()), 3), "| velocity != 0 in", round(float((vsh[sel] != 0).mean()), 3))
print("rho(shared ratio, bulk trend):", round(float(spearmanr(sh[sel], trend[sel]).correlation), 3), "| same sign:", round(float((np.sign(sh[sel]) == np.sign(trend[sel])).mean()), 3))
for k in ks:
    v = V[ks.index(k)]
    print(f"  {k}: rho(shared ratio, vel_{k}) = {spearmanr(sh[sel], v[sel]).correlation:.3f}; slope vel~shared = {(v[sel] * sh[sel]).sum() / (sh[sel] ** 2).sum():.3f}")
# level dependence
lv = np.log10(best[sel] + 1e-9)
print("rho(shared ratio, expression level):", round(float(spearmanr(sh[sel], lv).correlation), 3))
def cls(g):
    if g.startswith("mt-"): return "mito"
    if g.startswith(("Rps", "Rpl")): return "ribo"
    if g.startswith(("Hba", "Hbb")): return "globin"
    if g.startswith("Gm") or g.endswith("Rik"): return "unnamed"
    return ""
order = np.argsort(sh)
tab = lambda idx: pd.DataFrame({"gene": genes[idx], "shared": sh[idx].round(2), "bulk": trend[idx].round(2), "vel_sh": vsh[idx].round(2), "best_mean": best[idx].round(1), "cls": [cls(g) for g in genes[idx]]})
e = np.flatnonzero(expr)
oe = e[np.argsort(sh[e])]
print("most DOWN in the anchor, all lineages (expressed genes):"); print(tab(oe[:40]).to_string(index=False))
print("most UP in the anchor, all lineages (expressed genes):"); print(tab(oe[-40:][::-1]).to_string(index=False))
c = pd.Series([cls(g) for g in genes[e]])
for name in ("mito", "ribo", "globin"):
    m = (c == name).to_numpy()
    print(name, "n", int(m.sum()), "mean shared", round(float(sh[e][m].mean()), 3), "mean bulk", round(float(trend[e][m].mean()), 3), "mean vel_sh", round(float(vsh[e][m].mean()), 3))
