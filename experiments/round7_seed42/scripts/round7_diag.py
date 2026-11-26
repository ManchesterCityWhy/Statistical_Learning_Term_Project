"""Round 7 diagnostics on the test-time inputs: what the harmonisation removes and where each lineage is moved."""
import numpy as np, pandas as pd
from experiments.round5_anchor import load_anchor
from experiments.round6_common import REAL, Mem, depth_of, lineage_of
from experiments.round7_method import Anchor7
from experiments.bulk_trend import load_trend
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250, "display.max_rows", 200)
_, genes = board_spec("T1:test")
g = np.array(genes)
import os
ax, ainfo = load_anchor(genes, verbose=False, desoup=bool(int(os.environ.get("DESOUP", "0"))))
print("anchor:", {k: ainfo[k] for k in ("cells_after_qc", "median_umi", "desoup", "soup_fraction")})
from experiments.round7_method import soup_profile
soup, sinfo = (soup_profile(genes) if int(os.environ.get("SOUP", "0")) else (None, None))
print("soup:", sinfo, "" if soup is None else ("top genes: " + ", ".join(f"{g[i]} {soup[i]*1e4:.0f}" for i in np.argsort(-soup)[:40])))
anchor = Anchor7(ax, soup)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
lib3 = np.flatnonzero(e10.lib == "3")
ctx = anchor.context(e10, lib3)
print("ambient fraction per lineage:", {k: round(v, 3) for k, v in ctx["rho"].items()}, "genes used", ctx["rho_genes"])
print("matched lineages", ctx["used"], "| anchor lineage shares", {k: round(v, 3) for k, v in ctx["anchor_lineage_share"].items()})
ks, table = anchor.match_k(e10, lib3, ctx)
print("pooling size per lineage (match_k)", table)
c = ctx["c"]
print("shared shift c: genes != 0:", int((c != 0).sum()), " |c|>=0.5:", int((np.abs(c) >= .5).sum()), " |c|>=1:", int((np.abs(c) >= 1).sum()))
trend, _ = load_trend(genes)
ok = (c != 0) & np.isfinite(trend) & (trend != 0)
print("bulk E9.5->E14.5 trend available for", int((trend != 0).sum()), "genes; corr(-c, trend) over genes with both:", round(float(np.corrcoef(-c[ok], trend[ok])[0, 1]), 3), "n", int(ok.sum()))
markers = ["Tnnt2", "Myl2", "Myl3", "Myl7", "Myl4", "Myh6", "Myh7", "Nppa", "Nppb", "Ttn", "Ryr2", "Actc1", "Hcn4", "Top2a", "Mki67", "Hba-x", "Hbb-y", "Hbb-bh1", "Hba-a1", "Hbb-bs",
           "Col1a1", "Col3a1", "Postn", "Wt1", "Tbx18", "Upk3b", "Tcf21", "Pecam1", "Npr3", "Cdh5", "Fabp4", "Cd36", "Sox9", "Twist1", "Malat1", "Xist", "Rps2", "Rpl4", "mt-Co1"]
idx = pd.Index(genes).get_indexer(markers)
tab = pd.DataFrame({"c(shared)": c[idx], "bulk_trend": trend[idx], **{k: v[idx] for k, v in ctx["d"].items()}}, index=markers)
print("marker genes: shared shift c (source/anchor, removed from the anchor), bulk trend, lineage log-ratios d (anchor'/source):")
print(tab.round(2).to_string())
for k, v in ctx["d"].items():
    print(f"\n{k}: |d|>=0.5: {int((np.abs(v) >= .5).sum())}, |d|>=1: {int((np.abs(v) >= 1).sum())}; corr with bulk trend {np.corrcoef(v[trend != 0], trend[trend != 0])[0, 1]:.3f}")
    print("  up  :", ", ".join(f"{g[i]} {v[i]:.1f}" for i in np.argsort(-v)[:30]))
    print("  down:", ", ".join(f"{g[i]} {v[i]:.1f}" for i in np.argsort(v)[:30]))
dvals = np.stack(list(ctx["d"].values()))
print("\ncorrelation between lineage log-ratios (genes with mean |d| > 0.3):")
keep = np.abs(dvals).mean(axis=0) > .3
print(pd.DataFrame(np.corrcoef(dvals[:, keep]), index=list(ctx["d"]), columns=list(ctx["d"])).round(2).to_string())
