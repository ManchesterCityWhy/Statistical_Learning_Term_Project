"""Round 6, step 0: facts about the inputs (no prediction, no scoring).

    python -m experiments.round6_explore

Prints: per-stage / per-library cell counts, depth and cell-type composition; what the E14.5 anchor nuclei are
(marker-based lineage calls and nearest E10.5 cell type), and their depth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.round5_anchor import load_anchor
from experiments.round5_common import REAL, Mem
from experiments.round5_exp4 import depth_of
from t1_pipeline.data import board_spec

MARKERS = {
    "CM": ["Tnnt2", "Myh6", "Myh7", "Ttn", "Ryr2", "Actc1", "Myl7", "Myl4", "Actn2", "Nppa"],
    "Endo": ["Pecam1", "Cdh5", "Npr3", "Emcn", "Kdr", "Flt1", "Tek", "Egfl7"],
    "Fibro/Mes": ["Col1a1", "Col1a2", "Col3a1", "Postn", "Pdgfra", "Tcf21", "Dcn", "Twist1"],
    "Epi": ["Wt1", "Upk3b", "Msln", "Tbx18", "Aldh1a2", "Krt19"],
    "SMC/Peri": ["Myh11", "Acta2", "Pdgfrb", "Rgs5", "Tagln", "Cnn1"],
    "Ery": ["Hba-a1", "Hbb-bs", "Hbb-y", "Hba-x", "Hbb-bh1", "Gypa", "Alas2", "Slc4a1"],
    "Immune": ["Ptprc", "C1qb", "Fcgr1", "Adgre1", "Csf1r", "Lyz2", "Cx3cr1"],
    "Neural/NCC": ["Sox10", "Plp1", "Phox2b", "Foxd3", "Stmn2", "Elavl4"],
    "Hep/Endoderm": ["Alb", "Afp", "Apoa1", "Ttr", "Foxa2", "Epcam"],
    "Epithelial/Ecto": ["Krt8", "Krt18", "Krt5", "Tfap2a", "Cdh1"],
}


def lineage_calls(x, genes):
    """Arg-max of z-scored marker-set means (cells x lineages)."""
    index = pd.Index(genes)
    scores = {}
    for name, ms in MARKERS.items():
        cols = [index.get_loc(g) for g in ms if g in index]
        scores[name] = np.asarray(x[:, cols].mean(axis=1)).ravel()
    frame = pd.DataFrame(scores)
    z = (frame - frame.mean()) / (frame.std() + 1e-9)
    return z.idxmax(axis=1).to_numpy(), frame


def centroid_transfer(ref, labels, query, n_top=60):
    """Nearest reference cell type for every query cell: Pearson correlation on the union of type markers."""
    names = np.unique(labels)
    cent = np.stack([np.asarray(ref[np.flatnonzero(labels == g)].mean(axis=0)).ravel() for g in names])
    overall = np.asarray(ref.mean(axis=0)).ravel()
    genes = np.unique(np.concatenate([np.argsort(-(c - overall))[:n_top] for c in cent]))
    c = cent[:, genes]
    c = (c - c.mean(axis=1, keepdims=True)) / (c.std(axis=1, keepdims=True) + 1e-9)
    out = np.empty(query.shape[0], dtype=object)
    best = np.empty(query.shape[0])
    for s in range(0, query.shape[0], 2048):
        q = query[s:s + 2048][:, genes].toarray()
        q = (q - q.mean(axis=1, keepdims=True)) / (q.std(axis=1, keepdims=True) + 1e-9)
        corr = q @ c.T / len(genes)
        out[s:s + 2048] = names[corr.argmax(axis=1)]
        best[s:s + 2048] = corr.max(axis=1)
    return out.astype(str), best


def main():
    pd.set_option("display.width", 250, "display.max_rows", 200)
    _, genes = board_spec("T1:test")
    stages = {s: Mem(REAL / f"data/T1/{s}_RNA.h5ad", genes) for s in ("E8.5", "E9.5", "E10.5")}
    for name, st in stages.items():
        depth = depth_of(st.X)
        det = st.X.getnnz(axis=1)
        print(f"\n=== {name}: {st.n_cells} cells; libraries {dict(zip(*np.unique(st.lib, return_counts=True)))}")
        print(pd.DataFrame({"lib": st.lib, "umi": depth, "genes": det}).groupby("lib").agg(
            cells=("umi", "size"), umi_q25=("umi", lambda v: np.quantile(v, .25)), umi_med=("umi", "median"),
            umi_q75=("umi", lambda v: np.quantile(v, .75)), genes_med=("genes", "median")).round(0).to_string())
        comp = pd.crosstab(st.labels, st.lib, normalize="columns").mul(100).round(1)
        comp["all"] = pd.Series(st.labels).value_counts(normalize=True).mul(100).round(1)
        comp["umi_med"] = pd.Series(depth).groupby(st.labels).median().round(0)
        lin, _ = lineage_calls(st.X, genes)
        comp["marker_call"] = pd.Series(lin).groupby(st.labels).agg(lambda v: v.value_counts().index[0])
        print(comp.sort_values("all", ascending=False).to_string())
        print("marker-lineage shares by library (%):")
        print(pd.crosstab(lin, st.lib, normalize="columns").mul(100).round(1).to_string())
    anchor, info = load_anchor(genes, verbose=False)
    depth = depth_of(anchor)
    print(f"\n=== E14.5 anchor GSM8394736: {anchor.shape[0]} nuclei; UMI quantiles",
          np.quantile(depth, [.05, .25, .5, .75, .95]).round(0), "genes median", np.median(anchor.getnnz(axis=1)))
    lin, frame = lineage_calls(anchor, genes)
    e10 = stages["E10.5"]
    near, corr = centroid_transfer(e10.X, e10.labels, anchor)
    print("marker-lineage shares (%):")
    print((pd.Series(lin).value_counts(normalize=True) * 100).round(1).to_string())
    tab = pd.crosstab(near, lin)
    tab["share%"] = (tab.sum(axis=1) / len(near) * 100).round(1)
    tab["corr_med"] = pd.Series(corr).groupby(near).median().round(2)
    tab["umi_med"] = pd.Series(depth).groupby(near).median().round(0)
    print("nearest E10.5 type (rows) x marker lineage (columns):")
    print(tab.sort_values("share%", ascending=False).to_string())
    # the same transfer applied to E10.5 itself (sanity: how often does a cell get its own label back?)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(e10.n_cells, 4000, replace=False))
    own, _ = centroid_transfer(e10.X, e10.labels, e10.X[rows])
    print("self-transfer accuracy on 4,000 E10.5 cells:", round(float((own == e10.labels[rows]).mean()), 3))
    lin10, _ = lineage_calls(e10.X, genes)
    print("E10.5 lib3 marker-lineage shares (%):")
    print((pd.Series(lin10[e10.lib == "3"]).value_counts(normalize=True) * 100).round(1).to_string())
    index = pd.Index(genes)
    show = ["Tnnt2", "Myh6", "Myh7", "Nppa", "Col1a1", "Col3a1", "Postn", "Wt1", "Pecam1", "Npr3", "Hbb-y", "Hba-x",
            "Hbb-bh1", "Hbb-bs", "Hba-a1", "C1qb", "Malat1", "mt-Co1", "Xist", "Top2a", "Mki67", "Ttn", "Ryr2", "Gm42418",
            "Rps28", "Rpl13", "Actb", "Tmsb4x"]
    rows3 = np.flatnonzero(e10.lib == "3")
    tbl = pd.DataFrame({"E8.5": stages["E8.5"].mean(), "E9.5": stages["E9.5"].mean(), "E10.5": e10.mean(),
                        "E10.5_lib1": e10.mean(np.flatnonzero(e10.lib == "1")),
                        "E10.5_lib2": e10.mean(np.flatnonzero(e10.lib == "2")), "E10.5_lib3": e10.mean(rows3),
                        "E14.5": np.asarray(anchor.mean(axis=0)).ravel()}, index=genes)
    print(tbl.loc[[g for g in show if g in index]].round(3).to_string())
    print("pseudobulk Spearman between columns (genes with mean > 0.02 anywhere):")
    keep = (tbl > .02).any(axis=1)
    print(tbl[keep].corr(method="spearman").round(3).to_string())
    d = tbl.sub(tbl["E9.5"], axis=0)[keep]
    print("Spearman of changes relative to E9.5:")
    print(d.drop(columns="E9.5").corr(method="spearman").round(3).to_string())


if __name__ == "__main__":
    main()
