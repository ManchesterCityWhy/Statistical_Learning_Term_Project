"""External future anchor: wild-type E14.5 mouse heart, 10x Multiome nuclei (GEO GSE272154, sample GSM8394736).

Same assay and pipeline as the competition data (10x Multiome GEX on nuclei, Cell Ranger ARC 2.0.2, mm10).
E14.5 lies outside the excluded window (after E9.5 up to and including E13.5). Only the wild-type sample is
read; the three mutant samples of the series are never downloaded.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from t1_pipeline.data import ROOT, file_hash

DIR = ROOT / "data" / "external" / "GSE272154"
H5 = DIR / "GSM8394736_Wildtype_raw_feature_bc_matrix.h5"
METRICS = DIR / "GSM8394736_Wildtype_per_barcode_metrics.csv.gz"


def load_anchor(genes, min_umi=1000, min_genes=500, max_mito=.25, verbose=True, desoup=False, seed=42):
    """Return (CSR log1p(1e4 * counts / total) in official gene order, info dict)."""
    metrics = pd.read_csv(METRICS)
    called = set(metrics.loc[metrics["is_cell"] == 1, "barcode"] if "is_cell" in metrics else metrics["barcode"])
    with h5py.File(H5, "r") as f:
        m = f["matrix"]
        barcodes = m["barcodes"][:].astype(str)
        names = m["features/name"][:].astype(str)
        kinds = m["features/feature_type"][:].astype(str)
        shape = tuple(m["shape"][:])
        x = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape).T.tocsr()
    cells = np.flatnonzero(np.isin(barcodes, list(called)))
    gex = np.flatnonzero(kinds == "Gene Expression")
    rho = None
    if desoup:
        # Ambient-RNA removal in the spirit of SoupX (the competition data were SoupX-cleaned): the soup
        # profile comes from uncalled droplets with 10-100 UMIs, the contamination fraction from haemoglobin
        # transcripts, which in nuclei of a heart at E14.5 are almost entirely ambient.
        full = x[:, gex]
        tot_all = np.asarray(full.sum(axis=1)).ravel()
        empty = np.setdiff1d(np.flatnonzero((tot_all >= 10) & (tot_all <= 100)), cells)
        soup = np.asarray(full[empty].sum(axis=0)).ravel().astype(np.float64)
        soup /= soup.sum()
        hb = np.array([n.startswith(("Hba-", "Hbb-")) for n in names[gex]])
        sub = full[cells].astype(np.float64)
        tot = np.asarray(sub.sum(axis=1)).ravel()
        per_cell = np.asarray(sub[:, hb].sum(axis=1)).ravel() / np.maximum(tot, 1) / soup[hb].sum()
        rho = float(np.clip(np.median(per_cell), 0, .8))
        dense_rng = np.random.default_rng(seed)
        blocks = []
        for s0 in range(0, len(cells), 512):
            d = sub[s0:s0 + 512].toarray()
            d = np.maximum(d - rho * tot[s0:s0 + 512, None] * soup[None, :], 0)
            low = np.floor(d)
            d = low + (dense_rng.random(d.shape) < (d - low))
            blocks.append(sparse.csr_matrix(d.astype(np.float32)))
        x = sparse.vstack(blocks).tocsr()
    else:
        x = x[cells][:, gex].astype(np.float32)
    names = names[gex]
    index = pd.Index(genes).get_indexer(names)
    keep = np.flatnonzero(index >= 0)
    proj = sparse.csr_matrix((np.ones(len(keep), np.float32), (keep, index[keep])), shape=(len(names), len(genes)))
    counts = (x @ proj).tocsr()
    total = np.asarray(counts.sum(axis=1)).ravel()
    detected = counts.getnnz(axis=1)
    mito = np.array([g.lower().startswith("mt-") for g in genes])
    mito_frac = np.asarray(counts[:, mito].sum(axis=1)).ravel() / np.maximum(total, 1)
    ok = (total >= min_umi) & (detected >= min_genes) & (mito_frac <= max_mito)
    counts, total = counts[ok], total[ok]
    norm = counts.multiply((1e4 / total)[:, None]).tocsr()
    norm.data = np.log1p(norm.data).astype(np.float32)
    info = {"series": "GSE272154", "sample": "GSM8394736 (RNA Wild Type e14.5 rep1)", "stage": "E14.5",
            "assay": "10x Multiome nuclei, Cell Ranger ARC 2.0.2, mm10", "called_barcodes": int(len(cells)),
            "cells_after_qc": int(ok.sum()), "genes_matched": int(len(np.unique(index[keep]))),
            "genes_in_file": int(len(names)), "median_umi": float(np.median(total)),
            "median_genes": float(np.median(detected[ok])), "desoup": desoup, "soup_fraction": rho, "qc": {"min_umi": min_umi, "min_genes": min_genes, "max_mito": max_mito},
            "sha256": {H5.name: file_hash(H5), METRICS.name: file_hash(METRICS)}}
    if verbose:
        print(info, flush=True)
    return norm.astype(np.float32), info


if __name__ == "__main__":
    from t1_pipeline.data import board_spec
    _, genes = board_spec("T1:test")
    x, info = load_anchor(genes)
    mean = np.asarray(x.mean(axis=0)).ravel()
    det = np.asarray((x > 0).mean(axis=0)).ravel()
    g = pd.Index(genes)
    for name in ("Tnnt2", "Myl2", "Myl7", "Nppa", "Col1a1", "Col3a1", "Postn", "Wt1", "Pecam1", "Npr3", "Hbb-y", "Hba-x", "Hbb-bs",
                 "Hba-a1", "C1qb", "Malat1", "mt-Co1", "Xist", "Top2a", "Ttn", "Ryr2"):
        print(name, round(mean[g.get_loc(name)], 3), round(det[g.get_loc(name)], 3))
    total = np.asarray(x.expm1().sum(axis=1)).ravel()
    print("libsum check", total.min(), total.max(), "detected q", np.quantile(x.getnnz(axis=1), [.05, .25, .5, .75, .95]))
