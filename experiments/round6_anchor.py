"""Round 6: what separates the external E14.5 nuclei from the challenge laboratory's cells, lineage by lineage.

The anchor comes from another laboratory. Its nuclei differ from the challenge data in two ways that are not
development: far less cytoplasmic RNA (ribosomal-protein and other abundant mRNAs are several-fold lower, nuclear
transcripts such as Malat1 higher), and no ambient-RNA removal (adult globins in every nucleus). Both show up as a
change that is shared by *every* lineage. `common_shift` estimates that shared part from lineage-matched
comparisons with the source stage; `harmonise` removes a chosen share of it from the anchor nuclei.

Only the source stage (never a proxy target) and the anchor enter these estimates.

    python -m experiments.round6_anchor        # diagnostics on the test-time inputs (E10.5 library 3 vs anchor)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.round6_common import centroid_transfer, lineage_of

HARMONISE_LINEAGES = ("CM", "Endo", "Cushion", "Epi")   # macrophages are too few in the heart library (83 cells)


def lineage_means(x, lineages, min_cells=50):
    """Mean counts-per-10k profile of every lineage with at least `min_cells` cells."""
    out = {}
    for k in np.unique(lineages):
        rows = np.flatnonzero(lineages == k)
        if len(rows) >= min_cells:
            out[k] = np.asarray(x[rows].expm1().mean(axis=0)).ravel()
    return out


def common_shift(source, rows, anchor, eps=0.1, min_lineages=3, mode="min"):
    """Log-ratio source / anchor shared by all matched lineages (0 where lineages disagree in sign).

    Returns (shift, info). `shift[g] > 0` means the anchor under-represents gene g in every lineage.
    The shared part is the smallest absolute lineage log-ratio, signed, so it never exceeds what any single
    lineage shows (mode="min"); mode="median" takes the median absolute lineage log-ratio instead.
    """
    rows = np.asarray(rows)
    labels = source.labels[rows]
    near = centroid_transfer(source.X[rows], labels, anchor)
    lin_anchor, lin_source = lineage_of(near), lineage_of(labels)
    ma, ms = lineage_means(anchor, lin_anchor), lineage_means(source.X[rows], lin_source)
    used = [k for k in HARMONISE_LINEAGES if k in ma and k in ms]
    if len(used) < min_lineages:
        raise ValueError(f"only {used} lineages are present on both sides")
    ratios = np.stack([np.log((ms[k] + eps) / (ma[k] + eps)) for k in used])
    sign = np.sign(np.median(ratios, axis=0))
    agree = (np.sign(ratios) == sign).all(axis=0) & (sign != 0)
    size = np.abs(ratios).min(axis=0) if mode == "min" else np.median(np.abs(ratios), axis=0)
    shift = np.where(agree, sign * size, 0.0)
    info = {"lineages": used, "anchor_lineage_share": {k: float((lin_anchor == k).mean()) for k in np.unique(lin_anchor)},
            "genes_shifted_0.5": int((np.abs(shift) >= .5).sum()), "genes_shifted_1.0": int((np.abs(shift) >= 1).sum())}
    return shift.astype(np.float32), {"info": info, "ratios": dict(zip(used, ratios)), "near": near, "lineage": lin_anchor}


def harmonise(cells, shift, strength=1.0, clip=3.0):
    """Scale counts gene-wise by exp(strength * shift) and renormalise every cell to 10,000 (zeros stay zero)."""
    if strength == 0:
        return cells
    ratio = np.exp(np.clip(shift * strength, -clip, clip)).astype(np.float64)
    out = np.empty_like(cells, dtype=np.float32)
    for s in range(0, len(cells), 256):
        c = np.expm1(cells[s:s + 256].astype(np.float64)) * ratio
        tot = c.sum(axis=1, keepdims=True)
        out[s:s + 256] = np.log1p(c * np.divide(1e4, tot, out=np.ones_like(tot), where=tot > 0))
    return out


def main():
    from experiments.round5_anchor import load_anchor
    from experiments.round6_common import REAL, Mem
    from t1_pipeline.data import board_spec
    pd.set_option("display.width", 250, "display.max_rows", 300)
    _, genes = board_spec("T1:test")
    g = np.array(genes)
    anchor, _ = load_anchor(genes, verbose=False)
    for stage, pool_name in (("E10.5", "lib3"), ("E10.5", "all"), ("E9.5", "all"), ("E8.5", "all")):
        source = Mem(REAL / f"data/T1/{stage}_RNA.h5ad", genes)
        rows = np.flatnonzero(source.lib == "3") if pool_name == "lib3" else np.arange(source.n_cells)
        shift, ctx = common_shift(source, rows, anchor)
        print(f"\n=== source {stage} ({pool_name}): {ctx['info']}")
        fam = pd.Series(np.select([np.char.startswith(g, "Rpl") | np.char.startswith(g, "Rps"), np.char.startswith(g, "mt-"),
                                   np.char.startswith(g, "Hb") , np.char.startswith(g, "Gm") | np.char.endswith(g, "Rik")],
                                  ["ribosomal", "mito", "globin", "unnamed"], "other"))
        tab = pd.DataFrame({"gene": g, "family": fam, "shift": shift, **{f"r_{k}": v for k, v in ctx["ratios"].items()}})
        print("genes by family with |shift| >= 0.5:"); print(tab[tab["shift"].abs() >= .5].groupby("family")["shift"].agg(["size", "mean"]).round(2).to_string())
        print("anchor LOWER in every lineage (top 40):"); print(tab.sort_values("shift", ascending=False).head(40).round(2).to_string(index=False))
        print("anchor HIGHER in every lineage (top 40):"); print(tab.sort_values("shift").head(40).round(2).to_string(index=False))
        ratios = np.stack(list(ctx["ratios"].values()))
        print("correlation of lineage log-ratios across genes with mean ratio magnitude > 0.2:")
        keep = np.abs(ratios).mean(axis=0) > .2
        print(pd.DataFrame(np.corrcoef(ratios[:, keep]), index=ctx["info"]["lineages"], columns=ctx["info"]["lineages"]).round(2).to_string())


if __name__ == "__main__":
    main()
