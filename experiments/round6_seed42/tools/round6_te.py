"""Round 6, proxy TE: the real test inputs scored against a real, specifically dissected later heart.

    reference = E9.5 (the 1,706 rows of the round-2 split)      <- the reference stage of the test board
    source    = E10.5, all three libraries                      <- the source of the test prediction
    target    = E14.5 wild-type heart nuclei (GSE272154 / GSM8394736), disjoint halves of 1,000

What it imitates: the reference two stages back, the three-library source, a target that is a cleanly dissected
heart several days later. Blind spots: four days instead of two, a target from another laboratory (its own
nuclei preparation and depth), and no independent later anchor, so only *base* populations can be compared
fairly; rows with `f>0` mix in nuclei of the same E14.5 sample (disjoint from the target) and are only
meaningful as "same anchor, different base" comparisons.

    te/raw : E9.5 and E10.5 as released          te/dm : both thinned to the depth distribution of the target

    python -m experiments.round6_te [--split confirmation]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import split_rows
from experiments.round5_exp4 import r4_cells
from experiments.round5_exp6 import mix
from experiments.round6_common import (HEART, REAL, SEED, FastEval, Log, Mem, anchor_split, depth_of, lineage_of,
                                       take, thinned)
from t1_pipeline.data import board_spec

N = 2000


def bases(source, trend):
    """name -> (cells, donor rows or None). Only the source stage and the bulk trend are used."""
    rng = lambda: np.random.default_rng(SEED)
    everything = np.arange(source.n_cells)
    lib3 = np.flatnonzero(source.lib == "3")
    lib12 = np.flatnonzero(source.lib != "3")
    heart = np.flatnonzero(np.isin(source.labels, list(HEART)))
    heart3 = np.intersect1d(heart, lib3)
    ref10 = source.dense(np.sort(rng().choice(source.n_cells, N, replace=False)))
    for name, pool in (("all", everything), ("lib12", lib12), ("lib3", lib3), ("heart_all", heart),
                       ("heart_lib3", heart3)):
        rows = take(source, pool, N, rng())
        yield f"natural:{name}", source.dense(rows), rows
    for name, pool in (("all", everything), ("lib3", lib3), ("heart_all", heart)):
        r4, r2 = r4_cells(source, pool, ref10, trend, N)
        yield f"r2:{name}", r2, None
        yield f"r4:{name}", r4, None
    rows = take(source, lib3, N, rng())
    for s in (3.0, 6.0):
        yield f"natural:lib3+bulk{s:g}", apply_trend(source.dense(rows), trend, s / 5, False, SEED), rows


def library_control(source):
    """Heart-lineage cells with the lineage quotas of library 3, taken from library 3 or from libraries 1+2."""
    from experiments.round6_method import draw_base, lineage_shares
    heart = np.isin(source.labels, list(HEART))
    lib3 = np.flatnonzero(heart & (source.lib == "3"))
    lib12 = np.flatnonzero(heart & (source.lib != "3"))
    quotas = lineage_shares(source, lib3)
    for name, pool in (("heart-quota:lib3", lib3), ("heart-quota:lib12", lib12)):
        rows = draw_base(source, pool, N, np.random.default_rng(SEED), quotas)
        yield name, source.dense(rows), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--library-control", action="store_true",
                        help="only two bases: heart-lineage cells with IDENTICAL lineage quotas (those of library 3), drawn "
                             "from library 3 or from libraries 1+2. Separates 'which library' from 'which cell types'.")
    parser.add_argument("--harm-target", action="store_true",
                        help="remove the lineage-shared anchor/E10.5 shift (estimated on ALL E10.5 cells) from the E14.5 target: "
                             "a target that looks as if the challenge laboratory had prepared it")
    args = parser.parse_args()
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    anchor, _ = load_anchor(genes, verbose=False)
    t0 = time.time()
    e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    ref_rows = split_rows(e9, e10)["reference"]
    asp = anchor_split(anchor.shape[0])
    target, ceiling = anchor[asp[a]].toarray(), anchor[asp[b]].toarray()
    pool = anchor[asp["pool"]]
    depth = depth_of(anchor)
    sources = {"te/raw": (e9, e10), "te/dm": (thinned(e9, depth), thinned(e10, depth))}
    tag = ""
    if args.harm_target:
        from scipy import sparse
        from experiments.round6_anchor import common_shift, harmonise
        shift = common_shift(e10, np.arange(e10.n_cells), anchor)[0]
        target, ceiling = harmonise(target, shift), harmonise(ceiling, shift)
        pool = sparse.csr_matrix(harmonise(pool.toarray(), shift))
        sources = {"te-harm/raw": (e9, e10)}
        tag = "_harmtarget"
    if args.library_control:
        tag += "_librarycontrol"
    log = Log(REAL / "runs" / "round6_te" / f"scores_{args.split}{tag}.csv")
    lin = lineage_of(e10.labels)
    for proxy, (ref_stage, source) in sources.items():
        ev = FastEval(ref_stage.dense(ref_rows), target, ceiling)
        print(proxy, "ready", f"{time.time() - t0:.0f}s", "floor", ev.floor, "ceiling", ev.ceiling,
              "n_up", len(ev.up), "n_dn", len(ev.dn), flush=True)
        for name, cells, rows in (library_control(source) if args.library_control else bases(source, trend)):
            extra = {}
            if rows is not None:
                extra = {f"share_{k}": round(float((lin[rows] == k).mean()), 3) for k in ("CM", "Endo", "Cushion", "Epi", "Blood", "Mac", "Other")}
                extra["lib3_share"] = round(float((source.lib[rows] == "3").mean()), 3)
            for f in (0.0, 0.375):
                pred = mix(cells, pool, f, np.random.default_rng(SEED))
                r = log.add(ev, pred, proxy=proxy, base=name, f=f, split=args.split)
                r.update(extra)
        pd.DataFrame(log.records).to_csv(log.path, index=False)
    frame = pd.DataFrame(log.records)
    print(frame.pivot_table(index="base", columns=["proxy", "f"], values="score", sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
