"""Round 6, proxy D2: what each hedge costs if nothing had changed but the dissection (and perhaps the depth).

    reference = E9.5 (the 1,706 rows of the round-2 split)          <- as on the test board
    source    = E10.5, with the target cells removed
    target    = E10.5 library 3 (heart-proper), three depth scenarios:
                deep     as released (median ~10k UMI, like every E8.5/E9.5 library)
                mid      thinned to the depth distribution of E10.5 library 2 (~6.4k)
                shallow  thinned to the depth distribution of E10.5 library 1 (~3.8k)

A prediction made of other library-3 cells is exact for the deep scenario, so every departure from it (shallow
cells of libraries 1+2, anchor nuclei, the round-5 base) shows up as a cost; the same departures pay off in
the shallow scenarios. This prices the depth hedge and the anchor with the real test source, in the test's own
reference frame. Blind spot: zero elapsed time, so the anchor can only cost here; what it gains when time has
passed comes from the other proxies.

    python -m experiments.round6_d2 [--split confirmation]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import h3_rows, split_rows
from experiments.round5_exp4 import r4_cells
from experiments.round6_common import (HEART, REAL, SEED, FastEval, Log, Mem, centroid_transfer, depth_of, lineage_of,
                                       thin_rows)
from experiments.round6_method import Anchor, lineage_shares, predict
from experiments.round6_s1 import with_anchor
from t1_pipeline.data import board_spec

N = 2000


def deep_rows(depth, lineage, top=0.3, drop=0.02):
    """Anchor rows in the top `top` share by UMI count within their lineage, without the very top `drop` (doublets)."""
    keep = np.zeros(len(depth), bool)
    for k in np.unique(lineage):
        rows = np.flatnonzero(lineage == k)
        lo, hi = np.quantile(depth[rows], [1 - top, 1 - drop])
        keep[rows[(depth[rows] >= lo) & (depth[rows] <= hi)]] = True
    return np.flatnonzero(keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--extra", action="store_true", help="second grid: harmonisation mode, bulk trend, small depth hedge")
    args = parser.parse_args()
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    ax, _ = load_anchor(genes, verbose=False)
    anchor = Anchor(ax)
    t0 = time.time()
    e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    ref = e9.dense(split_rows(e9, e10)["reference"])
    h3 = h3_rows(e10)
    used = np.concatenate([h3[k] for k in h3])
    lib3 = np.setdiff1d(np.flatnonzero(e10.lib == "3"), used)          # library-3 cells no target half contains
    lib12 = np.flatnonzero(e10.lib != "3")
    rest = np.setdiff1d(np.arange(e10.n_cells), used)
    rng = lambda: np.random.default_rng(SEED)
    d1, d2 = depth_of(e10.X[np.flatnonzero(e10.lib == "1")]), depth_of(e10.X[np.flatnonzero(e10.lib == "2")])
    scen = {"deep": (e10.X[h3[a]], e10.X[h3[b]]),
            "mid": (thin_rows(e10.X[h3[a]], d2, rng()), thin_rows(e10.X[h3[b]], d2, rng())),
            "shallow": (thin_rows(e10.X[h3[a]], d1, rng()), thin_rows(e10.X[h3[b]], d1, rng()))}
    evs = {k: FastEval(ref, t.toarray(), c.toarray()) for k, (t, c) in scen.items()}
    for k, ev in evs.items():
        print(k, "floor", ev.floor, "ceiling", ev.ceiling, f"{time.time() - t0:.0f}s", flush=True)
    shares = lineage_shares(e10, lib3)
    near = centroid_transfer(e10.X[lib3], e10.labels[lib3], ax)
    lin_a, depth_a = lineage_of(near), depth_of(ax)
    deep = deep_rows(depth_a, lin_a)
    print("anchor deep subset:", len(deep), "nuclei, median UMI", float(np.median(depth_a[deep])), "vs all", float(np.median(depth_a)),
          "| lineage shares deep", {k: round(float((lin_a[deep] == k).mean()), 3) for k in np.unique(lin_a)},
          "all", {k: round(float((lin_a == k).mean()), 3) for k in np.unique(lin_a)}, flush=True)
    log = Log(REAL / "runs" / "round6_d2" / f"scores_{args.split}.csv")

    def score(cells, **meta):
        for k, ev in evs.items():
            log.add(ev, cells, scenario=k, split=args.split, **meta)

    kw = dict(seed=SEED, shares=shares, shallow_pool=lib12, shift_key="E10.5/lib3-rest")
    if args.extra:
        log = Log(REAL / "runs" / "round6_d2" / f"scores_{args.split}_extra.csv")
        for f in (0.0, 0.15, 0.2, 0.25, 0.3, 0.375):
            for mode, t, s in [("min", t, 0.0) for t in (0.0, 1.5, 3.0, 6.0)] + ([("median", 0.0, 0.0), ("min", 0.0, 0.1), ("min", 3.0, 0.1)] if f > 0 else [("min", 0.0, 0.1)]):
                score(predict(e10, lib3, anchor, N, f, shallow_share=s, trend=trend, trend_strength=t, harmonise_strength=1.0, harmonise_mode=mode, **kw),
                      kind="extra", anchor=f"harm-{mode}", shallow=s, f=f, trend=t)
        frame = pd.DataFrame(log.records)
        pd.set_option("display.width", 250, "display.max_rows", 300)
        t = frame.pivot_table(index=["anchor", "trend", "shallow", "f"], columns="scenario", values="score", sort=False)[["deep", "mid", "shallow"]]
        t["E(.6,.2,.2)"] = .6 * t["deep"] + .2 * t["mid"] + .2 * t["shallow"]
        print(t.round(2).to_string())
        return
    for s in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        score(predict(e10, lib3, anchor, N, 0.0, shallow_share=s, **kw), kind="depth-hedge", shallow=s, f=0.0, anchor="-")
    for variant, rows, h in (("natural", None, 0.0), ("deep", deep, 0.0), ("harm", None, 1.0), ("deep+harm", deep, 1.0)):
        for f in (0.1, 0.2, 0.3, 0.375, 0.45):
            score(predict(e10, lib3, anchor, N, f, anchor_rows=rows, harmonise_strength=h, **kw), kind="anchor", shallow=0.0, f=f, anchor=variant)
    for s, f in ((0.2, 0.2), (0.2, 0.3), (0.3, 0.2), (0.3, 0.3), (0.2, 0.375)):
        for variant, rows in (("natural", None), ("deep", deep)):
            score(predict(e10, lib3, anchor, N, f, anchor_rows=rows, shallow_share=s, **kw), kind="both", shallow=s, f=f, anchor=variant)
    # other bases of interest, same anchor share as the round-5 file
    ref10 = e10.dense(np.sort(rng().choice(rest, N, replace=False)))
    r4, r2 = r4_cells(e10, rest, ref10, trend, N)
    for f in (0.0, 0.375):
        score(with_anchor(r4, anchor, f, SEED), kind="round5-base", shallow=-1, f=f, anchor="natural")
        share3 = float((e10.lib == "3").mean())                      # natural library shares of the released stage
        score(predict(e10, [(lib3, share3), (lib12, 1 - share3)], anchor, N, f, seed=SEED), kind="all-libraries-natural", shallow=-1, f=f, anchor="natural")
        heart = np.isin(e10.labels, list(HEART))
        h3share = float((heart & (e10.lib == "3")).sum() / heart.sum())
        score(predict(e10, [(lib3[heart[lib3]], h3share), (lib12[heart[lib12]], 1 - h3share)], anchor, N, f, seed=SEED),
              kind="heart-all-libraries-natural", shallow=-1, f=f, anchor="natural")
    frame = pd.DataFrame(log.records)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(frame.pivot_table(index=["kind", "anchor", "shallow", "f"], columns="scenario", values="score", sort=False, dropna=False).round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(frame.pivot_table(index=["kind", "anchor", "shallow", "f"], columns="scenario", values=m, sort=False, dropna=False).round(3).to_string())


if __name__ == "__main__":
    main()
