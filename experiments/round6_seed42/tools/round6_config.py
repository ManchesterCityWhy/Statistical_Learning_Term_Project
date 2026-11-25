"""Round 6: named configurations and the one function that turns a configuration into cells.

`build(cfg, source, anchor, trend, n_total)` is called by the proxy evaluation (with a proxy's source stage)
and by the export (with E10.5). Nothing else constructs a round-6 prediction.

A configuration:
    base           "heart-library"  for a heart target: the stage's heart-proper libraries (>= 90% heart-lineage
                                    cells) in their natural composition; a stage without one falls back to its
                                    heart-labelled cells; for a target with the source's dissection: all cells
                   "all"            all cells of the pool, natural composition
    shallow_share  share of every lineage of the base that is drawn from the stage's other (surrounding-tissue,
                   shallow) libraries instead; 0 when the stage has no heart-proper library
    f              anchor share of the population; `f_per_elapsed` x elapsed share when given instead
    anchor         "natural" | "deep"  (top 30% of nuclei by UMI within each lineage, top 2% removed)
    harmonise      share of the lineage-shared anchor/source shift removed from the anchor nuclei
    trend          strength of the within-cell bulk trend applied to the base cells (round-2 scale: 3 = 3/5)
"""
from __future__ import annotations

import numpy as np

from experiments.round6_common import HEART, SEED, centroid_transfer, depth_of, lineage_of
from experiments.round6_method import lineage_shares, predict

HEART_LIBRARY_MIN = 0.90


def heart_libraries(source, pool):
    """Libraries of `pool` in which at least 90% of the cells carry a heart-lineage label."""
    pool = np.asarray(pool)
    is_heart = np.isin(source.labels[pool], list(HEART))
    out = []
    for lib in np.unique(source.lib[pool]):
        rows = source.lib[pool] == lib
        if is_heart[rows].mean() >= HEART_LIBRARY_MIN:
            out.append(str(lib))
    return out


def deep_rows(depth, lineage, top=0.3, drop=0.02):
    """Anchor rows in the top `top` share by UMI count within their lineage, without the very top `drop`."""
    keep = np.zeros(len(depth), bool)
    for k in np.unique(lineage):
        rows = np.flatnonzero(lineage == k)
        lo, hi = np.quantile(depth[rows], [1 - top, 1 - drop])
        keep[rows[(depth[rows] >= lo) & (depth[rows] <= hi)]] = True
    return np.flatnonzero(keep)


def build(cfg, source, pool, anchor, trend, n_total, *, elapsed=None, heart_target=True, seed=SEED, key="", info=None):
    """Cells of the prediction for one configuration. `pool` = source rows the prediction may use.

    heart_target: the target is a specifically dissected heart (true for E12.5 per the organisers' Data Protocol).
    With base "heart-library" the base cells are then the stage's heart-proper libraries; a stage without one
    falls back to its heart-labelled cells. If the target shares the source's dissection, all cells are used.
    """
    pool = np.asarray(pool)
    f = cfg["f"] if "f" in cfg else cfg["f_per_elapsed"] * elapsed
    libs = heart_libraries(source, pool) if cfg["base"] == "heart-library" and heart_target else []
    kw = dict(seed=seed, trend=trend, trend_strength=cfg.get("trend", 0.0), harmonise_strength=cfg.get("harmonise", 0.0))
    if libs:
        in_heart_lib = np.isin(source.lib[pool], libs)
        main, other = pool[in_heart_lib], pool[~in_heart_lib]
        if cfg.get("shallow_share", 0.0) > 0:
            kw.update(shares=lineage_shares(source, main), shallow_pool=other, shallow_share=cfg["shallow_share"])
    elif cfg["base"] == "heart-library" and heart_target:
        main = pool[np.isin(source.labels[pool], list(HEART))]
    else:
        main = pool
    kw["shift_key"] = f"{key}|{'+'.join(libs) or ('heart' if len(main) < len(pool) else 'all')}"
    if cfg.get("anchor", "natural") == "deep" and f > 0:
        store = anchor.__dict__.setdefault("_deep", {})
        if kw["shift_key"] not in store:
            near = centroid_transfer(source.X[main], source.labels[main], anchor.x)
            store[kw["shift_key"]] = deep_rows(depth_of(anchor.x), lineage_of(near))
        kw["anchor_rows"] = store[kw["shift_key"]]
    cells, rows, picked = predict(source, main, anchor, n_total, f, return_rows=True, **kw)
    if info is not None:
        info.update({"f": float(f), "heart_libraries": libs, "base_rows": rows, "anchor_rows": picked,
                     "base_cells": int(len(rows)), "anchor_cells": int(len(picked))})
    return cells
