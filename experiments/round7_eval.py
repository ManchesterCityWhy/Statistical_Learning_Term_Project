"""Round 7 evaluation of displacement-interpolation configurations.

    python -m experiments.round7_eval proxy --grid g1 --proxies gen,p8/deep,2d/deep [--split confirmation]
    python -m experiments.round7_eval scen  --grid g1 [--split confirmation]

proxy  time passes. Anchor share f = rho_f x elapsed, position t = rho_t x elapsed; forward shift of the base
       fwd = t, backward shift of the anchor nuclei per `bwd` ("interp" = 1 - t, or a number).
       Base "heart": the proxy form of the exported base (round6_config.build: heart-proper libraries, else
       heart-labelled cells, else all cells when the target shares the source's dissection).
       Base "oracle": lineage quotas of the proxy target (ANALYSIS ONLY).
scen   zero elapsed time, the real test inputs at the test-time parameters (elapsed = 0.5); targets = E10.5
       library 3 at three depths, or all E10.5 libraries. Prices what every step costs if nothing developed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from experiments.round5_anchor import load_anchor
from experiments.round6_common import HEART, REAL, SEED, Log
from experiments.round6_config import deep_rows, heart_libraries
from experiments.round6_common import depth_of, lineage_of
from experiments.round6_s1 import feasible_shares
from experiments.round6_suite import Suite
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.data import board_spec

N = 2000
TEST_ELAPSED = 0.5


def C(name, rho_f=0.0, rho_t=0.0, bwd=0.0, h=1.0, mode="min", level="lineage", pool_k=1, deep=False, base="heart", rho_dev=0.0, k_prof=10, rho_tc=None, days_trend=0.0,
      a_shift=None, tilt=0.0, rho_thin=0.0, depth_from="source", real_share=0.0):
    """a_shift: share of the developmental shifts (two-witness, trend) the anchor nuclei receive. None = predict7's
    default (1 when moved back completely, else 0); "pos" = position-consistent: nuclei left at position 1 - bwd on
    the source -> anchor path get only what is still missing up to the target position (the elapsed share)."""
    return dict(name=name, rho_f=rho_f, rho_t=rho_t, bwd=bwd, h=h, mode=mode, level=level, pool_k=pool_k, deep=deep, base=base, rho_dev=rho_dev, k_prof=k_prof,
                rho_tc=rho_tc, days_trend=days_trend, a_shift=a_shift, tilt=tilt, rho_thin=rho_thin, depth_from=depth_from, real_share=real_share)


def grid(name):
    g = []
    if name == "g1":      # first look: does moving the two parts help, and what does it cost at zero elapsed time
        g += [C("base only")]
        g += [C("R6b mix rho.4", rho_f=.4), C("R6v1 mix rho.75", rho_f=.75), C("mix rho1", rho_f=1.0)]
        for rf in (.4, .75, 1.0):
            g += [C(f"bwd1 rho{rf:g}", rho_f=rf, bwd=1.0), C(f"bwd.5 rho{rf:g}", rho_f=rf, bwd=.5)]
        for rt in (.5, 1.0, 1.5):
            g += [C(f"fwd only t{rt:g}", rho_t=rt)]
        for rf in (.4, .75, 1.0):
            for rt in (.5, 1.0):
                g += [C(f"interp rho{rf:g} t{rt:g}", rho_f=rf, rho_t=rt, bwd="interp")]
        g += [C("interp rho.75 t1 pool3", rho_f=.75, rho_t=1.0, bwd="interp", pool_k=3),
              C("interp rho.75 t1 poolM", rho_f=.75, rho_t=1.0, bwd="interp", pool_k="match"),
              C("mix rho.75 poolM", rho_f=.75, pool_k="match"),
              C("bwd1 rho.75 poolM", rho_f=.75, bwd=1.0, pool_k="match"),
              C("interp rho.75 t1 median", rho_f=.75, rho_t=1.0, bwd="interp", mode="median"),
              C("interp rho.75 t1 type", rho_f=.75, rho_t=1.0, bwd="interp", level="type"),
              C("interp rho.75 t1 h0", rho_f=.75, rho_t=1.0, bwd="interp", h=0.0)]
    elif name == "g1o":   # the same questions with the composition already right (analysis)
        g += [C("base only", base="oracle"), C("mix rho.75", rho_f=.75, base="oracle"), C("mix rho1", rho_f=1.0, base="oracle"),
              C("bwd1 rho.75", rho_f=.75, bwd=1.0, base="oracle"), C("bwd1 rho1", rho_f=1.0, bwd=1.0, base="oracle"),
              C("fwd only t1", rho_t=1.0, base="oracle"),
              C("interp rho.75 t1", rho_f=.75, rho_t=1.0, bwd="interp", base="oracle"),
              C("interp rho1 t1", rho_f=1.0, rho_t=1.0, bwd="interp", base="oracle"),
              C("interp rho1 t1 poolM", rho_f=1.0, rho_t=1.0, bwd="interp", pool_k="match", base="oracle"),
              C("interp rho1.5 t1", rho_f=1.5, rho_t=1.0, bwd="interp", base="oracle")]
    elif name == "g2":    # anchor nuclei made less foreign: pooled / re-sequenced at the source depth, cost at zero elapsed time
        g += [C("base only"), C("R6b mix rho.4", rho_f=.4), C("R6v1 mix rho.75", rho_f=.75)]
        for rf in (.4, .75, 1.0):
            g += [C(f"bwd1 poolM rho{rf:g}", rho_f=rf, bwd=1.0, pool_k="match"),
                  C(f"bwd1 resample rho{rf:g}", rho_f=rf, bwd=1.0, pool_k="resample"),
                  C(f"resample rho{rf:g}", rho_f=rf, pool_k="resample"),
                  C(f"interp t1 resample rho{rf:g}", rho_f=rf, rho_t=1.0, bwd="interp", pool_k="resample"),
                  C(f"interp t1 poolM rho{rf:g}", rho_f=rf, rho_t=1.0, bwd="interp", pool_k="match")]
        g += [C("bwd1 resample rho.75 median", rho_f=.75, bwd=1.0, pool_k="resample", mode="median"),
              C("bwd1 resample rho.75 type", rho_f=.75, bwd=1.0, pool_k="resample", level="type")]
    elif name in ("g3", "g3o"):   # two-witness developmental shift (needs a previous stage: p8-type proxies and the scenario table)
        b = "oracle" if name == "g3o" else "heart"
        g += [C("base only", base=b)]
        for rd in (.5, 1.0, 1.5, 2.0):
            g += [C(f"dev {rd:g}", rho_dev=rd, base=b)]
        for rf in (.4, .75):
            g += [C(f"back-resample rho{rf:g}", rho_f=rf, bwd=1.0, pool_k="resample", base=b)]
            for rd in (.5, 1.0, 1.5):
                g += [C(f"back-resample rho{rf:g} dev {rd:g}", rho_f=rf, bwd=1.0, pool_k="resample", rho_dev=rd, base=b)]
            g += [C(f"raw rho{rf:g}", rho_f=rf, base=b), C(f"raw rho{rf:g} dev 1", rho_f=rf, rho_dev=1.0, base=b)]
    elif name == "g4":    # details of the re-sequenced anchor: neighbourhood size, type-level back shift, share
        g += [C("base only")]
        for rf in (.4, .75):
            for kp in (3, 5, 10, 20):
                g += [C(f"back-resample rho{rf:g} k{kp}", rho_f=rf, bwd=1.0, pool_k="resample", k_prof=kp)]
            g += [C(f"back-resample rho{rf:g} k10 type", rho_f=rf, bwd=1.0, pool_k="resample", level="type"),
                  C(f"back-resample rho{rf:g} k5 type", rho_f=rf, bwd=1.0, pool_k="resample", level="type", k_prof=5),
                  C(f"back-resample rho{rf:g} k10 median", rho_f=rf, bwd=1.0, pool_k="resample", mode="median")]
        for rf in (.2, .6, .9):
            g += [C(f"back-resample rho{rf:g} k10", rho_f=rf, bwd=1.0, pool_k="resample")]
    elif name == "g5":    # composition only (no anchor nuclei): heart cells resampled to the interpolated lineage composition
        g += [C("base only")]
        for r in (.4, .75, 1.0, 1.5):
            g += [C(f"composition rho{r:g}", rho_tc=r),
                  C(f"back-resample rho{r:g} type", rho_f=r, bwd=1.0, pool_k="resample", level="type"),
                  C(f"composition rho{r:g} + half as anchor", rho_tc=r, rho_f=r / 2, bwd=1.0, pool_k="resample", level="type")]
    elif name in ("g6", "g6o"):   # add-ons on the type-level re-sequenced anchor: two-witness shift, bulk trend (days of development per day elapsed)
        b = "oracle" if name == "g6o" else "heart"
        kw = dict(bwd=1.0, pool_k="resample", level="type", base=b)
        g += [C("base only", base=b), C("base + trend x1", days_trend=1.0, base=b), C("base + trend x1.5", days_trend=1.5, base=b)]
        for rf in (.75, 1.0):
            g += [C(f"anchor rho{rf:g}", rho_f=rf, **kw), C(f"anchor rho{rf:g} + dev 1", rho_f=rf, rho_dev=1.0, **kw),
                  C(f"anchor rho{rf:g} + trend x0.5", rho_f=rf, days_trend=.5, **kw), C(f"anchor rho{rf:g} + trend x1", rho_f=rf, days_trend=1.0, **kw),
                  C(f"anchor rho{rf:g} + trend x1.5", rho_f=rf, days_trend=1.5, **kw),
                  C(f"anchor rho{rf:g} + dev 1 + trend x1", rho_f=rf, rho_dev=1.0, days_trend=1.0, **kw)]
    elif name == "g7":    # the round-7 configuration, its ablations and the round-6 files, one table
        kw = dict(bwd=1.0, pool_k="resample", level="type")
        g += [C("base only"),
              C("round 6 file (raw anchor, rho 0.4)", rho_f=.4), C("round 6 first version (raw anchor, rho 0.75)", rho_f=.75),
              C("R7 anchor only (rho 0.9)", rho_f=.9, **kw),
              C("R7 anchor + two-witness", rho_f=.9, rho_dev=1.0, **kw),
              C("R7 anchor + trend", rho_f=.9, days_trend=1.0, **kw),
              C("R7 anchor + two-witness + trend x0.5", rho_f=.9, rho_dev=1.0, days_trend=.5, **kw),
              C("R7 (candidate): anchor + two-witness + trend", rho_f=.9, rho_dev=1.0, days_trend=1.0, **kw),
              C("R7 anchor + two-witness + trend x1.5", rho_f=.9, rho_dev=1.0, days_trend=1.5, **kw),
              C("R7 with rho 0.6", rho_f=.6, rho_dev=1.0, days_trend=1.0, **kw),
              C("R7 with rho 1.2", rho_f=1.2, rho_dev=1.0, days_trend=1.0, **kw),
              C("R7 two-witness + trend, no anchor", rho_dev=1.0, days_trend=1.0),
              C("R7 with lineage-level back shift", rho_f=.9, rho_dev=1.0, days_trend=1.0, bwd=1.0, pool_k="resample", level="lineage")]
    elif name == "g8":    # how far the re-sequenced anchor nuclei are moved back to the source's cell-type means (1 = completely)
        kw = dict(pool_k="resample", level="type", rho_dev=1.0, days_trend=1.0)
        g += [C("base only"), C("round 6 first version (raw anchor, rho 0.75)", rho_f=.75), C("raw anchor rho 0.9 + trend", rho_f=.9, days_trend=1.0)]
        for b in (1.0, .75, .5, .25, 0.0):
            g += [C(f"R7 rho 0.9, anchor moved back {b:g}", rho_f=.9, bwd=b, **kw)]
        g += [C("R7 rho 0.9, anchor moved back 0.5, lineage level", rho_f=.9, bwd=.5, pool_k="resample", level="lineage", rho_dev=1.0, days_trend=1.0),
              C("R7 rho 0.6, anchor moved back 0.5", rho_f=.6, bwd=.5, **kw), C("R7 rho 1.2, anchor moved back 0.5", rho_f=1.2, bwd=.5, **kw),
              C("R7 rho 0.9, anchor moved back 0.5, no two-witness", rho_f=.9, bwd=.5, pool_k="resample", level="type", days_trend=1.0),
              C("R7 rho 0.9, anchor moved back 0.5, k5", rho_f=.9, bwd=.5, k_prof=5, **kw)]
    elif name == "g9":    # neighbourhood size of the re-sequenced nuclei, for the complete configuration
        kw = dict(rho_f=.9, bwd=1.0, pool_k="resample", level="type", rho_dev=1.0, days_trend=1.0)
        g += [C("base only")] + [C(f"R7 candidate, k_prof {k}", k_prof=k, **kw) for k in (3, 5, 10, 20)]
    elif name == "g10":   # final configuration (k_prof 20) and its one-at-a-time ablations, plus the round-6 files
        kw = dict(rho_f=.9, bwd=1.0, pool_k="resample", level="type", k_prof=20)
        g += [C("base only"),
              C("round 6 file (raw anchor, rho 0.4)", rho_f=.4), C("round 6 first version (raw anchor, rho 0.75)", rho_f=.75),
              C("R7 (final): anchor + two-witness + trend", rho_dev=1.0, days_trend=1.0, **kw),
              C("R7 without two-witness", days_trend=1.0, **kw), C("R7 without trend", rho_dev=1.0, **kw),
              C("R7 without both", **kw),
              C("R7 without anchor", rho_dev=1.0, days_trend=1.0),
              C("R7 with raw anchor instead (rho 0.9)", rho_f=.9, rho_dev=1.0, days_trend=1.0)]
    elif name == "g11":   # fair versions of the comparisons the independent review flagged (anchor shifts made explicit)
        kw = dict(rho_f=.9, pool_k="resample", level="type", k_prof=20, rho_dev=1.0, days_trend=1.0)
        g += [C("base only"),
              C("round 6 file (raw anchor, rho 0.4)", rho_f=.4), C("round 6 first version (raw anchor, rho 0.75)", rho_f=.75),
              C("raw anchor rho 0.9; two-witness + trend on the base only", rho_f=.9, rho_dev=1.0, days_trend=1.0),
              C("raw anchor rho 0.75; two-witness + trend on the base only", rho_f=.75, rho_dev=1.0, days_trend=1.0),
              C("R7 (final): moved back 1", bwd=1.0, **kw)]
        g += [C(f"R7, moved back {b:g}, position-consistent shifts", bwd=b, a_shift="pos", **kw) for b in (.75, .5, .25, 0.0)]
        g += [C("R7, moved back 0.5, no shift on the anchor", bwd=.5, a_shift=0.0, **kw)]
    elif name == "g12":   # individual low-rank deviation restored in the re-sequenced nuclei (tilt); strength 3 was set by the
        # score-free dispersion criterion (round7_dispcheck), 1.5 is shown for the dose only
        kw = dict(rho_f=.9, bwd=1.0, pool_k="resample", level="type", k_prof=20, rho_dev=1.0, days_trend=1.0)
        g += [C("base only"),
              C("R7 (final): tilt 0", **kw), C("R7 + tilt 1.5", tilt=1.5, **kw), C("R7 + tilt 3", tilt=3.0, **kw),
              C("round 6 first version (raw anchor, rho 0.75)", rho_f=.75)]
    elif name == "g14":   # the synthetic nuclei of the final configuration replaced by real source cells of the assigned cell
        # types (real_share 1: all of them, 0.5: every second one); same cell-type composition, same shifts
        kw = dict(rho_f=.9, bwd=1.0, pool_k="resample", level="type", k_prof=20, rho_dev=1.0, days_trend=1.0)
        g += [C("base only"),
              C("R7 (final): synthetic nuclei", **kw),
              C("R7 with half of the nuclei replaced by real cells", real_share=.5, **kw),
              C("R7 with all nuclei replaced by real cells", real_share=1.0, **kw),
              C("round 6 file (raw anchor, rho 0.4)", rho_f=.4)]
    elif name == "g13":   # ANALYSIS: do the E10.5-target proxies reward shallow cells as such? No developmental shifts.
        rs = dict(rho_f=.9, pool_k="resample", level="type", k_prof=20)
        g += [C("base only"),
              C("base, a share of its cells thinned to the anchor depth (no anchor)", rho_thin=.9),
              C("raw nuclei", rho_f=.9),
              C("re-sequenced at source depth, moved back 1", bwd=1.0, **rs),
              C("re-sequenced at source depth, not moved back", bwd=0.0, **rs),
              C("re-sequenced at the nuclei's own depth, moved back 1", bwd=1.0, depth_from="anchor", **rs),
              C("re-sequenced at the nuclei's own depth, not moved back", bwd=0.0, depth_from="anchor", **rs)]
    else:
        raise SystemExit(f"unknown grid {name}")
    return g


def params(cfg, elapsed):
    f = cfg["rho_f"] * elapsed
    t = cfg["rho_t"] * elapsed
    bwd = (1.0 - t) if cfg["bwd"] == "interp" else float(cfg["bwd"])
    return f, t, bwd


def base_pool(source, pool, heart_target):
    """The exported base in proxy form (same rule as round6_config.build)."""
    pool = np.asarray(pool)
    if not heart_target:
        return pool
    libs = heart_libraries(source, pool)
    if libs:
        return pool[np.isin(source.lib[pool], libs)]
    return pool[np.isin(source.labels[pool], list(HEART))]


def anchor_subset(anchor, source, pool, cfg):
    if not cfg["deep"]:
        return None
    ctx = anchor.context(source, pool, cfg["mode"], cfg["level"])
    return deep_rows(anchor.depth, ctx["lineage"])


TREND = {}


def cells_of(cfg, source, pool, anchor, elapsed, heart_target, target_shares=None, seed=SEED, n=N, prev=None, days_to_anchor=4.0):
    f, t, bwd = params(cfg, elapsed)
    tkw = {}
    if cfg.get("days_trend"):
        # the bulk trend spans five days (E9.5 -> E14.5); elapsed x days_to_anchor = days between source and target
        if "trend" not in TREND:
            from experiments.bulk_trend import load_trend
            TREND["trend"] = load_trend(board_spec("T1:val")[1])[0]
        tkw = dict(trend=TREND["trend"], trend_share=cfg["days_trend"] * elapsed * days_to_anchor / 5.0)
    dkw = dict(prev=prev, days_to_anchor=days_to_anchor, dev=cfg.get("rho_dev", 0.0) * elapsed) if cfg.get("rho_dev") else {}
    a_shift = cfg.get("a_shift")
    if a_shift == "pos":      # what the nuclei still lack up to the target position `elapsed`, as a share of the base's shift
        a_shift = float(np.clip((bwd - (1.0 - elapsed)) / elapsed, 0.0, 1.0))
    if cfg["base"] == "oracle":
        main, shares = np.asarray(pool), feasible_shares(source, np.asarray(pool), target_shares, n)
    else:
        main, shares = base_pool(source, pool, heart_target), None
        if cfg.get("rho_tc") is not None:
            from experiments.round7_method import composition_shares
            shares = composition_shares(source, main, anchor, cfg["rho_tc"] * elapsed, f, n, feasible_shares)
    cells = predict7(source, main, anchor, n, f, fwd=t, bwd=bwd if f > 0 else 0.0, harmonise=cfg["h"], mode=cfg["mode"],
                     level=cfg["level"], pool_k=cfg["pool_k"], anchor_rows=anchor_subset(anchor, source, main, cfg),
                     shares=shares, seed=seed, k_prof=cfg.get("k_prof", 10), anchor_shift=a_shift, tilt=cfg.get("tilt", 0.0),
                     depth_from=cfg.get("depth_from", "source"), real_share=cfg.get("real_share", 0.0),
                     return_info=bool(cfg.get("rho_thin")), **dkw, **tkw)
    if cfg.get("rho_thin"):
        # analysis only (g13): a share rho_thin x elapsed of the base cells is thinned to the anchor's depth distribution
        # (binomial thinning of the cell's own counts; no anchor expression enters). Only for configurations without shifts.
        from experiments.round6_common import thin_rows
        cells, info = cells
        rng = np.random.default_rng(seed + 7)
        rows = info["base_rows"]
        sub = np.sort(rng.choice(len(rows), int(round(cfg["rho_thin"] * elapsed * n)), replace=False))
        target = rng.choice(anchor.depth, len(sub), replace=True)
        thinned = thin_rows(source.X[rows[sub]], target, rng)
        cells[sub] = thinned.toarray() if hasattr(thinned, "toarray") else np.asarray(thinned)
    return cells, f, t, bwd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("proxy", "scen"))
    parser.add_argument("--grid", required=True)
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--proxies", default="gen,p8/deep,2d/deep")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seeds", default=str(SEED))
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]
    _, genes = board_spec("T1:val")
    ax, _ = load_anchor(genes, verbose=False)
    anchor = Anchor7(ax)
    cfgs = grid(args.grid)
    log = Log(REAL / "runs" / "round7_eval" / f"{args.mode}_{args.grid}_{args.split}{args.tag}.csv")
    pd.set_option("display.width", 250, "display.max_rows", 400)
    t0 = time.time()
    if args.mode == "scen":
        from experiments.round6_final import scenario_setup
        e10, pool, _, evs = scenario_setup(args.split, genes)
        from experiments.round6_common import Mem as _Mem
        e9 = _Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
        prev = (e9, np.arange(e9.n_cells))
        print("scenario pool", len(pool), f"{time.time() - t0:.0f}s", flush=True)
        for seed in seeds:
            for cfg in cfgs:
                if cfg["base"] == "oracle":
                    continue
                cells, f, t, bwd = cells_of(cfg, e10, pool, anchor, TEST_ELAPSED, True, seed=seed, prev=prev, days_to_anchor=4.0)
                for scen, ev in evs.items():
                    log.add(ev, cells, config=cfg["name"], scenario=scen, split=args.split, f=round(f, 4), t=round(t, 4), bwd=round(bwd, 4), seed=seed)
        col = "scenario"
    else:
        suite = Suite(args.split, tuple(args.proxies.split(",")), anchor=ax)
        for pname, p in suite.proxies.items():
            # previous stage of the proxy's source (only E9.5 sources have one: E8.5); days from the source to the anchor
            e8 = suite.stages.get("E8.5")
            is_e9 = p.source is suite.stages["E9.5"]
            prev = (e8, np.arange(e8.n_cells)) if (is_e9 and e8 is not None) else None
            days = 5.0 if is_e9 else 6.0
            for seed in seeds:
                for cfg in cfgs:
                    if cfg.get("rho_dev") and prev is None:
                        if args.grid not in ("g7", "g8", "g9", "g10", "g11", "g12", "g14"):
                            continue
                        cfg = {**cfg, "rho_dev": 0.0}      # no previous stage for this source: the two-witness shift is unavailable
                    cells, f, t, bwd = cells_of(cfg, p.source, p.pool, anchor, p.elapsed, p.heart_target, p.target_shares, seed=seed,
                                                prev=prev, days_to_anchor=days)
                    log.add(p.ev, cells, config=cfg["name"], proxy=pname, split=args.split, f=round(f, 4), t=round(t, 4), bwd=round(bwd, 4), seed=seed,
                            two_witness=bool(cfg.get("rho_dev")), trend_days=cfg.get("days_trend", 0.0))
        col = "proxy"
    frame = pd.DataFrame(log.records)
    print(f"done {time.time() - t0:.0f}s")
    print(frame.pivot_table(index="config", columns=col, values="score", sort=False).round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m)
        print(frame.pivot_table(index="config", columns=col, values=m, sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
