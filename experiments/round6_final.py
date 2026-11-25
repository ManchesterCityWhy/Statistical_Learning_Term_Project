"""Round 6 final evaluation of named configurations (the file as exported, its ablations, the round-5 file).

    python -m experiments.round6_final scen  [--split confirmation]    zero elapsed time, the real test source
    python -m experiments.round6_final proxy [--split confirmation]    time passes, proxy form of every configuration

`scen` (scenario table). reference = E9.5; predictions are built from E10.5 cells that are in no target half,
at the fractions the exported file uses. Scenarios for the unknown E12.5 sample:
    heart/deep, heart/mid, heart/shallow   target = E10.5 library 3 at three depths (hearts dissected specifically)
    all/asis                               target = all three E10.5 libraries (as much surrounding tissue as E10.5)
No time passes in these scenarios, so they price the base, the depth and the cost of the anchor, not its benefit.

`proxy`. Every configuration in proxy form: anchor share = rho x elapsed share of the source-to-anchor interval;
heart targets use heart cells, same-dissection targets all cells (see round6_config.build).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import h3_rows, split_rows
from experiments.round5_exp4 import r4_cells
from experiments.round6_common import REAL, SEED, FastEval, Log, Mem, depth_of, thin_rows
from experiments.round6_config import build
from experiments.round6_method import Anchor
from experiments.round6_s1 import with_anchor
from experiments.round6_suite import Suite
from t1_pipeline.data import board_spec

N = 2000
TEST_ELAPSED = 0.5       # E12.5 lies halfway between the source (E10.5) and the anchor (E14.5)
R5_RHO = 0.75            # round-5 rule: anchor share = 0.75 x elapsed share (0.375 on the test)
PRIOR = {"heart/deep": .8 * .6, "heart/mid": .8 * .2, "heart/shallow": .8 * .2, "all/asis": .2}


def configs(rho, trend):
    r6 = {"kind": "round6", "base": "heart-library", "f_per_elapsed": rho, "harmonise": 1.0, "trend": trend}
    out = {
        # "R5" is the round-5 METHOD re-run on the proxy's own pool (its base and anchor share), not the round-5 file
        "R5 (round-5 file)": {"kind": "round5", "rho": R5_RHO, "harmonise": 0.0},
        "R6 (candidate)": r6,
        # ablations of the candidate, one change at a time
        "R6 raw anchor": {**r6, "harmonise": 0.0},
        "R6 base only": {**r6, "f_per_elapsed": 0.0},
        "R6 base, all libraries": {**r6, "base": "all"},
        "R5 base only": {"kind": "round5", "rho": 0.0, "harmonise": 0.0},
        "R5 base, harmonised anchor": {"kind": "round5", "rho": R5_RHO, "harmonise": 1.0},
    }
    if rho != R5_RHO:     # only informative when the candidate's share rule differs from round 5's
        out["R6 at round-5 share"] = {**r6, "f_per_elapsed": R5_RHO}
        out["R6 raw anchor at round-5 share"] = {**r6, "harmonise": 0.0, "f_per_elapsed": R5_RHO}
        out["R5 base, R6 anchor rule"] = {"kind": "round5", "rho": rho, "harmonise": 1.0}
    return out


def cells_hash(cells):
    """SHA-256 of the float32 cells: two runs produced the same prediction iff the hashes agree."""
    return hashlib.sha256(np.ascontiguousarray(cells, dtype=np.float32).tobytes()).hexdigest()


def cells_of(cfg, source, pool, r4_ref, anchor, trend, elapsed, heart_target, key, n=N, info=None, seed=SEED):
    """seed: sampling seed. The round-5 base draws its cells with the fixed seed of the round-5 code; for round-5
    configurations only the anchor draw and the base subsample follow `seed`."""
    if cfg["kind"] == "round5":
        f = cfg["rho"] * elapsed
        k = int(round(n * f))
        r4 = r4_cells(source, np.asarray(pool), r4_ref, trend, n)[0]
        shift = anchor.shift(source, np.asarray(pool), f"{key}|r5") if cfg["harmonise"] else None
        if info is not None:
            info.update({"f": f, "base_cells": n - k, "anchor_cells": k})
        return with_anchor(r4, anchor, f, seed, shift, cfg["harmonise"])
    return build(cfg, source, pool, anchor, trend, n, elapsed=elapsed, heart_target=heart_target, key=key, info=info, seed=seed)


def scenario_setup(split, genes):
    a, b = ("development", "ceiling") if split == "development" else ("confirmation", "confirmation_ceiling")
    e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    split10, h3 = split_rows(e9, e10), h3_rows(e10)
    ref = e9.dense(split10["reference"])
    used = np.unique(np.concatenate([h3[k] for k in h3] + [split10[k] for k in split10 if k != "reference"]))
    rest = np.setdiff1d(np.arange(e10.n_cells), used)
    rng = lambda: np.random.default_rng(SEED)
    lib3, lib12 = rest[e10.lib[rest] == "3"], rest[e10.lib[rest] != "3"]
    share3 = float((e10.lib == "3").mean())
    keep12 = np.sort(rng().choice(lib12, int(round(len(lib3) * (1 - share3) / share3)), replace=False))
    pool = np.sort(np.concatenate([lib3, keep12]))                 # no target cell, natural library shares
    d1, d2 = depth_of(e10.X[np.flatnonzero(e10.lib == "1")]), depth_of(e10.X[np.flatnonzero(e10.lib == "2")])
    targets = {"heart/deep": (e10.X[h3[a]], e10.X[h3[b]]),
               "heart/mid": (thin_rows(e10.X[h3[a]], d2, rng()), thin_rows(e10.X[h3[b]], d2, rng())),
               "heart/shallow": (thin_rows(e10.X[h3[a]], d1, rng()), thin_rows(e10.X[h3[b]], d1, rng())),
               "all/asis": (e10.X[split10[a]], e10.X[split10[b]])}
    evs = {k: FastEval(ref, t.toarray(), c.toarray()) for k, (t, c) in targets.items()}
    r4_ref = e10.dense(np.sort(rng().choice(pool, N, replace=False)))
    return e10, pool, r4_ref, evs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("scen", "proxy"))
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--rho", type=float, required=True, help="candidate anchor share per elapsed share")
    parser.add_argument("--trend", type=float, default=0.0)
    parser.add_argument("--only", default="", help="comma-separated configuration names (default: all)")
    parser.add_argument("--proxies", default="gen,p8/deep,p8/all,2d/deep,2d/all,D/raw,D/dm,raw,h3,dm")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seeds", default=str(SEED), help="sampling seeds; more than one = resampling noise (development only)")
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]
    assert seeds == [SEED] or args.split == "development", "noise runs use the development split"
    assert seeds == [SEED] or args.tag, "a multi-seed run needs --tag so that it cannot overwrite the main table"
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    ax, _ = load_anchor(genes, verbose=False)
    anchor = Anchor(ax)
    cfgs = configs(args.rho, args.trend)
    if args.only:
        cfgs = {k: v for k, v in cfgs.items() if k in args.only.split(",")}
    log = Log(REAL / "runs" / "round6_final" / f"{args.mode}_{args.split}_rho{args.rho:g}_trend{args.trend:g}{args.tag}.csv")
    pd.set_option("display.width", 250, "display.max_rows", 300)
    t0 = time.time()
    if args.mode == "scen":
        e10, pool, r4_ref, evs = scenario_setup(args.split, genes)
        print("scenario pool", len(pool), "cells; library shares", {k: round(float((e10.lib[pool] == k).mean()), 3) for k in "123"}, f"{time.time() - t0:.0f}s", flush=True)
        hashes = {}
        for seed in seeds:
            for name, cfg in cfgs.items():
                info = {}
                cells = cells_of(cfg, e10, pool, r4_ref, anchor, trend, TEST_ELAPSED, True, "scen", info=info, seed=seed)
                if seed == SEED:
                    hashes[name] = cells_hash(cells)
                for scen, ev in evs.items():
                    log.add(ev, cells, config=name, scenario=scen, split=args.split, f=round(info["f"], 4), seed=seed)
        log.path.with_suffix(".cells_sha256.json").write_text(json.dumps(hashes, indent=1))
        frame = pd.DataFrame(log.records)
        if len(seeds) > 1:
            print("resampling sd over seeds", seeds)
            print(frame.pivot_table(index="config", columns="scenario", values="score", aggfunc=["mean", "std"], sort=False).round(2).to_string())
            wide = frame.pivot_table(index="seed", columns=["config", "scenario"], values="score")
            names = list(cfgs)
            if len(names) >= 2:
                diff = wide[names[1]] - wide[names[0]]
                print(f"difference {names[1]} - {names[0]} per seed:"); print(diff.round(2).to_string()); print("sd", diff.std().round(2).to_dict())
        table = frame.pivot_table(index="config", columns="scenario", values="score", sort=False)[list(PRIOR)]
        table["expected"] = sum(table[k] * w for k, w in PRIOR.items())
        table["heart only"] = (.6 * table["heart/deep"] + .2 * table["heart/mid"] + .2 * table["heart/shallow"])
        print(table.round(2).to_string())
    else:
        suite = Suite(args.split, tuple(args.proxies.split(",")), anchor=ax)
        for pname, p in suite.proxies.items():
            for seed in seeds:
                for name, cfg in cfgs.items():
                    info = {}
                    cells = cells_of(cfg, p.source, p.pool, p.r4_ref, anchor, trend, p.elapsed, p.heart_target, pname, info=info, seed=seed)
                    log.add(p.ev, cells, config=name, proxy=pname, split=args.split, f=round(info["f"], 4), seed=seed)
        frame = pd.DataFrame(log.records)
        if len(seeds) > 1:
            print("resampling sd over seeds", seeds)
            print(frame.pivot_table(index="config", columns="proxy", values="score", aggfunc=["mean", "std"], sort=False).round(2).to_string())
        print(frame.pivot_table(index="config", columns="proxy", values="score", sort=False).round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m)
        print(frame.pivot_table(index="config", columns="scenario" if args.mode == "scen" else "proxy", values=m, sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
