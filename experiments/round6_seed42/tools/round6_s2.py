"""Round 6, sweep 2 (development split): harmonisation mode and the within-cell bulk trend on the base cells.

Bases as in sweep 1 (`oracle` is ANALYSIS ONLY). Anchor harmonised at full strength with the shared shift taken
as the minimum ("min") or the median ("median") absolute lineage log-ratio; base cells optionally moved along the
bulk E9.5 -> E14.5 trend (strength on the round-2 scale, 3 = the value used since round 2).

    python -m experiments.round6_s2 --proxies gen,2d/deep,2d/all
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round6_common import HEART, REAL, SEED, Log
from experiments.round6_method import Anchor, predict
from experiments.round6_s1 import feasible_shares
from experiments.round6_suite import Suite
from t1_pipeline.data import board_spec

N = 2000
FRACTIONS = (0.0, 0.15, 0.2, 0.25, 0.3, 0.375)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--proxies", default="gen,p8/deep,p8/all,2d/deep,2d/all,D/raw,D/dm")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    ax, _ = load_anchor(genes, verbose=False)
    anchor = Anchor(ax)
    suite = Suite(args.split, tuple(args.proxies.split(",")), anchor=ax)
    log = Log(REAL / "runs" / "round6_s2" / f"scores_{args.split}{args.tag}.csv")
    for name, p in suite.proxies.items():
        heart = p.pool[np.isin(p.source.labels[p.pool], list(HEART))]
        oracle = feasible_shares(p.source, p.pool, p.target_shares, N)
        bases = {"nat": dict(pool=p.pool), "oracle": dict(pool=p.pool, shares=oracle), "heart": dict(pool=heart)}
        meta = dict(proxy=name, split=args.split, elapsed=round(p.elapsed, 3))
        for base, b in bases.items():
            key = f"{name}|{base}"
            for f in FRACTIONS:
                grid = [("min", t) for t in (0.0, 1.5, 3.0, 6.0)] + ([("median", 0.0)] if f > 0 else [])
                for mode, t in grid:
                    cells = predict(p.source, b["pool"], anchor, N, f, seed=SEED, shares=b.get("shares"), trend=trend, trend_strength=t,
                                    harmonise_strength=1.0, harmonise_mode=mode, shift_key=key)
                    log.add(p.ev, cells, base=base, f=f, mode=mode, trend=t, **meta)
    frame = pd.DataFrame(log.records)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(frame.pivot_table(index=["base", "mode", "trend", "f"], columns="proxy", values="score", sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
