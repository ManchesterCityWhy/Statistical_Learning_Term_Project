"""Round 6, sweep 1 (development split unless --split confirmation): what the anchor adds, for which base.

Bases (drawn from the proxy's source pool):
    nat      uniform over the pool                       (proxy form of "take the library as it is")
    heart    heart-labelled cells of the pool, natural proportions
    oracle   lineage quotas of the proxy TARGET          (ANALYSIS ONLY: uses the target's composition to ask what
                                                          the anchor adds once the composition is already right)
    r4       round-4/5 deconvolution + direction shift   (the base of the round-5 file)
Anchor: fraction f of the population, raw or harmonised (strength h of the shared lineage shift removed).

    python -m experiments.round6_s1 --proxies gen,2d/deep,2d/all
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_exp4 import r4_cells
from experiments.round6_anchor import harmonise
from experiments.round6_common import HEART, LINEAGES, REAL, SEED, Log, lineage_of
from experiments.round6_method import Anchor, predict
from experiments.round6_suite import Suite
from t1_pipeline.data import board_spec

N = 2000
FRACTIONS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375, 0.45)


def feasible_shares(source, pool, shares, n):
    """Target lineage quotas made feasible for a source pool (ANALYSIS helper for the oracle base).

    Lineages the source does not annotate are folded into their closest relative (cushion mesenchyme -> endocardium,
    macrophage -> other); a quota larger than what the pool holds is capped and the rest spread proportionally.
    """
    lin = lineage_of(source.labels[pool])
    have = {k: int((lin == k).sum()) for k in LINEAGES}
    s = dict(shares)
    for missing, into in (("Cushion", "Endo"), ("Mac", "Other")):
        if have[missing] == 0:
            s[into] = s.get(into, 0) + s.get(missing, 0)
            s[missing] = 0.0
    for _ in range(10):
        over = {k: s[k] * n - have[k] for k in LINEAGES if s[k] * n > have[k]}
        if not over:
            break
        spill = sum(over.values()) / n
        for k in over:
            s[k] = have[k] / n
        free = [k for k in LINEAGES if k not in over and s[k] > 0 and s[k] * n < have[k]]
        total = sum(s[k] for k in free)
        for k in free:
            s[k] += spill * s[k] / total
    return s


def with_anchor(cells, anchor, f, seed, shift=None, h=0.0):
    """Round-5 style mixing for a pre-built base: keep a random (1-f) of the base cells, add f anchor nuclei."""
    rng = np.random.default_rng(seed)
    k = int(round(f * len(cells)))
    if k == 0:
        return cells
    take = np.sort(rng.choice(anchor.x.shape[0], k, replace=False))
    keep = np.sort(rng.choice(len(cells), len(cells) - k, replace=False))
    nuclei = anchor.x[take].toarray().astype(np.float32)
    if h:
        nuclei = harmonise(nuclei, shift, h)
    return np.vstack([cells[keep], nuclei])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--proxies", default="gen,p8/deep,p8/all,2d/deep,2d/all,D/raw,D/dm")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    names = tuple(args.proxies.split(","))
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    ax, _ = load_anchor(genes, verbose=False)
    anchor = Anchor(ax)
    suite = Suite(args.split, names, anchor=ax)
    log = Log(REAL / "runs" / "round6_s1" / f"scores_{args.split}{args.tag}.csv")
    for name, p in suite.proxies.items():
        heart = p.pool[np.isin(p.source.labels[p.pool], list(HEART))]
        shift = anchor.shift(p.source, p.pool, name)
        oracle = feasible_shares(p.source, p.pool, p.target_shares, N)
        print(name, "oracle quotas", {k: round(v, 3) for k, v in oracle.items()}, "target", {k: round(v, 3) for k, v in p.target_shares.items()}, flush=True)
        r4 = r4_cells(p.source, p.pool, p.r4_ref, trend, N)[0]
        meta = dict(proxy=name, split=args.split, elapsed=round(p.elapsed, 3))
        for f in FRACTIONS:
            for h in ((0.0,) if f == 0 else (0.0, 0.5, 1.0)):
                kw = dict(seed=SEED, harmonise_strength=h, shift_key=name)
                log.add(p.ev, predict(p.source, p.pool, anchor, N, f, **kw), base="nat", f=f, h=h, **meta)
                log.add(p.ev, predict(p.source, p.pool, anchor, N, f, shares=oracle, **kw), base="oracle", f=f, h=h, **meta)
                log.add(p.ev, with_anchor(r4, anchor, f, SEED, shift, h), base="r4", f=f, h=h, **meta)
            log.add(p.ev, predict(p.source, heart, anchor, N, f, seed=SEED), base="heart", f=f, h=0.0, **meta)
        # resampling noise: the same configuration with other sampling seeds
        for seed in (1, 2, 3, 4):
            for f in (0.0, 0.2):
                log.add(p.ev, predict(p.source, p.pool, anchor, N, f, seed=seed), base="nat", f=f, h=0.0, noise_seed=seed, **meta)
    frame = pd.DataFrame(log.records)
    main_rows = frame[frame.get("noise_seed").isna()] if "noise_seed" in frame else frame
    pd.set_option("display.width", 250)
    for base in ("nat", "oracle", "r4", "heart"):
        print(f"\n=== base {base}: score by proxy (rows) and anchor fraction (columns), h = 0 / 0.5 / 1")
        for h in (0.0, 0.5, 1.0):
            sub = main_rows[(main_rows.base == base) & ((main_rows.h == h) | (main_rows.f == 0))]
            if len(sub):
                print(f"h={h}"); print(sub.pivot_table(index="proxy", columns="f", values="score", sort=False).round(2).to_string())
    if "noise_seed" in frame:
        noise = frame[(frame.base == "nat") & (frame.h == 0)].groupby(["proxy", "f"])["score"].agg(["std", "count"])
        print("\nresampling sd (nat base, 5 seeds incl. 42):"); print(noise[noise["count"] >= 5].round(3).to_string())


if __name__ == "__main__":
    main()
