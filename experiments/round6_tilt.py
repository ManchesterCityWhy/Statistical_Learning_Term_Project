"""Round 6 exploration (development split): a two-sided maturity tilt.

Idea. E12.5 lies between the source cells (E10.5) and the anchor nuclei (E14.5). Within every lineage the source
cells differ in how far they have matured, and so do the anchor nuclei. Instead of drawing both populations
uniformly, draw the source cells that look older and the anchor nuclei that look younger:

    lineage direction  d_L = lineage-specific log-ratio anchor / source (the shift shared by all lineages removed)
    maturity of a cell = mean expression of the 150 genes that rise most along d_L minus that of the 150 that fall
                         most, with log-depth regressed out, standardised within the lineage and the population
    source cells drawn with weight exp(+beta * maturity), anchor nuclei with weight exp(-gamma * maturity),
    lineage shares of each population kept as they are.

Only the source stage and the anchor enter; lineages without a counterpart (blood) are drawn uniformly.

    python -m experiments.round6_tilt --proxies p8/deep,p8/all
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round6_anchor import HARMONISE_LINEAGES, common_shift, harmonise, lineage_means
from experiments.round6_common import LINEAGES, REAL, SEED, Log, centroid_transfer, depth_of, lineage_of
from experiments.round6_method import Anchor, lineage_shares
from experiments.round6_s1 import feasible_shares
from experiments.round6_suite import Suite
from t1_pipeline.data import board_spec

N = 2000


def directions(source, rows, anchor_x, eps=0.1, n_genes=150, min_cp10k=0.3):
    """Per lineage: (up genes, down genes) of the lineage-specific anchor/source log-ratio; plus anchor lineages."""
    rows = np.asarray(rows)
    shift, ctx = common_shift(source, rows, anchor_x)
    lin_a = ctx["lineage"]
    ma = lineage_means(anchor_x, lin_a)
    ms = lineage_means(source.X[rows], lineage_of(source.labels[rows]))
    out = {}
    for k in HARMONISE_LINEAGES:
        if k in ma and k in ms:
            d = np.log((ma[k] + eps) / (ms[k] + eps)) + shift          # shared (non-developmental) part removed
            d[(ma[k] + ms[k]) / 2 < min_cp10k] = 0.0
            out[k] = (np.argsort(-d)[:n_genes], np.argsort(d)[:n_genes])
    return out, lin_a, shift


def maturity(x, lineage, dirs):
    """Within-lineage z-score of up-minus-down expression with log-depth regressed out (0 for lineages without a direction)."""
    z = np.zeros(x.shape[0])
    logd = np.log(depth_of(x))
    for k, (up, down) in dirs.items():
        rows = np.flatnonzero(lineage == k)
        if len(rows) < 20:
            continue
        s = np.asarray(x[rows][:, up].mean(axis=1)).ravel() - np.asarray(x[rows][:, down].mean(axis=1)).ravel()
        slope, intercept = np.polyfit(logd[rows], s, 1)
        r = s - (slope * logd[rows] + intercept)
        z[rows] = (r - r.mean()) / (r.std() + 1e-9)
    return z


def weighted_rows(rows, weights, n, rng):
    """n distinct rows, inclusion favouring large weights (Efraimidis-Spirakis)."""
    keys = rng.exponential(size=len(rows)) / np.maximum(weights, 1e-300)
    return rows[np.argpartition(keys, n - 1)[:n]]


def draw(rows, lineage, z, n, rng, strength, shares=None):
    """n rows with lineage quotas (natural shares of `rows` unless given) and weights exp(strength * z)."""
    if shares is None:
        shares = {k: float((lineage == k).mean()) for k in np.unique(lineage)}
    quota = {k: int(round(n * v)) for k, v in shares.items()}
    quota[max(quota, key=quota.get)] += n - sum(quota.values())
    out = []
    for k, q in quota.items():
        mine = np.flatnonzero(lineage == k)
        if q <= 0:
            continue
        if q > len(mine):
            raise ValueError(f"{k}: {q} requested, {len(mine)} available")
        out.append(rows[weighted_rows(mine, np.exp(strength * z[mine]), q, rng)])
    return np.sort(np.concatenate(out))


class Tilt:
    """Everything that depends only on (source pool, anchor): directions, maturities, harmonisation shift."""

    def __init__(self, source, pool, anchor_x):
        self.source, self.pool, self.anchor = source, np.asarray(pool), anchor_x
        dirs, self.lin_a, self.shift = directions(source, self.pool, anchor_x)
        self.lin_s = lineage_of(source.labels[self.pool])
        self.z_s = maturity(source.X[self.pool], self.lin_s, dirs)
        self.z_a = maturity(anchor_x, self.lin_a, dirs)
        self.lineages = list(dirs)

    def predict(self, n_total, f, beta, gamma, seed=SEED, shares=None, harmonise_strength=1.0):
        rng = np.random.default_rng(seed)
        k = int(round(n_total * f))
        rows = draw(self.pool, self.lin_s, self.z_s, n_total - k, rng, beta, shares)
        cells = self.source.dense(rows)
        if k:
            picked = draw(np.arange(self.anchor.shape[0]), self.lin_a, self.z_a, k, rng, -gamma)
            nuclei = self.anchor[picked].toarray().astype(np.float32)
            cells = np.vstack([cells, harmonise(nuclei, self.shift, harmonise_strength)])
        return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--proxies", default="gen,p8/deep,p8/all,2d/deep,2d/all,D/raw,D/dm")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    _, genes = board_spec("T1:val")
    ax, _ = load_anchor(genes, verbose=False)
    suite = Suite(args.split, tuple(args.proxies.split(",")), anchor=ax)
    log = Log(REAL / "runs" / "round6_tilt" / f"scores_{args.split}{args.tag}.csv")
    for name, p in suite.proxies.items():
        tilt = Tilt(p.source, p.pool, ax)
        f_rule = round(0.75 * p.elapsed, 4)
        oracle = feasible_shares(p.source, p.pool, p.target_shares, N)
        print(name, "tilted lineages", tilt.lineages, "anchor share by rule", f_rule, flush=True)
        meta = dict(proxy=name, split=args.split)
        for base, shares in (("nat", None), ("oracle", oracle)):
            for f in sorted({0.0, f_rule, 0.375}):
                for beta in (0.0, 0.5, 1.0, 2.0):
                    for gamma in ((0.0,) if f == 0 else (0.0, 1.0, 2.0)):
                        log.add(p.ev, tilt.predict(N, f, beta, gamma, shares=shares), base=base, f=f, beta=beta, gamma=gamma, **meta)
    frame = pd.DataFrame(log.records)
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print(frame.pivot_table(index=["base", "f", "beta", "gamma"], columns="proxy", values="score", sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
