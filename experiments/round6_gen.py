"""Round 6: do the round-2/4 deconvolution, the within-cell bulk trend and the E14.5 anchor carry over to a
transition they were never tuned on?  E8.5 -> E9.5 (one day, anchor five more days ahead: elapsed share 1/6).

    reference = source = a 10% sample of E8.5 (no earlier stage exists to serve as a reference two stages back)
    target    = disjoint halves of 1,249 E9.5 cells

Every hyper-parameter of rounds 2-5 was chosen on E9.5 -> E10.5 (or E8.5 -> E10.5); nothing was ever scored
against E9.5 as a target, so this transition is an out-of-sample check of the *components*.

    python -m experiments.round6_gen [--split confirmation]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_exp4 import r4_cells
from experiments.round5_exp6 import mix
from experiments.round6_common import REAL, SEED, FastEval, Log, Mem
from t1_pipeline.data import board_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    args = parser.parse_args()
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    anchor, _ = load_anchor(genes, verbose=False)
    t0 = time.time()
    e8 = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
    e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    order = np.random.default_rng(SEED).permutation(e9.n_cells)
    k = 0 if args.split == "development" else 2
    target, ceiling = e9.dense(np.sort(order[k * 1249:(k + 1) * 1249])), e9.dense(np.sort(order[(k + 1) * 1249:(k + 2) * 1249]))
    ref = e8.dense(np.sort(np.random.default_rng(SEED).choice(e8.n_cells, round(.1 * e8.n_cells), replace=False)))
    ev = FastEval(ref, target, ceiling)
    print("ready", f"{time.time() - t0:.0f}s", "floor", ev.floor, "ceiling", ev.ceiling, "n_up", len(ev.up), "n_dn", len(ev.dn), flush=True)
    log = Log(REAL / "runs" / "round6_gen" / f"scores_{args.split}.csv")
    cands = {"unchanged": ref, "bulk3_only": apply_trend(ref.copy(), trend, 3 / 5, False, SEED),
             "bulk1.5_only": apply_trend(ref.copy(), trend, 1.5 / 5, False, SEED)}
    for s in (1.5, 3.0, 6.0):
        r4, r2 = r4_cells(e8, np.arange(e8.n_cells), ref, trend, len(ref), strength=s, bulk=3.0)
        cands[f"r2_s{s:g}"], cands[f"r4_s{s:g}"] = r2, r4
    for name, cells in cands.items():
        for f in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
            log.add(ev, mix(cells, anchor, f, np.random.default_rng(SEED)), proxy="gen/E8.5->E9.5", base=name, f=f, split=args.split)
    frame = pd.DataFrame(log.records)
    print(frame.pivot_table(index="base", columns="f", values="score", sort=False).round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m)
        print(frame.pivot_table(index="base", columns="f", values=m, sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
