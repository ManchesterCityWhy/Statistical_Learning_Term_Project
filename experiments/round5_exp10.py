"""Round 5, batch 10: restrict the donor pool to cardiac lineages? Scored against a heart-proper target (E10.5 library 3)
and against all of E10.5, with and without the E14.5 anchor (fraction 0.15, the one-day value)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import CARDIAC, REAL, SEED, FastEval, Mem, h3_rows, split_rows
from experiments.round5_exp4 import r4_cells
from experiments.round5_exp6 import mix
from t1_pipeline.data import board_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    args = parser.parse_args()
    out = REAL / "runs" / "round5_exp10"
    out.mkdir(parents=True, exist_ok=True)
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    anchor, _ = load_anchor(genes)
    future = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    source = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    past = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
    split = split_rows(source, future)
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    h3 = h3_rows(future)
    ref8 = past.dense(np.sort(np.random.default_rng(SEED).choice(past.n_cells, round(.1 * past.n_cells), replace=False)))
    src_ref = source.dense(split["reference"])
    evals = {"p8/deep": FastEval(ref8, future.dense(h3[a]), future.dense(h3[b])),
             "p8/all": FastEval(ref8, future.dense(split[a]), future.dense(split[b])),
             "h3": FastEval(src_ref, future.dense(h3[a]), future.dense(h3[b])),
             "raw": FastEval(src_ref, future.dense(split[a]), future.dense(split[b]))}
    pools = {"all": np.arange(source.n_cells), "cardiac": np.flatnonzero(np.isin(source.labels, list(CARDIAC)))}
    records = []
    for pool_name, pool in pools.items():
        for h in (1, 2):
            cells = r4_cells(source, pool, src_ref, trend, len(src_ref), strength=3.0 * h, bulk=3.0 * h)[0]
            for f in (0, .15):
                pred = mix(cells, anchor, f, np.random.default_rng(SEED))
                for proxy, ev in evals.items():
                    records.append({"pool": pool_name, "h": h, "anchor_fraction": f, "proxy": proxy, "split": args.split, **ev(pred)})
            pd.DataFrame(records).to_csv(out / f"scores_{args.split}.csv", index=False)
            print(pool_name, h, "done", flush=True)
    print(pd.DataFrame(records).pivot_table(index=["pool", "h", "anchor_fraction"], columns="proxy", values="score").round(2).to_string())


if __name__ == "__main__":
    main()
