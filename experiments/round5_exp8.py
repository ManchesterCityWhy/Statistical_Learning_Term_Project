"""Round 5, batch 8: does the anchor fraction scale with the time gap? Two-day analogue E8.5 -> E10.5.

E10.5 lies one third of the way from E8.5 to E14.5 (one fifth from E9.5). If the best mixing fraction follows
elapsed time, it should move from about 0.15 (one-day proxies) to about 0.25 here. Reference = source = E8.5.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import REAL, SEED, FastEval, Mem, h3_rows, split_rows
from experiments.round5_exp4 import r4_cells
from experiments.round5_exp6 import mix
from t1_pipeline.data import board_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    args = parser.parse_args()
    out = REAL / "runs" / "round5_exp8"
    out.mkdir(parents=True, exist_ok=True)
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    anchor, _ = load_anchor(genes)
    t0 = time.time()
    future = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    split = split_rows(e9, future)
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    h3 = h3_rows(future)
    records = []
    for tag, path in (("raw", "data/T1/E8.5_RNA.h5ad"), ("dm", "data/proxy/E8.5_RNA.h5ad")):
        source = Mem(REAL / path, genes)
        rows = np.sort(np.random.default_rng(SEED).choice(source.n_cells, round(.1 * source.n_cells), replace=False))
        ref = source.dense(rows)
        evals = {"all": FastEval(ref, future.dense(split[a]), future.dense(split[b]))}
        if tag == "raw":
            evals["deep"] = FastEval(ref, future.dense(h3[a]), future.dense(h3[b]))
        bases = {"unchanged": ref}
        for h in (1, 2):
            bases[f"r4_h{h}"] = r4_cells(source, np.arange(source.n_cells), ref, trend, len(ref), strength=3.0 * h, bulk=3.0 * h)[0]
        print(tag, "ready", f"{time.time() - t0:.0f}s", flush=True)
        for target, ev in evals.items():
            for base_name, cells in bases.items():
                for f in (0, .1, .15, .2, .25, .3, .35, .4, .5):
                    r = {"proxy": f"{tag}/{target}", "base": base_name, "anchor_fraction": f, "split": args.split,
                         **ev(mix(cells, anchor, f, np.random.default_rng(SEED)))}
                    records.append(r)
                pd.DataFrame(records).to_csv(out / f"scores_{args.split}.csv", index=False)
                print(tag, target, base_name, "done", flush=True)
    frame = pd.DataFrame(records)
    print(frame.pivot_table(index=["base", "anchor_fraction"], columns="proxy", values="score").round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(frame[frame.base == "r4_h1"].pivot_table(index="anchor_fraction", columns="proxy", values=m).round(3).to_string())


if __name__ == "__main__":
    main()
