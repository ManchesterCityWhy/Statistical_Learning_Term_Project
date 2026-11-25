"""Round 5, batch 6: mixing in real E14.5 Multiome nuclei (GSE272154 wild type) as a future anchor.

Linear time puts E10.5 one fifth of the way from E9.5 to E14.5, so the natural mixing fraction locally is 0.2;
on the test (E10.5 -> E12.5 with E14.5 ahead) it would be 0.5. Scored on
    p8/deep, p8/all : reference E8.5, source E9.5 as released, target E10.5 library 3 / all E10.5
    dm              : reference and source = depth-matched E9.5, target all E10.5 (the round-4 selection proxy)
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
from t1_pipeline.data import board_spec


def mix(cells, anchor, fraction, rng):
    k = round(fraction * len(cells))
    if k == 0:
        return cells
    take = np.sort(rng.choice(anchor.shape[0], k, replace=False))
    keep = np.sort(rng.choice(len(cells), len(cells) - k, replace=False))
    return np.vstack([cells[keep], anchor[take].toarray().astype(np.float32)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--fractions", default="0,0.1,0.2,0.3,0.5")
    parser.add_argument("--min-umi", type=int, default=1000)
    parser.add_argument("--desoup", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    fractions = [float(v) for v in args.fractions.split(",")]
    out = REAL / "runs" / "round5_exp6"
    out.mkdir(parents=True, exist_ok=True)
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    anchor, info = load_anchor(genes, min_umi=args.min_umi, desoup=args.desoup)
    t0 = time.time()
    future = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    source = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    source_dm = Mem(REAL / "data/proxy/E9.5_RNA.h5ad", genes)
    past = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
    split = split_rows(source, future)
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    h3 = h3_rows(future)
    ref8 = past.dense(np.sort(np.random.default_rng(SEED).choice(past.n_cells, round(.1 * past.n_cells), replace=False)))
    src_ref, dm_ref = source.dense(split["reference"]), source_dm.dense(split["reference"])
    evals = {"p8/deep": FastEval(ref8, future.dense(h3[a]), future.dense(h3[b])),
             "p8/all": FastEval(ref8, future.dense(split[a]), future.dense(split[b])),
             "h3": FastEval(src_ref, future.dense(h3[a]), future.dense(h3[b])),
             "raw": FastEval(src_ref, future.dense(split[a]), future.dense(split[b])),
             "dm": FastEval(dm_ref, future.dense(split[a]), future.dense(split[b]))}
    if args.only:
        evals = {k: v for k, v in evals.items() if k in args.only.split(",")}
    n = len(src_ref)
    rows_all = np.arange(source.n_cells)
    bases = {"raw": {"unchanged": src_ref, "r4": r4_cells(source, rows_all, src_ref, trend, n)[0]},
             "dm": {"unchanged": dm_ref, "r4": r4_cells(source_dm, rows_all, dm_ref, trend, n)[0]}}
    print("ready", f"{time.time() - t0:.0f}s", flush=True)
    records = []
    for proxy, ev in evals.items():
        for base_name, cells in bases["dm" if proxy == "dm" else "raw"].items():
            for f in fractions:
                pred = mix(cells, anchor, f, np.random.default_rng(SEED))
                r = {"anchor": args.tag, "proxy": proxy, "base": base_name, "anchor_fraction": f, "split": args.split, **ev(pred)}
                records.append(r)
                print(f"{proxy:<8} {base_name:<10} f={f:<4} {r['score']:6.2f}  DES {r['skill_de_score']:.3f} DCS {r['skill_de_direction']:.3f} "
                      f"MMD {r['skill_mmd_u']:.3f} CSS {r['skill_variogram']:.3f}", flush=True)
            pd.DataFrame(records).to_csv(out / f"scores_{args.split}{args.tag}.csv", index=False)
    print(pd.DataFrame(records).pivot_table(index=["base", "anchor_fraction"], columns="proxy", values="score").round(2).to_string())


if __name__ == "__main__":
    main()
