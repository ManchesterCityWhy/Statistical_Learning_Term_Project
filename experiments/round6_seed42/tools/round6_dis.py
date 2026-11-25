"""Round 6, proxy D: a pure change of dissection, no time gap.

    reference = E9.5 (the 1,706 rows of the round-2 split)
    source    = E10.5 libraries 1+2 only   (surrounding tissue with some heart, shallow)
    target    = E10.5 library 3            (the heart-proper library of the same laboratory and stage)

The test asks for a specifically dissected heart (E12.5) from a source that mixes heart and surrounding tissue.
This proxy isolates that one aspect: how should cells be chosen from a surrounding-rich source when the target
is a heart-proper library of the same laboratory? Blind spots: zero elapsed time, and the source here has no
heart-proper library of its own (at test time it has one, library 3).

    D/raw : libraries as released (source shallow, target deep)
    D/dm  : target thinned to the depth distribution of libraries 1+2 (composition only)

    python -m experiments.round6_dis [--split confirmation]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_common import h3_rows, split_rows
from experiments.round5_exp4 import r4_cells
from experiments.round6_common import (HEART, REAL, SEED, FastEval, Log, Mem, depth_of, lineage_of, take, thin_rows)
from t1_pipeline.data import board_spec

N = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    args = parser.parse_args()
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    t0 = time.time()
    e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    ref = e9.dense(split_rows(e9, e10)["reference"])
    h3 = h3_rows(e10)
    lib12 = np.flatnonzero(e10.lib != "3")
    heart12 = np.intersect1d(lib12, np.flatnonzero(np.isin(e10.labels, list(HEART))))
    shallow = depth_of(e10.X[lib12])
    rng = lambda: np.random.default_rng(SEED)
    targets = {"D/raw": (e10.X[h3[a]], e10.X[h3[b]]),
               "D/dm": (thin_rows(e10.X[h3[a]], shallow, rng()), thin_rows(e10.X[h3[b]], shallow, rng()))}
    lin = lineage_of(e10.labels)
    ref10 = e10.dense(take(e10, lib12, N, rng()))
    cands = {}
    rows = take(e10, lib12, N, rng())
    cands["natural:lib12"] = (e10.dense(rows), rows)
    rows = take(e10, heart12, N, rng())
    cands["heart-natural:lib12"] = (e10.dense(rows), rows)
    for name, pool in (("lib12", lib12), ("heart12", heart12)):
        r4, r2 = r4_cells(e10, pool, ref10, trend, N)
        cands[f"r2:{name}"], cands[f"r4:{name}"] = (r2, None), (r4, None)
    log = Log(REAL / "runs" / "round6_dis" / f"scores_{args.split}.csv")
    for proxy, (target, ceiling) in targets.items():
        ev = FastEval(ref, target.toarray(), ceiling.toarray())
        print(proxy, "ready", f"{time.time() - t0:.0f}s", "floor", ev.floor, "ceiling", ev.ceiling, flush=True)
        for name, (cells, rows) in cands.items():
            r = log.add(ev, cells, proxy=proxy, base=name, split=args.split)
            if rows is not None:
                r.update({f"share_{k}": round(float((lin[rows] == k).mean()), 3) for k in ("CM", "Endo", "Cushion", "Epi", "Blood", "Mac", "Other")})
    frame = pd.DataFrame(log.records)
    frame.to_csv(log.path, index=False)
    print(frame.pivot_table(index="base", columns="proxy", values="score", sort=False).round(2).to_string())
    print("lineage shares of the target (library 3):", {k: round(float((lin[h3[a]] == k).mean()), 3) for k in ("CM", "Endo", "Cushion", "Epi", "Blood", "Mac", "Other")})


if __name__ == "__main__":
    main()
