"""Round 5, batch 5: a proxy with the reference two stages back, as on the test board.

Public reports say that E10.5 cells submitted unchanged score about 69 on T1:test, so the board's floor
(copy_last = 50) must be E9.5, the last *training* stage, and DES/DCS measure the change from E9.5 to E12.5.
The local analogue: reference = E8.5, known source = E9.5, target = E10.5. A method is worth what it adds
over "submit the source cells unchanged".

Targets: E10.5 library 3 (deep heart-proper), all of E10.5, libraries 1+2 (shallow).
Sources: E9.5 as released (deep), and a synthetic mixed-depth E9.5 (53% of cells thinned to the library-1+2
depth distribution) to price the choice of pool when the source mixes deep and shallow libraries.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round5_common import REAL, SEED, FastEval, Mem, pb, split_rows
from experiments.round5_exp4 import depth_of, r4_cells, thin_rows
from t1_pipeline.data import board_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    args = parser.parse_args()
    out = REAL / "runs" / "round5_exp5"
    out.mkdir(parents=True, exist_ok=True)
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    t0 = time.time()
    future = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    source = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    past = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
    split = split_rows(source, future)
    rng = np.random.default_rng(SEED)
    ref = past.dense(np.sort(rng.choice(past.n_cells, round(.1 * past.n_cells), replace=False)))
    shallow_target = depth_of(future.X[np.flatnonzero(future.lib != "3")])
    shallow = np.sort(rng.choice(source.n_cells, round(.53 * source.n_cells), replace=False))
    mixed = copy.copy(source)
    x = source.X.tolil()
    x[shallow] = thin_rows(source.X[shallow], shallow_target, rng)
    mixed.X = x.tocsr()
    deep_rows = np.setdiff1d(np.arange(source.n_cells), shallow)
    order3 = rng.permutation(np.flatnonzero(future.lib == "3"))
    order12 = rng.permutation(np.flatnonzero(future.lib != "3"))
    k = 0 if args.split == "development" else 2
    names = ("development", "ceiling") if k == 0 else ("confirmation", "confirmation_ceiling")
    cut = lambda o: (o[k * 1249:(k + 1) * 1249], o[(k + 1) * 1249:(k + 2) * 1249])
    targets = {"deep(lib3)": cut(order3), "mixed(all)": (split[names[0]], split[names[1]]), "shallow(lib1+2)": cut(order12)}
    evals = {key: FastEval(ref, future.dense(a), future.dense(b)) for key, (a, b) in targets.items()}
    print("ready", f"{time.time() - t0:.0f}s", flush=True)
    n = len(split["reference"])
    src_ref, mix_ref = source.dense(split["reference"]), mixed.dense(split["reference"])
    all_rows = np.arange(source.n_cells)
    cands = {"copy_last(E8.5)": ref, "source_unchanged": src_ref}
    for s in (1.5, 3.0, 6.0):
        r4, base = r4_cells(source, all_rows, src_ref, trend, n, strength=s, bulk=3.0)
        cands[f"r2_s{s}"], cands[f"r4_s{s}"] = base, r4
    cands["bulk3_only"] = apply_trend(src_ref.copy(), trend, 3 / 5, False, SEED)
    cands["MIX:source_unchanged"] = mix_ref
    cands["MIX:deep_cells_only"] = mixed.dense(np.sort(rng.choice(deep_rows, n, replace=False)))
    cands["MIX:shallow_cells_only"] = mixed.dense(np.sort(rng.choice(shallow, n, replace=False)))
    for pool_name, pool in (("all", all_rows), ("deep", deep_rows)):
        r4, base = r4_cells(mixed, pool, mix_ref, trend, n)
        cands[f"MIX:r4_pool_{pool_name}"], cands[f"MIX:r2_pool_{pool_name}"] = r4, base
    records = []
    for name, cells in cands.items():
        for target, ev in evals.items():
            r = {"candidate": name, "target": target, "split": args.split, **ev(cells)}
            records.append(r)
            print(f"{name:<24} {target:<16} {r['score']:6.2f}  DES {r['skill_de_score']:.3f} DCS {r['skill_de_direction']:.3f} "
                  f"MMD {r['skill_mmd_u']:.3f} CSS {r['skill_variogram']:.3f}", flush=True)
        pd.DataFrame(records).to_csv(out / f"scores_{args.split}.csv", index=False)
    print(pd.DataFrame(records).pivot_table(index="candidate", columns="target", values="score", sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
