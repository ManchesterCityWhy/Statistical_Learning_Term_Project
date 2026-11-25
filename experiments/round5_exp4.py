"""Round 5, batch 4: what is the choice of source pool worth? A depth payoff table on a synthetic mixed-depth E9.5.

At test time the source (E10.5) is a mix of one deep heart-proper library and two shallow libraries, and the
depth of E12.5 is unknown. To price that decision locally we rebuild the same situation one stage earlier:
a random 53% of the E9.5 cells are thinned to the depth distribution of E10.5 libraries 1+2 ("shallow"), the
rest stay as released ("deep"). The reference is the usual 1,706 E9.5 rows taken from this mixed stage.
Each candidate is then scored against three possible targets: E10.5 library 3 only (deep), all of E10.5
(mixed) and E10.5 libraries 1+2 only (shallow).

    python -m experiments.round5_exp4
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round4_direction import impose
from experiments.round5_common import REAL, SEED, FastEval, Mem, pb, split_rows
from experiments.round5_exp2 import deconv_q, profiles_of
from t1_pipeline.data import board_spec

R4_W = {"bulk": -0.48592635027583775, "deconv": 0.8311397285098777}


def depth_of(x):
    x = x.tocsr()
    return 1e4 / np.expm1(np.minimum.reduceat(x.data, x.indptr[:-1]))


def thin_rows(x, target_depths, rng, batch=1024):
    """Binomially thin the UMI counts of every row of CSR x to quantile-matched target depths."""
    depth = depth_of(x)
    quantile = (np.argsort(np.argsort(depth)) + .5) / len(depth)
    keep = np.minimum(1.0, np.quantile(target_depths, quantile) / depth)
    blocks = []
    for s in range(0, x.shape[0], batch):
        dense = np.expm1(x[s:s + batch].toarray().astype(np.float64)) * (depth[s:s + batch, None] / 1e4)
        thinned = rng.binomial(np.rint(dense).astype(np.int64), keep[s:s + batch, None]).astype(np.float64)
        total = np.maximum(thinned.sum(axis=1, keepdims=True), 1)
        blocks.append(sparse.csr_matrix(np.log1p(thinned / total * 1e4).astype(np.float32)))
    return sparse.vstack(blocks).tocsr()


def r4_cells(source, pool, ref, trend, n, strength=3.0, bulk=3.0):
    """Round-4 prediction with the deconvolution and the sampling restricted to the rows in `pool`."""
    labels = source.labels[pool]
    names = np.unique(labels)
    prof = np.stack([np.asarray(source.X[pool[labels == g]].expm1().mean(axis=0)).ravel() for g in names], axis=1)
    p = np.array([(labels == g).mean() for g in names])
    q = deconv_q(prof, p, trend, strength, 10)
    w = np.zeros(len(pool))
    for g, prob in zip(names, q):
        w[labels == g] = prob / (labels == g).sum()
    rows = pool[np.random.default_rng(SEED).choice(len(pool), n, replace=False, p=w / w.sum())]
    base = apply_trend(source.dense(rows), trend, bulk / 5, False, SEED)
    ref_pb = pb(ref)
    feats = {"bulk": pb(apply_trend(ref.copy(), trend, bulk / 5, False, SEED)) - ref_pb, "deconv": pb(base) - ref_pb}
    return impose(base, ref_pb + sum(wt * feats[k] for k, wt in R4_W.items()), 1.0), base


def main():
    out = REAL / "runs" / "round5_exp4"
    out.mkdir(parents=True, exist_ok=True)
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    t0 = time.time()
    future = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    source = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    split = split_rows(source, future)
    rng = np.random.default_rng(SEED)
    shallow_target = depth_of(future.X[np.flatnonzero(future.lib != "3")])
    shallow = np.sort(rng.choice(source.n_cells, round(.53 * source.n_cells), replace=False))
    x = source.X.tolil()
    x[shallow] = thin_rows(source.X[shallow], shallow_target, rng)
    source.X = x.tocsr()
    is_deep = np.ones(source.n_cells, bool)
    is_deep[shallow] = False
    print("mixed E9.5 built", f"{time.time() - t0:.0f}s", "median depth deep/shallow",
          np.median(depth_of(source.X[np.flatnonzero(is_deep)])), np.median(depth_of(source.X[shallow])), flush=True)
    ref = source.dense(split["reference"])
    n = len(ref)
    order3 = rng.permutation(np.flatnonzero(future.lib == "3"))
    order12 = rng.permutation(np.flatnonzero(future.lib != "3"))
    targets = {"deep(lib3)": (order3[:1249], order3[1249:2498]), "mixed(all)": (split["development"], split["ceiling"]),
               "shallow(lib1+2)": (order12[:1249], order12[1249:2498])}
    evals = {k: FastEval(ref, future.dense(a), future.dense(b)) for k, (a, b) in targets.items()}
    deep_rows, all_rows = np.flatnonzero(is_deep), np.arange(source.n_cells)
    cands = {"copy_last": ref,
             "random_deep": source.dense(np.sort(rng.choice(deep_rows, n, replace=False))),
             "random_shallow": source.dense(np.sort(rng.choice(shallow, n, replace=False)))}
    for pool_name, pool in (("all", all_rows), ("deep", deep_rows), ("shallow", shallow)):
        r4, base = r4_cells(source, pool, ref, trend, n)
        cands[f"r4_pool_{pool_name}"] = r4
        cands[f"r2_pool_{pool_name}"] = base
    records = []
    for name, cells in cands.items():
        for target, ev in evals.items():
            r = {"candidate": name, "target": target, **ev(cells)}
            records.append(r)
            print(f"{name:<18} {target:<16} {r['score']:6.2f}  DES {r['skill_de_score']:.3f} DCS {r['skill_de_direction']:.3f} "
                  f"MMD {r['skill_mmd_u']:.3f} CSS {r['skill_variogram']:.3f}", flush=True)
        pd.DataFrame(records).to_csv(out / "scores.csv", index=False)
    print(pd.DataFrame(records).pivot_table(index="candidate", columns="target", values="score").round(2).to_string())


if __name__ == "__main__":
    main()
