"""Round 5, batch 1: does 'keep the heart-proper cells' / a maturity tilt beat round 4, and on which proxy?

    python -m experiments.round5_exp1 [--confirm name1,name2]

Development split only unless --confirm lists candidate names (then those are also scored on confirmation).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round4_direction import deconv_rows, impose
from experiments.round5_common import (CARDIAC, REAL, SEED, FastEval, Mem, coarse, h3_rows, pb, split_rows)
from experiments.round3_common import weighted_sample_without_replacement
from t1_pipeline.data import board_spec

R4_W = {"bulk": -0.48592635027583775, "deconv": 0.8311397285098777}


def maturity_logit(past, source, seed=SEED, n_hvg=3000, n_pc=50):
    """Cross-fitted logit of P(cell is from `source` rather than `past`) for every source cell."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(seed)
    fit = np.vstack([past.dense(np.sort(rng.choice(past.n_cells, 3000, replace=False))),
                     source.dense(np.sort(rng.choice(source.n_cells, 3000, replace=False)))])
    genes = np.argsort(fit.var(axis=0))[-n_hvg:]
    pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=seed).fit(fit[:, genes])
    project = lambda st: np.vstack([pca.transform(st.dense(r)[:, genes])
                                    for r in np.array_split(np.arange(st.n_cells), st.n_cells // 2048 + 1)])
    zp, zs = project(past), project(source)
    scale = zs.std(axis=0)
    zp, zs = zp / scale, zs / scale
    logit = np.zeros(source.n_cells)
    folds_s = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(zs, np.zeros(len(zs))))
    folds_p = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(zp, np.zeros(len(zp))))
    for (tr_s, te_s), (tr_p, _) in zip(folds_s, folds_p):
        x = np.vstack([zp[tr_p], zs[tr_s]])
        y = np.r_[np.zeros(len(tr_p)), np.ones(len(tr_s))]
        model = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced").fit(x, y)
        logit[te_s] = model.decision_function(zs[te_s])
    return logit


def candidates(past, source, ref, trend, n):
    """Yield (name, cells). Everything is computed from the source stage, the stage before it and the bulk trend."""
    rng = lambda: np.random.default_rng(SEED)
    ref_pb = pb(ref)
    yield "copy_last", ref
    rows = deconv_rows(source, trend, 3, n, rng())
    base = apply_trend(source.dense(rows), trend, 3 / 5, False, SEED)
    yield "r2_deconv3_bulk3", base
    feats = {"bulk": pb(apply_trend(ref.copy(), trend, 3 / 5, False, SEED)) - ref_pb, "deconv": pb(base) - ref_pb}
    yield "r4_bulk+deconv", impose(base, ref_pb + sum(w * feats[k] for k, w in R4_W.items()), 1.0)

    cardiac = np.flatnonzero(np.isin(source.labels, list(CARDIAC)))
    core_rows = np.sort(rng().choice(cardiac, n, replace=False))
    core = source.dense(core_rows)
    yield "core", core
    yield "core_bulk1", apply_trend(core, trend, 1 / 5, False, SEED)
    yield "core_bulk3", apply_trend(core, trend, 3 / 5, False, SEED)
    half = np.sort(rng().choice(source.n_cells, n // 2, replace=False))
    yield "core50_all50", np.vstack([core[: n - len(half)], source.dense(half)])

    # coarse-class share extrapolation: share(t+1) ~ share(t) * (share(t)/share(t-1))^beta
    cs, cp = coarse(source.labels), coarse(past.labels)
    for beta in (0.5, 1.0):
        w = np.zeros(source.n_cells)
        for k in np.unique(cs):
            now, before = (cs == k).mean(), max((cp == k).mean(), 0.005)
            w[cs == k] = (now / before) ** beta
        yield f"coarse_extrap_b{beta}", source.dense(weighted_sample_without_replacement(w, n, rng()))

    logit = np.clip(maturity_logit(past, source), -4, 4)
    for beta in (0.5, 1.0):
        w = np.exp(beta * logit)
        yield f"tilt_all_b{beta}", source.dense(weighted_sample_without_replacement(w, n, rng()))
        wc = np.where(np.isin(source.labels, list(CARDIAC)), w, 0.0)
        yield f"tilt_core_b{beta}", source.dense(weighted_sample_without_replacement(wc, n, rng()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", default="")
    parser.add_argument("--output", default="round5_exp1")
    args = parser.parse_args()
    confirm = set(filter(None, args.confirm.split(",")))
    out = REAL / "runs" / args.output
    out.mkdir(parents=True, exist_ok=True)
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    t0 = time.time()
    future = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    stages = {"raw": (Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes), Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)),
              "dm": (Mem(REAL / "data/proxy/E8.5_RNA.h5ad", genes), Mem(REAL / "data/proxy/E9.5_RNA.h5ad", genes))}
    stages["h3"] = stages["raw"]
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    split = split_rows(stages["raw"][1], future)
    targets = {"raw": split, "dm": split, "h3": h3_rows(future)}
    records = []
    for proxy in ("dm", "raw", "h3"):
        past, source = stages[proxy]
        ref = source.dense(split["reference"])
        tg = targets[proxy]
        evals = {"development": FastEval(ref, future.dense(tg["development"]), future.dense(tg["ceiling"]))}
        if confirm:
            evals["confirmation"] = FastEval(ref, future.dense(tg["confirmation"]),
                                             future.dense(tg["confirmation_ceiling"]))
        print(proxy, "floor", evals["development"].floor, "ceiling", evals["development"].ceiling, flush=True)
        for name, cells in candidates(past, source, ref, trend, len(ref)):
            if name == "r2_deconv3_bulk3" and proxy == "dm":
                print("check official vs fast", evals["development"].check(cells), flush=True)
            for split_name, ev in evals.items():
                if split_name == "confirmation" and name not in confirm:
                    continue
                r = {"proxy": proxy, "candidate": name, "split": split_name, **ev(cells),
                     "nonzero": float((cells > 0).mean())}
                records.append(r)
                pd.DataFrame(records).to_csv(out / "scores.csv", index=False)
                print(f"{proxy:>4} {split_name[:4]} {name:<22} {r['score']:6.2f}  DES {r['skill_de_score']:.3f} "
                      f"DCS {r['skill_de_direction']:.3f} MMD {r['skill_mmd_u']:.3f} CSS {r['skill_variogram']:.3f}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
