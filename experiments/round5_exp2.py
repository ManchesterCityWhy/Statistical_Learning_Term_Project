"""Round 5, batch 2: finer cell states in the bulk-guided deconvolution (development split, three proxies).

Round 2 re-weights ~21 annotated types. Here the same NNLS deconvolution runs on k-means cell states
(optionally nested inside the annotated types), with the ridge and the trend strength varied.

    python -m experiments.round5_exp2
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round4_direction import impose
from experiments.round5_common import REAL, SEED, FastEval, Mem, h3_rows, pb, split_rows
from t1_pipeline.data import board_spec

R4_W = {"bulk": -0.48592635027583775, "deconv": 0.8311397285098777}


def state_space(stage, seed=SEED, n_hvg=2000, n_pc=30):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    fit = stage.dense(np.sort(rng.choice(stage.n_cells, min(5000, stage.n_cells), replace=False)))
    genes = np.argsort(fit.var(axis=0))[-n_hvg:]
    pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=seed).fit(fit[:, genes])
    return np.vstack([pca.transform(stage.dense(r)[:, genes])
                      for r in np.array_split(np.arange(stage.n_cells), stage.n_cells // 2048 + 1)])


def groupings(stage, latent, seed=SEED):
    from sklearn.cluster import KMeans
    out = {"types": stage.labels}
    for k in (30, 60, 120):
        out[f"km{k}"] = KMeans(k, n_init=3, random_state=seed).fit_predict(latent).astype(str)
    nested = np.empty(stage.n_cells, dtype=object)
    for lab in np.unique(stage.labels):
        ids = np.flatnonzero(stage.labels == lab)
        k = int(np.clip(len(ids) // 250, 1, 6))
        sub = KMeans(k, n_init=3, random_state=seed).fit_predict(latent[ids]) if k > 1 else np.zeros(len(ids), int)
        nested[ids] = [f"{lab}#{s}" for s in sub]
    out["nested"] = nested.astype(str)
    return out


def profiles_of(stage, groups):
    names = np.unique(groups)
    prof = np.stack([np.asarray(stage.X[np.flatnonzero(groups == g)].expm1().mean(axis=0)).ravel() for g in names], axis=1)
    return names, prof, np.array([(groups == g).mean() for g in names])


def deconv_q(prof, p, trend, strength, ridge):
    base = prof @ p
    keep = (base > .1) & (np.std(prof, axis=1) > .1)
    design = prof[keep] / np.sqrt(base[keep, None] + 1)
    desired = base * np.exp(np.clip(trend * strength, -2, 2))
    desired *= base.sum() / desired.sum()
    q = nnls(np.vstack([design, ridge * np.eye(len(p))]), np.r_[desired[keep] / np.sqrt(base[keep] + 1), ridge * p])[0]
    return q / q.sum()


def sample(groups, names, q, n, rng):
    w = np.zeros(len(groups))
    for g, prob in zip(names, q):
        w[groups == g] = prob / (groups == g).sum()
    return rng.choice(len(groups), n, replace=False, p=w / w.sum())


def main():
    out = REAL / "runs" / "round5_exp2"
    out.mkdir(parents=True, exist_ok=True)
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    t0 = time.time()
    future = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    sources = {"raw": Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes), "dm": Mem(REAL / "data/proxy/E9.5_RNA.h5ad", genes)}
    split = split_rows(sources["raw"], future)
    targets = {"raw": split, "dm": split, "h3": h3_rows(future)}
    ctx = {}
    for proxy in ("dm", "raw", "h3"):
        source = sources["raw" if proxy == "h3" else proxy]
        ref = source.dense(split["reference"])
        tg = targets[proxy]
        ev = FastEval(ref, future.dense(tg["development"]), future.dense(tg["ceiling"]))
        if proxy == "h3":
            groups = ctx["raw"]["groups"]
        else:
            groups = {k: profiles_of(source, g) + (g,) for k, g in groupings(source, state_space(source)).items()}
        ctx[proxy] = {"source": source, "ref": ref, "ev": ev, "groups": groups,
                      "bulk": pb(apply_trend(ref.copy(), trend, 3 / 5, False, SEED)) - pb(ref)}
        print(proxy, "ready", f"{time.time() - t0:.0f}s", flush=True)
    records = []
    grid = list(itertools.product(("types", "km30", "km60", "km120", "nested"), (2, 3, 4, 6), (3, 10, 30)))
    for grouping, strength, ridge in grid:
        for proxy, c in ctx.items():
            names, prof, p, g = c["groups"][grouping]
            q = deconv_q(prof, p, trend, strength, ridge)
            rows = sample(g, names, q, len(c["ref"]), np.random.default_rng(SEED))
            base = apply_trend(c["source"].dense(rows), trend, 3 / 5, False, SEED)
            ref_pb = pb(c["ref"])
            r4 = impose(base, ref_pb + R4_W["bulk"] * c["bulk"] + R4_W["deconv"] * (pb(base) - ref_pb), 1.0)
            ess = 1 / (len(q) * np.sum((q / p) ** 2 * p) ) if (p > 0).all() else np.nan
            for variant, cells in (("base", base), ("r4", r4)):
                r = {"proxy": proxy, "grouping": grouping, "strength": strength, "ridge": ridge, "variant": variant,
                     **c["ev"](cells), "max_q": float(q.max()), "n_active": int((q > 1e-4).sum())}
                records.append(r)
            pd.DataFrame(records).to_csv(out / "scores.csv", index=False)
        last = pd.DataFrame(records[-6:])
        print(f"{grouping:>7} s{strength} r{ridge:<3}", " ".join(
            f"{x.proxy}/{x.variant}:{x.score:5.2f}" for x in last.itertuples()), f"({time.time() - t0:.0f}s)", flush=True)
    frame = pd.DataFrame(records)
    table = frame.pivot_table(index=["grouping", "strength", "ridge"], columns=["variant", "proxy"], values="score")
    table["mean_r4"] = table["r4"].mean(axis=1)
    print(table.sort_values("mean_r4", ascending=False).round(2).head(25).to_string())


if __name__ == "__main__":
    main()
