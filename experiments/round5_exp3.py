"""Round 5, batch 3: gene-wise direction features at vector level (DES / DCS only, development split).

Adds the within-lineage change of the previous day (composition-free 'internal' trend) to the round-4
features and checks whether weights fitted on one proxy carry over to the others.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round4_direction import deconv_rows, fit_weights
from experiments.round5_common import REAL, SEED, FastEval, Mem, coarse, h3_rows, pb, split_rows
from t1_pipeline.data import board_spec

LINEAGES = ("CM", "Endo", "Epi", "Blood", "Other")


def lineage_internal(past, source, source_rows=None, shares=None):
    """Sum over lineages of share_k x (mean of lineage k now - mean of lineage k one stage earlier)."""
    cp, cs = coarse(past.labels), coarse(source.labels)
    cp[past.labels == "JCF"] = "Epi"
    use = np.ones(source.n_cells, bool) if source_rows is None else np.isin(np.arange(source.n_cells), source_rows)
    total, parts = 0.0, {}
    for k in LINEAGES:
        a, b = np.flatnonzero(cp == k), np.flatnonzero((cs == k) & use)
        if len(a) < 50 or len(b) < 50:
            continue
        parts[k] = source.mean(b) - past.mean(a)
        share = (cs == k).mean() if shares is None else shares.get(k, 0.0)
        total = total + share * parts[k]
    return total, parts


def main():
    out = REAL / "runs" / "round5_exp3"
    out.mkdir(parents=True, exist_ok=True)
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    t0 = time.time()
    future = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    stages = {"raw": (Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes), Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)),
              "dm": (Mem(REAL / "data/proxy/E8.5_RNA.h5ad", genes), Mem(REAL / "data/proxy/E9.5_RNA.h5ad", genes))}
    stages["h3"] = stages["raw"]
    split = split_rows(stages["raw"][1], future)
    targets = {"raw": split, "dm": split, "h3": h3_rows(future)}
    ctx = {}
    for proxy in ("dm", "raw", "h3"):
        past, source = stages[proxy]
        ref = source.dense(split["reference"])
        ref_pb = pb(ref)
        ev = FastEval(ref, future.dense(targets[proxy]["development"]), future.dense(targets[proxy]["ceiling"]))
        rows = deconv_rows(source, trend, 3, len(ref), np.random.default_rng(SEED))
        base = apply_trend(source.dense(rows), trend, 3 / 5, False, SEED)
        lin, parts = lineage_internal(past, source)
        feats = {"bulk": pb(apply_trend(ref.copy(), trend, 3 / 5, False, SEED)) - ref_pb,
                 "bulk1": pb(apply_trend(ref.copy(), trend, 1 / 5, False, SEED)) - ref_pb,
                 "deconv": pb(base) - ref_pb, "internal": source.mean() - past.mean(), "lin": lin,
                 **{f"lin_{k}": v for k, v in parts.items()}, "level": ref_pb, "trend": trend.astype(float)}
        ctx[proxy] = {"ev": ev, "feats": feats, "true": ev.dt.astype(float)}
        print(proxy, "ready", f"{time.time() - t0:.0f}s", flush=True)
    records = []
    for proxy, c in ctx.items():
        for name, v in c["feats"].items():
            des, dcs = c["ev"].direction(v)
            records.append({"kind": "feature", "name": name, "proxy": proxy, "des": des, "dcs": dcs})
    combos = [("bulk", "deconv"), ("bulk", "deconv", "lin"), ("bulk", "deconv", "internal"),
              ("bulk", "deconv", "lin_CM", "lin_Endo", "lin_Other"), ("bulk", "deconv", "lin", "level"),
              ("deconv",), ("deconv", "lin"), ("bulk", "lin"), ("bulk", "deconv", "bulk1")]
    for combo, fit_on in itertools.product(combos, ("dm", "raw", "h3")):
        if any(k not in ctx[fit_on]["feats"] for k in combo):
            continue
        w = fit_weights([ctx[fit_on]["feats"][k] for k in combo], ctx[fit_on]["true"])
        for proxy, c in ctx.items():
            vec = sum(wi * c["feats"][k] for k, wi in zip(combo, w))
            des, dcs = c["ev"].direction(vec)
            records.append({"kind": "fit", "name": "+".join(combo), "fit_on": fit_on, "proxy": proxy, "des": des,
                            "dcs": dcs, "weights": " ".join(f"{k}={wi:.3f}" for k, wi in zip(combo, w))})
    frame = pd.DataFrame(records)
    frame.to_csv(out / "direction.csv", index=False)
    pd.set_option("display.width", 250)
    print(frame[frame.kind == "feature"].pivot_table(index="name", columns="proxy", values=["des", "dcs"]).round(3).to_string())
    fit = frame[frame.kind == "fit"]
    print(fit.pivot_table(index=["name", "fit_on"], columns="proxy", values=["des", "dcs"]).round(3).to_string())
    print(fit.drop_duplicates(["name", "fit_on"])[["name", "fit_on", "weights"]].to_string())
    print("ceilings", {p: c["ev"].ceiling for p, c in ctx.items()})


if __name__ == "__main__":
    main()
