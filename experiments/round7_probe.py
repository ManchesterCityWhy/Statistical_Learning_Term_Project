"""Round 7: choosing among candidate files with the returned scores of the first official submission.

The only feedback from the hidden E12.5 target is the per-metric score of the first submission ("probe"):
h1 scored DES 66.3, DCS 76.9, MMD 69.0, CSS 70.5 on T1:test. Every candidate file is a hypothesis about what the
E12.5 population looks like. If a candidate K were the truth, the probe would score

    s_hat(probe | K) = official panel(probe; reference = E9.5 sample, target half = 638 cells of K,
                                      ceiling half = 638 other cells of K)

A candidate under which the probe would have scored what it really scored is consistent with the feedback; one
under which it would have scored something very different is not. Consistency distance:

    D(K) = sum_m w_m (s_hat_m(probe | K) - s_obs_m)^2        (w = the official metric weights)

This uses the returned scores to choose among predictions; no candidate is built from them (rules, section 10).

Does the rule pick good files? `calib` answers that where the truth is known:

    zero-gap   reference E9.5, source = E10.5 without the target cells, truth = E10.5 library 3 half at three
               depths or all libraries; probe = the h1 method re-run on the source pool
    p8*        reference E8.5, source = E9.5 rebuilt with the test's library structure (a deep heart-enriched
               pseudo-library "3" and a shallow surrounding-tissue pseudo-library "1"), truth = E10.5 library 3
               half (deep) or all E10.5 libraries; probe = the h1 method re-run on that source

For every candidate recipe: its actual score against the truth, and D from the probe's actual score. Reported:
rank correlation between D and the actual score (negative = good), and the regret of taking the candidate with the
smallest D.

    python -m experiments.round7_probe calib --setting zero|p8s [--split confirmation]
    python -m experiments.round7_probe test
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd

from experiments.bulk_trend import apply_trend, load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import h3_rows, split_rows
from experiments.round5_exp2 import deconv_q
from experiments.round6_common import HEART, LINEAGES, REAL, SEED, FastEval, Mem, depth_of, lineage_of, thin_rows
from experiments.round6_config import heart_libraries
from experiments.round6_method import lineage_shares
from experiments.round6_s1 import feasible_shares
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.data import board_spec

W = {"skill_de_score": .25, "skill_de_direction": .25, "skill_mmd_u": .30, "skill_variogram": .20}
H1_OBSERVED = {"skill_de_score": .663, "skill_de_direction": .769, "skill_mmd_u": .690, "skill_variogram": .705}
N_PROBE = 2000


def h1_like(source, pool, trend, n, strength=3.0, bulk=3.0, ridge=10):
    """The method of the first submission (round-2 deconvolution + within-cell bulk trend) on a source pool.

    Same statements as round5_exp4.r4_cells up to its `base`; also returns the sampled rows."""
    pool = np.asarray(pool)
    labels = source.labels[pool]
    names = np.unique(labels)
    prof = np.stack([np.asarray(source.X[pool[labels == g]].expm1().mean(axis=0)).ravel() for g in names], axis=1)
    p = np.array([(labels == g).mean() for g in names])
    q = deconv_q(prof, p, trend, strength, ridge)
    w = np.zeros(len(pool))
    for g, prob in zip(names, q):
        w[labels == g] = prob / (labels == g).sum()
    rows = pool[np.random.default_rng(SEED).choice(len(pool), n, replace=False, p=w / w.sum())]
    return apply_trend(source.dense(rows), trend, bulk / 5, False, SEED), rows


def R(name, base="heartlib", blood=1.0, f=0.0, form="raw", fwd=False, tc=None, pos=None, level="lineage", dev=0.0):
    return dict(name=name, base=base, blood=blood, f=f, form=form, fwd=fwd, tc=tc, pos=pos, level=level, dev=dev)


def recipes(which="v1"):
    """Candidate recipes. f = anchor share of the file; tc = position of the TOTAL lineage composition on the
    source -> anchor line (None: base cells in their natural composition, so the total sits at tc = f)."""
    if which == "grid":
        # one depth-clean family: heart library + f anchor nuclei re-sequenced at the source depth, harmonised, and
        # both parts placed at position pos on the source -> anchor path (base moved forward by pos, anchor moved
        # back by 1 - pos). pos = 0: every lineage mean is the source's; the anchor then only changes the
        # composition and the within-lineage variety.
        out = [R(f"f{f:g} pos{pos:g}", f=f, form="resback", pos=pos) for f in (0.0, .15, .3, .45) for pos in (0.0, .25, .5)]
        # reference points outside the family (never selected by the rule)
        out += [R("ref round 6 file (0.2 raw)", f=.2), R("ref round 6 first version (0.375 raw)", f=.375)]
        return out
    if which == "grid2":
        # credible directions only: anchor nuclei re-sequenced and moved back completely, by cell type (pos = 0), so the
        # anchor changes the composition and the within-type variety; lineage means move only by the two-witness
        # developmental shift (dev = share of the source -> anchor path)
        out = [R(f"f{f:g} dev{d:g}", f=f, form="resback", pos=0.0, level="type", dev=d) for f in (.3, .375, .45, .6) for d in (0.0, .5)]
        # for information: the same total composition without any anchor nucleus
        out += [R(f"ref composition only tc{t:g}", tc=t, f=0.0, form="resback", pos=0.0) for t in (.3, .45)]
        return out
    if which == "final":
        return [R("A0 heartlib"),
                R("A1 heartlib + 0.2 raw (round 6)", f=.2),
                R("A2 heartlib + 0.375 raw (round 6, first version)", f=.375),
                R("B1 interp: fwd + 0.375 pooled back", f=.375, form="poolbwd", fwd=True),
                R("B2 interp: fwd + 0.5 pooled back", f=.5, form="poolbwd", fwd=True),
                R("C1 heartlib at composition 0.375, fwd", tc=.375, fwd=True),
                R("C2 heartlib at composition 0.5, fwd", tc=.5, fwd=True),
                R("C3 composition 0.5: fwd + 0.2 pooled back", tc=.5, f=.2, form="poolbwd", fwd=True),
                # twins without the forward shift (second step of the rule) and controls; not finalists
                R("x B1 without fwd", f=.375, form="poolbwd"), R("x B2 without fwd", f=.5, form="poolbwd"),
                R("x C1 without fwd", tc=.375), R("x C2 without fwd", tc=.5),
                R("x C3 without fwd", tc=.5, f=.2, form="poolbwd"),
                R("x heartlib + 0.375 pooled", f=.375, form="pool"), R("x all libraries", base="all")]
    out = [R("heartlib"), R("heartlib blood.5", blood=.5), R("all libraries", base="all"), R("heart cells, all libraries", base="heart-all")]
    for f in (.2, .375):
        out += [R(f"heartlib + {f:g} raw", f=f), R(f"heartlib + {f:g} pooled", f=f, form="pool"),
                R(f"heartlib + {f:g} pooled back", f=f, form="poolbwd"),
                R(f"heartlib fwd + {f:g} pooled back", f=f, form="poolbwd", fwd=True),
                R(f"heartlib blood.5 + {f:g} raw", f=f, blood=.5),
                R(f"heartlib blood.5 fwd + {f:g} pooled back", f=f, form="poolbwd", fwd=True, blood=.5)]
    out += [R("heartlib fwd", fwd=True), R("all libraries + 0.375 raw", base="all", f=.375),
            R("heart cells, all libraries + 0.2 raw", base="heart-all", f=.2), R("heartlib + 0.5 raw", f=.5),
            R("heartlib fwd + 0.5 pooled back", f=.5, form="poolbwd", fwd=True)]
    return out


def build(rc, source, pool, anchor, n, t, seed, prev=None, days=4.0):
    """Cells of one recipe. pool = source rows that may be used; t = position on the source -> anchor path.
    prev = (previous stage, rows) and days = days from the source to the anchor: only for recipes with dev > 0."""
    pool = np.asarray(pool)
    if rc["base"] == "heartlib":
        libs = heart_libraries(source, pool)
        main = pool[np.isin(source.lib[pool], libs)] if libs else pool[np.isin(source.labels[pool], list(HEART))]
    elif rc["base"] == "heart-all":
        main = pool[np.isin(source.labels[pool], list(HEART))]
    else:
        main = pool
    shares = None
    if rc["blood"] != 1.0:
        shares = lineage_shares(source, main)
        shares["Blood"] *= rc["blood"]
        tot = sum(shares.values())
        shares = {k: v / tot for k, v in shares.items()}
    if rc.get("tc") is not None:
        # total composition = (1 - tc) x source lineage shares + tc x anchor lineage shares; the base supplies
        # what the f anchor nuclei (in the anchor's own composition) do not
        pi_s = lineage_shares(source, main)
        pi_a = anchor.context(source, main)["anchor_lineage_share"]
        base = {k: max((1 - rc["tc"]) * pi_s[k] + (rc["tc"] - rc["f"]) * pi_a.get(k, 0.0), 0.0) for k in LINEAGES}
        tot = sum(base.values())
        shares = feasible_shares(source, main, {k: v / tot for k, v in base.items()}, int(round(n * (1 - rc["f"]))))
    form = rc["form"]
    if form == "resback":
        pos = rc["pos"]
        dkw = dict(prev=prev, days_to_anchor=days, dev=rc["dev"]) if rc.get("dev") else {}
        return predict7(source, main, anchor, n, rc["f"], fwd=pos, bwd=1.0 - pos, harmonise=1.0, pool_k="resample" if rc["f"] > 0 else 1,
                        level=rc.get("level", "lineage"), shares=shares, seed=seed, **dkw)
    return predict7(source, main, anchor, n, rc["f"], fwd=t if rc["fwd"] else 0.0, bwd=(1 - t) if form == "poolbwd" else 0.0,
                    harmonise=1.0, pool_k="match" if form in ("pool", "poolbwd") else 1, shares=shares, seed=seed)


def consistency(ref, probe, probe_obs, rc, source, pool, anchor, half, t, seeds, extra=None, prev=None, days=4.0):
    """Probe skills if recipe `rc` were the truth (mean over seeds) and the distance D to the observed skills.

    Also returned: D of every single seed (for a standard error) and, if `extra` cells are given (a second
    probe), their mean total score under the same hypothetical truth."""
    rows, other = [], []
    for seed in seeds:
        cells = build(rc, source, pool, anchor, 2 * half, t, 1000 + seed, prev=prev, days=days)
        order = np.random.default_rng(seed).permutation(len(cells))
        ev = FastEval(ref, cells[order[:half]], cells[order[half:]])
        rows.append(ev(probe))
        if extra is not None:
            other.append(ev(extra)["score"])
    mean = {k: float(np.mean([r[k] for r in rows])) for k in W}
    d = float(sum(w * (mean[k] - probe_obs[k]) ** 2 for k, w in W.items()))
    per_seed = [float(sum(w * (r[k] - probe_obs[k]) ** 2 for k, w in W.items())) for r in rows]
    info = {"D_seed_mean": float(np.mean(per_seed)), "D_seed_se": float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else float("nan"),
            **{"sd_" + k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else float("nan") for k in W}}
    if other:
        info["extra_total"] = float(np.mean(other))
    return mean, d, float(np.mean([r["score"] for r in rows])), info


def pseudo_library_source(e9, e10, seed=SEED):
    """E9.5 with the test's library structure: a deep heart-enriched pseudo-library and a shallow surrounding one.

    Proxy construction, not a prediction: the structure copies E10.5 (40% of the cells in a library with 94% heart
    cells; the other cells thinned to the depth distribution of E10.5 libraries 1+2). E10.5 is the target stage of
    the p8s settings, so this uses target-stage statistics to BUILD THE PROXY SOURCE; the recipes scored on it see
    only this source and the anchor. Heart-library recipes never touch the thinned cells; the probe and the
    all-library recipes do."""
    rng = np.random.default_rng(seed)
    heart = np.isin(e9.labels, list(HEART))
    h_rows, o_rows = np.flatnonzero(heart), np.flatnonzero(~heart)
    n_a = int(round(.4 * e9.n_cells))
    a_heart = rng.choice(h_rows, int(round(.94 * n_a)), replace=False)
    a_other = rng.choice(o_rows, n_a - len(a_heart), replace=False)
    in_a = np.zeros(e9.n_cells, bool)
    in_a[a_heart] = True
    in_a[a_other] = True
    out = copy.copy(e9)
    shallow = np.flatnonzero(~in_a)
    x = e9.X.tolil()
    x[shallow] = thin_rows(e9.X[shallow], depth_of(e10.X[np.flatnonzero(e10.lib != "3")]), rng)
    out.X = x.tocsr()
    out.lib = np.where(in_a, "3", "1")
    return out


def report(frame, title):
    pd.set_option("display.width", 250, "display.max_rows", 400)
    print(f"\n=== {title}")
    cols = [c for c in ("actual", "D", "D_seed_se", "probe_total", "skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram", "extra_total") if c in frame]
    print(frame.set_index("recipe")[cols].round(4).sort_values("D").to_string())
    if "actual" in frame:
        frame = frame[~frame["recipe"].str.startswith("ref ")]
        best = frame.loc[frame["D"].idxmin()]
        rho = frame[["actual", "D"]].corr(method="spearman").iloc[0, 1]
        rank = int((frame["actual"] > best["actual"]).sum()) + 1
        print(f"rule picks '{best['recipe']}': actual {best['actual']:.2f}, rank {rank}/{len(frame)}, best {frame['actual'].max():.2f}, "
              f"regret {frame['actual'].max() - best['actual']:.2f}, median {frame['actual'].median():.2f}; Spearman(actual, D) = {rho:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calib", "test"))
    parser.add_argument("--setting", default="zero", choices=("zero", "p8s"))
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--tag", default="")
    parser.add_argument("--only", default="", help="comma-separated scenario / target names of the setting")
    parser.add_argument("--recipes", default="v1", choices=("v1", "final", "grid", "grid2"))
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]
    only = set(args.only.split(",")) if args.only else set()
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    _, genes = board_spec("T1:val")
    trend, _ = load_trend(genes)
    ax, _ = load_anchor(genes, verbose=False)
    anchor = Anchor7(ax)
    out = REAL / "runs" / "round7_probe"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    split10, h3 = split_rows(e9, e10), h3_rows(e10)
    rcs = recipes(args.recipes)
    records = []

    def run(setting, ev_true, ref, source, pool, probe, probe_rows, t, half, prev=None, days=4.0):
        """One calibration setting: actual scores of all recipes, then D from the probe's actual skills."""
        obs = ev_true(probe)
        print(f"[{setting}] probe actual {obs['score']:.2f}  " + " ".join(f"{k[6:]} {obs[k]:.3f}" for k in W), f"{time.time() - t0:.0f}s", flush=True)
        free = np.setdiff1d(np.asarray(pool), probe_rows)
        for rc in rcs:
            actual = ev_true(build(rc, source, pool, anchor, N_PROBE, t, SEED, prev=prev, days=days))["score"]
            mean, d, total, info = consistency(ref, probe, obs, rc, source, free, anchor, half, t, seeds, prev=prev, days=days)
            records.append({"setting": setting, "recipe": rc["name"], "actual": actual, "D": d, "probe_total": total, **mean, **info})
            print(f"[{setting}] {rc['name']:<45} actual {actual:6.2f}  D {d:.5f}  probe-if-true {total:6.2f}", flush=True)
            pd.DataFrame(records).to_csv(out / f"calib_{args.recipes}_{args.setting}_{args.split}{args.tag}.csv", index=False)

    if args.mode == "calib" and args.setting == "zero":
        from experiments.round6_final import scenario_setup
        e10s, pool, r4_ref, evs = scenario_setup(args.split, genes)
        ref = e9.dense(split10["reference"])
        probe, probe_rows = h1_like(e10s, pool, trend, N_PROBE)
        print("probe library shares", {k: round(float((e10s.lib[probe_rows] == k).mean()), 3) for k in "123"}, flush=True)
        for scen, ev in evs.items():
            if only and scen not in only:
                continue
            run(f"zero/{scen}", ev, ref, e10s, pool, probe, probe_rows, .5, 1249, prev=(e9, np.arange(e9.n_cells)), days=4.0)
    elif args.mode == "calib":
        e8 = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
        ref = e8.dense(np.sort(np.random.default_rng(SEED).choice(e8.n_cells, round(.1 * e8.n_cells), replace=False)))
        src = pseudo_library_source(e9, e10)
        pool = np.arange(src.n_cells)
        print("pseudo-library source: library 3 share", round(float((src.lib == "3").mean()), 3), "heart share in 3 / 1",
              round(float(np.isin(src.labels[src.lib == "3"], list(HEART)).mean()), 3), round(float(np.isin(src.labels[src.lib == "1"], list(HEART)).mean()), 3),
              "median depth 3 / 1", float(np.median(depth_of(src.X[np.flatnonzero(src.lib == "3")]))), float(np.median(depth_of(src.X[np.flatnonzero(src.lib == "1")]))), flush=True)
        probe, probe_rows = h1_like(src, pool, trend, N_PROBE)
        print("probe library shares", {k: round(float((src.lib[probe_rows] == k).mean()), 3) for k in "13"}, flush=True)
        for name, rows in (("p8s/deep", h3), ("p8s/all", split10)):
            if only and name not in only:
                continue
            ev = FastEval(ref, e10.dense(rows[a]), e10.dense(rows[b]))
            run(name, ev, ref, src, pool, probe, probe_rows, .2, 1249, prev=(e8, np.arange(e8.n_cells)), days=5.0)
    else:
        saved = np.load(REAL / "runs" / "round7_h1" / "h1.npz")
        probe, probe_rows = saved["cells"], saved["rows"]
        ref = e9.dense(split10["reference"])
        # second probe (diagnostic only, not part of the rule): E10.5 cells as released; a public note puts such a
        # file "near 69.4" on T1:test
        verb_rows = np.sort(np.random.default_rng(7).choice(np.setdiff1d(np.arange(e10.n_cells), probe_rows), 2000, replace=False))
        verbatim = e10.dense(verb_rows)
        free = np.setdiff1d(np.arange(e10.n_cells), np.concatenate([probe_rows, verb_rows]))
        if only:
            rcs = [rc for rc in rcs if rc["name"].split()[0] in only]
        for rc in rcs:
            mean, d, total, info = consistency(ref, probe, H1_OBSERVED, rc, e10, free, anchor, 638, .5, seeds, extra=verbatim,
                                               prev=(e9, np.arange(e9.n_cells)), days=4.0)
            records.append({"setting": "test", "recipe": rc["name"], "D": d, "probe_total": total, **mean, **info})
            print(f"[test] {rc['name']:<48} D {d:.5f} (se {info['D_seed_se']:.5f})  h1-if-true {total:6.2f}  " + " ".join(f"{k[6:]} {mean[k]:.3f}" for k in W)
                  + f"  verbatim-if-true {info['extra_total']:.2f}", flush=True)
            pd.DataFrame(records).to_csv(out / f"test_{args.recipes}{args.tag}.csv", index=False)
    frame = pd.DataFrame(records)
    for setting, sub in frame.groupby("setting", sort=False):
        report(sub, setting)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
