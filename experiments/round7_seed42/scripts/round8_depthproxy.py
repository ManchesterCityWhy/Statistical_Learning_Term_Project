"""Round 8, afternoon of 11-26: the five proxies with real elapsed time, with the source cells thinned before the recipe runs.

Does thinning help where the target really is sparser than the source (E9.5 -> E10.5 library 3: every lineage has 12-20% fewer
molecules per cell; E10.5 all libraries: much sparser), and what does it cost where it is not (E8.5 -> E9.5)?

thin = share of its molecules a source cell keeps. "per2d s": s ** (days elapsed / 2), i.e. the test-time factor s scaled to the
proxy's elapsed time; "fixed s": s whatever the elapsed time.

    python -m experiments.round8_depthproxy <tag> <development|confirmation> <seeds>
"""
import copy
import sys
import time

import numpy as np
import pandas as pd

from experiments.round5_anchor import load_anchor
from experiments.round6_common import REAL, Log, depth_of, thin_rows
from experiments.round6_suite import Suite
from experiments.round7_eval import C, cells_of
from experiments.round7_method import Anchor7
from t1_pipeline.data import board_spec

tag, split = sys.argv[1], sys.argv[2]
seeds = [int(v) for v in sys.argv[3].split(",")]
names = tuple((sys.argv[4] if len(sys.argv) > 4 else "gen,p8/deep,p8/all,2d/deep,2d/all").split(","))
t0 = time.time()
_, genes = board_spec("T1:val")
ax, _ = load_anchor(genes, verbose=False)
anchor = Anchor7(ax)
suite = Suite(split, names, anchor=ax)
real = dict(rho_f=.9, bwd=1.0, pool_k="resample", level="type", k_prof=20, real_share=1.0, rho_dev=1.0, days_trend=1.0)
R8 = C("R8", rho_mix=.5, mix_low=.3, **real)
CFGS = [("R7h", C("R7h", **{**real, "real_share": .5}), None),
        ("R8", R8, None),
        ("R8 thin per2d 0.92", R8, ("per2d", .92)), ("R8 thin per2d 0.85", R8, ("per2d", .85)), ("R8 thin per2d 0.70", R8, ("per2d", .70)),
        ("R8 thin fixed 0.85", R8, ("fixed", .85)), ("R8 thin fixed 0.70", R8, ("fixed", .70))]
if len(sys.argv) > 5 and sys.argv[5] == "set2":      # confirmation of the candidates: depth x mixture ratio (test-time values in the names)
    M = lambda m: C(f"mix{m:g}", rho_mix=2 * m, mix_low=.3, **real)
    CFGS = [("R7h", C("R7h", **{**real, "real_share": .5}), None), ("R8 (mix 0.25)", R8, None),
            ("mix 0.25 thin fixed 0.85", R8, ("fixed", .85)), ("mix 0.25 thin per2d 0.85", R8, ("per2d", .85)),
            ("mix 0.35", M(.35), None), ("mix 0.35 thin fixed 0.85", M(.35), ("fixed", .85)),
            ("mix 0.5", M(.5), None), ("mix 0.5 thin fixed 0.85", M(.5), ("fixed", .85)), ("mix 0.5 thin per2d 0.85", M(.5), ("per2d", .85)),
            ("mix 0.75", M(.75), None), ("mix 0.75 thin fixed 0.85", M(.75), ("fixed", .85))]
_THIN = {}


def thinned(source, s):
    key = (id(source), round(float(s), 4))
    if key not in _THIN:
        rng = np.random.default_rng(900 + int(round(s * 1000)))
        out = copy.copy(source)
        out.X = thin_rows(source.X, s * depth_of(source.X), rng).astype(np.float32).tocsr()
        _THIN[key] = out
    return _THIN[key]


log = Log(REAL / "runs" / "round8_probe2" / f"depthproxy_{tag}_{split}.csv")
for pname, p in suite.proxies.items():
    e8 = suite.stages.get("E8.5")
    is_e9 = p.source is suite.stages["E9.5"]
    prev = (e8, np.arange(e8.n_cells)) if (is_e9 and e8 is not None) else None
    days = 5.0 if is_e9 else 6.0
    for seed in seeds:
        for name, cfg, thin in CFGS:
            if cfg.get("rho_dev") and prev is None:
                cfg = {**cfg, "rho_dev": 0.0}
            s = 1.0
            if thin is not None:
                s = thin[1] ** (p.elapsed * days / 2.0) if thin[0] == "per2d" else thin[1]
            src = p.source if s >= 1.0 else thinned(p.source, s)
            cells, f, t, bwd = cells_of(cfg, src, p.pool, anchor, p.elapsed, p.heart_target, p.target_shares, seed=seed, prev=prev, days_to_anchor=days)
            log.add(p.ev, cells, config=name, proxy=pname, split=split, seed=seed, thin=round(float(s), 4))
    _THIN.clear()
frame = pd.DataFrame(log.records)
pd.set_option("display.width", 250)
print(f"done {time.time() - t0:.0f}s")
print(frame.pivot_table(index="config", columns="proxy", values="score", sort=False).round(2).to_string())
for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
    print(m)
    print(frame.pivot_table(index="config", columns="proxy", values=m, sort=False).round(3).to_string())
