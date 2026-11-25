"""Round 6 validation suite: every proxy, what it imitates, and its blind spots.

Structure of the hidden test: reference = E9.5 (two stages before the target), source = E10.5 (one heart-proper
library + two shallow surrounding-tissue libraries), target = E12.5 (hearts dissected specifically, same
laboratory, depth unknown), an external anchor at E14.5 (elapsed share of the source-to-anchor interval: 0.5).

name      reference      source (pool)        target                elapsed  imitates / blind spot
gen       E8.5 sample    E8.5                 E9.5                  1/6      a transition nobody tuned on; same dissection, deep target / no reference two stages back
p8/deep   E8.5 sample    E9.5                 E10.5 library 3       1/5      reference two stages back, target = heart-proper deep / every round-2..5 choice was tuned on this transition
p8/all    E8.5 sample    E9.5                 E10.5 all libraries   1/5      same, target mixed depth with surrounding tissue
2d/deep   E8.5 sample    E8.5                 E10.5 library 3       1/3      two-day horizon / reference = source
2d/all    E8.5 sample    E8.5                 E10.5 all libraries   1/3      two-day horizon / reference = source
D/raw     E9.5 rows      E10.5 libraries 1+2  E10.5 library 3       0        pure dissection change within one laboratory and stage / no time gap
D/dm      E9.5 rows      E10.5 libraries 1+2  library 3, thinned    0        same with depth equalised
TE/raw    E9.5 rows      E10.5 (any pool)     E14.5 other lab       1        the real test inputs against a real later heart / four days, other laboratory
TE/dm     E9.5 thinned   E10.5 thinned        E14.5 other lab       1        same with depth equalised
raw,h3    E9.5 rows      E9.5                 E10.5 all / library 3 1/5      round-4/5 selection proxies (reference = source stage; structure differs from the test)
dm        E9.5 thinned   E9.5 thinned         E10.5 all             1/5      round-4/5 depth-matched proxy

Targets are split in disjoint halves: `development`/`ceiling` for tuning, `confirmation`/`confirmation_ceiling`
looked at once per finalist. Local scores are calibrated per proxy (floor = its reference, ceiling = the other
half); they are not leaderboard scores.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from experiments.round5_common import h3_rows, split_rows
from experiments.round6_common import (REAL, SEED, FastEval, Mem, anchor_split, depth_of, lineage_of, thin_rows,
                                       thinned, LINEAGES)
from t1_pipeline.data import board_spec

ALL = ("gen", "p8/deep", "p8/all", "2d/deep", "2d/all", "D/raw", "D/dm", "TE/raw", "TE/dm", "raw", "h3", "dm")
MIRROR = ("gen", "p8/deep", "p8/all")          # reference one stage before the source (or none), time passes
HEART_TARGET = ("p8/deep", "2d/deep", "D/raw", "D/dm", "h3")   # the target is a heart-proper library
ELAPSED = {"gen": 1 / 6, "p8/deep": .2, "p8/all": .2, "2d/deep": 1 / 3, "2d/all": 1 / 3, "D/raw": 0., "D/dm": 0.,
           "TE/raw": 1., "TE/dm": 1., "raw": .2, "h3": .2, "dm": .2}


@dataclass
class Proxy:
    name: str
    ev: FastEval
    source: Mem
    pool: np.ndarray
    r4_ref: np.ndarray
    elapsed: float
    target_shares: dict = field(default_factory=dict)   # lineage shares of the target: for oracle ANALYSIS only
    pools: dict = field(default_factory=dict)

    @property
    def heart_target(self):
        return self.name in HEART_TARGET


def shares_of(labels):
    lin = lineage_of(labels)
    return {k: float((lin == k).mean()) for k in LINEAGES}


class Suite:
    def __init__(self, split="development", names=ALL, anchor=None, verbose=True, allow_te=False):
        assert split in ("development", "confirmation")
        a, b = ("development", "ceiling") if split == "development" else ("confirmation", "confirmation_ceiling")
        self.split, self.proxies = split, {}
        _, genes = board_spec("T1:val")
        t0 = time.time()
        need = lambda *keys: any(k in names for k in keys)
        e8 = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes) if need("gen", "p8/deep", "p8/all", "2d/deep", "2d/all") else None
        e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
        e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
        self.stages = {"E8.5": e8, "E9.5": e9, "E10.5": e10}
        split10 = split_rows(e9, e10)
        h3 = h3_rows(e10)
        ref9_rows = split10["reference"]
        rng = lambda: np.random.default_rng(SEED)
        if e8 is not None:
            ref8_rows = np.sort(rng().choice(e8.n_cells, round(.1 * e8.n_cells), replace=False))
            ref8 = e8.dense(ref8_rows)
            order9 = rng().permutation(e9.n_cells)
            k = 0 if split == "development" else 2
            t9 = {a: np.sort(order9[k * 1249:(k + 1) * 1249]), b: np.sort(order9[(k + 1) * 1249:(k + 2) * 1249])}
        ref9 = e9.dense(ref9_rows)
        lib12 = np.flatnonzero(e10.lib != "3")
        lib3 = np.flatnonzero(e10.lib == "3")
        all9, all10 = np.arange(e9.n_cells), np.arange(e10.n_cells)

        def add(name, ref, target, ceiling, source, pool, r4_ref, labels, pools=None):
            ev = FastEval(ref, target, ceiling)
            self.proxies[name] = Proxy(name, ev, source, pool, r4_ref, ELAPSED[name], shares_of(labels), pools or {})
            if verbose:
                print(f"[suite] {name:<8} ready {time.time() - t0:5.0f}s  floor mmd {ev.floor['mmd_u']:.4f} vario {ev.floor['variogram']:.5f}"
                      f" | ceiling des {ev.ceiling['de_score']:.3f} dcs {ev.ceiling['de_direction']:.3f} mmd {ev.ceiling['mmd_u']:.4f}"
                      f" vario {ev.ceiling['variogram']:.5f} | n_up {len(ev.up)} n_dn {len(ev.dn)}", flush=True)

        if "gen" in names:
            add("gen", ref8, e9.dense(t9[a]), e9.dense(t9[b]), e8, np.arange(e8.n_cells), ref8, e9.labels[t9[a]])
        if "p8/deep" in names:
            add("p8/deep", ref8, e10.dense(h3[a]), e10.dense(h3[b]), e9, all9, ref9, e10.labels[h3[a]])
        if "p8/all" in names:
            add("p8/all", ref8, e10.dense(split10[a]), e10.dense(split10[b]), e9, all9, ref9, e10.labels[split10[a]])
        if "2d/deep" in names:
            add("2d/deep", ref8, e10.dense(h3[a]), e10.dense(h3[b]), e8, np.arange(e8.n_cells), ref8, e10.labels[h3[a]])
        if "2d/all" in names:
            add("2d/all", ref8, e10.dense(split10[a]), e10.dense(split10[b]), e8, np.arange(e8.n_cells), ref8, e10.labels[split10[a]])
        if need("D/raw", "D/dm"):
            r4_ref = e10.dense(np.sort(lib12[rng().choice(len(lib12), 2000, replace=False)]))
            if "D/raw" in names:
                add("D/raw", ref9, e10.dense(h3[a]), e10.dense(h3[b]), e10, lib12, r4_ref, e10.labels[h3[a]])
            if "D/dm" in names:
                shallow = depth_of(e10.X[lib12])
                add("D/dm", ref9, thin_rows(e10.X[h3[a]], shallow, rng()).toarray(), thin_rows(e10.X[h3[b]], shallow, rng()).toarray(),
                    e10, lib12, r4_ref, e10.labels[h3[a]])
        if need("TE/raw", "TE/dm"):
            # The TE targets are halves of the external E14.5 sample, the same sample every other script mixes in as
            # the anchor. Predictions built by round6_method.predict draw from ALL anchor nuclei, so scoring them on
            # TE would put target nuclei into the prediction. TE is therefore only available to callers that mix
            # exclusively from `anchor_pool` (round6_te.py builds its own targets and does exactly that).
            if not allow_te:
                raise ValueError("TE proxies: use experiments.round6_te, which mixes only nuclei outside the target halves")
            assert anchor is not None
            asp = anchor_split(anchor.shape[0])
            self.anchor_pool = asp["pool"]
            target, ceiling = anchor[asp[a]].toarray(), anchor[asp[b]].toarray()
            pools = {"all": all10, "lib3": lib3, "lib12": lib12}
            r4_ref = e10.dense(np.sort(rng().choice(e10.n_cells, 2000, replace=False)))
            if "TE/raw" in names:
                add("TE/raw", ref9, target, ceiling, e10, all10, r4_ref, np.array(["?"]), pools)
            if "TE/dm" in names:
                depth = depth_of(anchor)
                e9t, e10t = thinned(e9, depth), thinned(e10, depth)
                r4t = e10t.dense(np.sort(rng().choice(e10.n_cells, 2000, replace=False)))
                add("TE/dm", e9t.dense(ref9_rows), target, ceiling, e10t, all10, r4t, np.array(["?"]), pools)
        if "raw" in names:
            add("raw", ref9, e10.dense(split10[a]), e10.dense(split10[b]), e9, all9, ref9, e10.labels[split10[a]])
        if "h3" in names:
            add("h3", ref9, e10.dense(h3[a]), e10.dense(h3[b]), e9, all9, ref9, e10.labels[h3[a]])
        if "dm" in names:
            e9d = Mem(REAL / "data/proxy/E9.5_RNA.h5ad", genes)
            ref9d = e9d.dense(ref9_rows)
            add("dm", ref9d, e10.dense(split10[a]), e10.dense(split10[b]), e9d, all9, ref9d, e10.labels[split10[a]])
