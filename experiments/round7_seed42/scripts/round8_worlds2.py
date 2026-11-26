"""Round 8, second set of hypothetical E12.5 populations: SMALL displacements along the part of the anchor / source
ratio that all lineages share ("shared", fwd_kind dc) or along the whole ratio ("whole", dr), with / without the
interpolated composition and the round-7 shifts. Same clean split as round8_worlds. Three seeds per world.

Read with the public test board in mind: files of real E10.5 cells only score about DES 72, DCS 82, MMD 86-92,
CSS 79-83 there; E10.5 + E14.5 mixtures score about DES 78-81, DCS 88-92, MMD 78-81, CSS 78-88. A world is compatible
with that if its B row and its A rows look like these two classes."""
import sys
import numpy as np, pandas as pd
from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import split_rows
from experiments.round6_common import REAL, FastEval, Mem
from experiments.round6_s1 import feasible_shares
from experiments.round7_method import Anchor7, composition_shares, predict7
from t1_pipeline.data import board_spec
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 70)
tag = sys.argv[1] if len(sys.argv) > 1 else "w2"
_, genes = board_spec("T1:test")
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
ax, _ = load_anchor(genes, verbose=False)
trend, _ = load_trend(genes)
ref = e9.dense(split_rows(e9, e10)["reference"])
prev = (e9, np.arange(e9.n_cells))
lib3 = np.flatnonzero(e10.lib == "3")
r0 = np.random.default_rng(2026)
perm = r0.permutation(lib3)
pool_t, pool_f = np.sort(perm[:len(lib3) // 3]), np.sort(perm[len(lib3) // 3:])
nperm = r0.permutation(ax.shape[0])
nuc_t, nuc_f = np.sort(nperm[:int(.4 * ax.shape[0])]), np.sort(nperm[int(.4 * ax.shape[0]):])
anchor_t, anchor_f = Anchor7(ax[nuc_t]), Anchor7(ax[nuc_f])
shifts = dict(prev=prev, days_to_anchor=4.0, dev=0.5, trend=trend, trend_share=0.4)
r7 = dict(bwd=1.0, harmonise=1.0, pool_k="resample", level="type", k_prof=20, real_share=0.5)
F = {
    "B: library 3 as it is": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.0),
    "A20: 20% raw nuclei, shared correction": lambda: predict7(e10, pool_f, anchor_f, 4000, 0.2),
    "A40: 40% raw nuclei, no correction": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.4, harmonise=0.0),
    "R7h: file recipe": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, **r7, **shifts),
}
for a in (0.05, 0.1, 0.175, 0.25, 0.5):
    F[f"R7h + shared {a:g}"] = (lambda a=a: predict7(e10, pool_f, anchor_f, 5000, 0.45, fwd=a, fwd_kind="dc", **r7, **shifts))
for a in (0.05, 0.1, 0.25):
    F[f"R7h + whole {a:g}"] = (lambda a=a: predict7(e10, pool_f, anchor_f, 5000, 0.45, fwd=a, fwd_kind="dr", **r7, **shifts))
for a in (0.1, 0.25):
    F[f"B + shared {a:g}"] = (lambda a=a: predict7(e10, pool_f, anchor_f, 5000, 0.0, fwd=a, fwd_kind="dc", level="type"))
files = {k: f() for k, f in F.items()}
print("files built", len(files), flush=True)
comp = lambda n: composition_shares(e10, pool_t, anchor_t, 0.45, 0.0, n, feasible_shares)
N = 2 * 638
T = lambda seed, **kw: predict7(e10, pool_t, anchor_t, N, 0.0, seed=seed, **kw)
W = {
    "T0 library 3 as it is": lambda s: T(s),
    "T3 composition + round-7 shifts": lambda s: T(s, shares=comp(N), **shifts),
    "S10 shared 0.1": lambda s: T(s, fwd=0.1, fwd_kind="dc", level="type"),
    "S25 shared 0.25": lambda s: T(s, fwd=0.25, fwd_kind="dc", level="type"),
    "S50 shared 0.5": lambda s: T(s, fwd=0.5, fwd_kind="dc", level="type"),
    "S10c shared 0.1 + composition": lambda s: T(s, fwd=0.1, fwd_kind="dc", level="type", shares=comp(N)),
    "S25c shared 0.25 + composition": lambda s: T(s, fwd=0.25, fwd_kind="dc", level="type", shares=comp(N)),
    "S10c7 shared 0.1 + composition + round-7 shifts": lambda s: T(s, fwd=0.1, fwd_kind="dc", level="type", shares=comp(N), **shifts),
    "S25c7 shared 0.25 + composition + round-7 shifts": lambda s: T(s, fwd=0.25, fwd_kind="dc", level="type", shares=comp(N), **shifts),
    "W10 whole ratio 0.1": lambda s: T(s, fwd=0.1, fwd_kind="dr", level="type"),
    "W25c whole ratio 0.25 + composition": lambda s: T(s, fwd=0.25, fwd_kind="dr", level="type", shares=comp(N)),
}
rows = []
for wname, make in W.items():
    for seed in (101, 102, 103):
        t = make(seed)
        order = np.random.default_rng(seed).permutation(len(t))
        ev = FastEval(ref, t[order[:638]], t[order[638:]])
        for fname, f in files.items():
            s = ev(f)
            rows.append(dict(world=wname, file=fname, seed=seed, score=s["score"], DES=100 * s["skill_de_score"], DCS=100 * s["skill_de_direction"],
                             MMD=100 * s["skill_mmd_u"], CSS=100 * s["skill_variogram"]))
    print(wname, "done", flush=True)
frame = pd.DataFrame(rows)
out = REAL / "runs" / "round8_worlds"
out.mkdir(parents=True, exist_ok=True)
frame.to_csv(out / f"worlds_{tag}.csv", index=False)
short = lambda w: w.split(" ")[0]
frame["w"] = frame.world.map(short)
print(frame.pivot_table(index="file", columns="w", values="score", sort=False).round(1).to_string())
for m in ("DES", "DCS", "MMD", "CSS"):
    print(m); print(frame.pivot_table(index="file", columns="w", values=m, sort=False).round(1).to_string())
