"""Round 8: hypothetical E12.5 populations ("worlds") and what several kinds of file would score in them.

Same clean design as round7_payoff3: library 3 is split at random, one third for the worlds, two thirds for the files;
the anchor nuclei are split 40% / 60% and each side estimates its own anchor / source ratios. No returned score is used.
Reference: the E9.5 sample of the test. Three seeds per world.

    files   B    library-3 cells as they are (the kind of file made of E10.5 real cells only)
            A20  80% library-3 cells + 20% raw anchor nuclei with the shared correction (round 6 file's recipe)
            A40  60% library-3 cells + 40% raw anchor nuclei, no correction (plain E10.5 + E14.5 mixture)
            R7h  the file's recipe (55% + 45%, half of the added cells real, two-witness shift + bulk trend)
            further candidates given in CANDIDATES
    worlds  T0 library 3 as it is; T1 expression moved by the round-7 shifts; T2 composition moved (0.45); T3 both;
            T4 the all-synthetic round-7 construction;
            T5 real cells moved along the whole anchor / source ratio by 0.5; T6 = T5 + composition 0.45;
            T7 whole ratio 0.25 + composition 0.45; T8 = T6 + round-7 shifts
"""
import sys
import numpy as np, pandas as pd
from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round5_common import split_rows
from experiments.round6_common import REAL, FastEval, Mem
from experiments.round6_s1 import feasible_shares
from experiments.round7_method import Anchor7, composition_shares, predict7
from t1_pipeline.data import board_spec
pd.set_option("display.width", 280); pd.set_option("display.max_colwidth", 70)
tag = sys.argv[1] if len(sys.argv) > 1 else "w1"
which = sys.argv[2].split(",") if len(sys.argv) > 2 else None
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
CANDIDATES = {
    "B: library 3 as it is": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.0),
    "A20: 20% raw nuclei, shared correction": lambda: predict7(e10, pool_f, anchor_f, 4000, 0.2),
    "A40: 40% raw nuclei, no correction": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.4, harmonise=0.0),
    "R7h: file recipe": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, **r7, **shifts),
    "R7h + whole ratio 0.25": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, fwd=0.25, fwd_kind="dr", **r7, **shifts),
    "R7h + whole ratio 0.5": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, fwd=0.5, fwd_kind="dr", **r7, **shifts),
    "R7h cells, whole ratio 0.5 only": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, fwd=0.5, fwd_kind="dr", **r7),
    "R7h, shifts in count space": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, shift_mode="counts", **r7, **shifts),
    "R7h + whole ratio 0.5, count space": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, fwd=0.5, fwd_kind="dr", shift_mode="counts", **r7, **shifts),
}
if which:
    CANDIDATES = {k: v for k, v in CANDIDATES.items() if k.split(":")[0] in which or k in which}
files = {k: f() for k, f in CANDIDATES.items()}
print("files built", {k: v.shape[0] for k, v in files.items()}, flush=True)
comp = lambda n: composition_shares(e10, pool_t, anchor_t, 0.45, 0.0, n, feasible_shares)

def truth(kind, seed):
    n = 2 * 638
    kw = dict(seed=seed)
    if kind == "T0":
        return predict7(e10, pool_t, anchor_t, n, 0.0, **kw)
    if kind == "T1":
        return predict7(e10, pool_t, anchor_t, n, 0.0, **kw, **shifts)
    if kind == "T2":
        return predict7(e10, pool_t, anchor_t, n, 0.0, shares=comp(n), **kw)
    if kind == "T3":
        return predict7(e10, pool_t, anchor_t, n, 0.0, shares=comp(n), **kw, **shifts)
    if kind == "T4":
        return predict7(e10, pool_t, anchor_t, n, 0.45, bwd=1.0, harmonise=1.0, pool_k="resample", level="type", k_prof=20, **kw, **shifts)
    if kind == "T5":
        return predict7(e10, pool_t, anchor_t, n, 0.0, fwd=0.5, fwd_kind="dr", level="type", **kw)
    if kind == "T6":
        return predict7(e10, pool_t, anchor_t, n, 0.0, fwd=0.5, fwd_kind="dr", level="type", shares=comp(n), **kw)
    if kind == "T7":
        return predict7(e10, pool_t, anchor_t, n, 0.0, fwd=0.25, fwd_kind="dr", level="type", shares=comp(n), **kw)
    if kind == "T8":
        return predict7(e10, pool_t, anchor_t, n, 0.0, fwd=0.5, fwd_kind="dr", level="type", shares=comp(n), **kw, **shifts)
    if kind == "T9":      # count-space version of T6: molecules removed / added
        return predict7(e10, pool_t, anchor_t, n, 0.0, fwd=0.5, fwd_kind="dr", level="type", shares=comp(n), shift_mode="counts", **kw)
    raise SystemExit(kind)

names = {"T0": "library 3 as it is", "T1": "expression moved (round-7 shifts)", "T2": "composition moved (0.45)", "T3": "composition + round-7 shifts",
         "T4": "all-synthetic round-7 construction", "T5": "whole ratio 0.5", "T6": "whole ratio 0.5 + composition",
         "T7": "whole ratio 0.25 + composition", "T8": "whole ratio 0.5 + composition + round-7 shifts", "T9": "whole ratio 0.5 + composition, count space"}
rows = []
for kind in names:
    for seed in (101, 102, 103):
        t = truth(kind, seed)
        order = np.random.default_rng(seed).permutation(len(t))
        ev = FastEval(ref, t[order[:638]], t[order[638:]])
        for fname, f in files.items():
            s = ev(f)
            rows.append(dict(world=kind + " " + names[kind], file=fname, seed=seed, score=s["score"], DES=100 * s["skill_de_score"], DCS=100 * s["skill_de_direction"],
                             MMD=100 * s["skill_mmd_u"], CSS=100 * s["skill_variogram"]))
    print(kind, "done", flush=True)
frame = pd.DataFrame(rows)
out = REAL / "runs" / "round8_worlds"
out.mkdir(parents=True, exist_ok=True)
frame.to_csv(out / f"worlds_{tag}.csv", index=False)
print(frame.pivot_table(index="world", columns="file", values="score", sort=False).round(1).T.to_string())
for m in ("DES", "DCS", "MMD", "CSS"):
    print(m); print(frame.pivot_table(index="world", columns="file", values=m, sort=False).round(1).T.to_string())
p = frame.groupby(["world", "file"], sort=False)[["score", "DES", "DCS", "MMD", "CSS"]].mean()
b, a20, a40 = "B: library 3 as it is", "A20: 20% raw nuclei, shared correction", "A40: 40% raw nuclei, no correction"
if all(k in files for k in (b, a20, a40)):
    print("class pattern, mixture minus real-cells-only (public board: DES +6..+9, DCS +5..+9, MMD -6..-13, CSS -1..+6)")
    for a in (a20, a40):
        d = (p.xs(a, level="file") - p.xs(b, level="file")).round(1)
        print(a); print(d.to_string())
