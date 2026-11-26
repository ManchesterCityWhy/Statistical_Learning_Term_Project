"""Round 8, follow-up to the replication: worlds made of REAL cells only in which the low-gene difference is concentrated in a
share of the cells (those cells carry the anchor's low genes completely, the others not at all), next to the worlds where
it is spread evenly and to the all-synthetic round-7 construction. New random split (argument 2), seeds 301-303."""




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
tag = sys.argv[1] if len(sys.argv) > 1 else "w7"
_, genes = board_spec("T1:test")
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
ax, _ = load_anchor(genes, verbose=False)
trend, _ = load_trend(genes)
ref = e9.dense(split_rows(e9, e10)["reference"])
prev = (e9, np.arange(e9.n_cells))
lib3 = np.flatnonzero(e10.lib == "3")
r0 = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 2027)
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
    "R7h: file recipe": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, **r7, **shifts),
    "R7 all synthetic": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, **{**r7, "real_share": 0.0}, **shifts),
    "R7 all real": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, **{**r7, "real_share": 1.0}, **shifts),
}
allreal = {**r7, "real_share": 1.0}
F["candidate: all real + low genes 0.25"] = lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, mix=0.25, mix_low=0.3, **allreal, **shifts)
F["all real + low genes 1 in 25% of the cells"] = lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, mix=1.0, mix_low=0.3, mix_cells=0.25, **allreal, **shifts)
F["all real + low genes 0.15"] = lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, mix=0.15, mix_low=0.3, **allreal, **shifts)
F["file recipe + low genes 0.15 on its real cells"] = lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, mix=0.15, mix_low=0.3, **r7, **shifts)
files = {k: f() for k, f in F.items()}
print("files built", len(files), flush=True)
comp = lambda n: composition_shares(e10, pool_t, anchor_t, 0.45, 0.0, n, feasible_shares)
N = 2 * 638
T = lambda seed, **kw: predict7(e10, pool_t, anchor_t, N, 0.0, seed=seed, level="type", **kw)
SYN = dict(bwd=1.0, harmonise=1.0, pool_k="resample", level="type", k_prof=20)
W = {
    "T0 library 3 as it is": lambda s: T(s),
    "T3 composition + round-7 shifts": lambda s: T(s, shares=comp(N), **shifts),
    "T4 all-synthetic round-7 construction": lambda s: predict7(e10, pool_t, anchor_t, N, 0.45, seed=s, **SYN, **shifts),
    "T4h round-7 construction, half of the nuclei real": lambda s: predict7(e10, pool_t, anchor_t, N, 0.45, seed=s, real_share=0.5, **SYN, **shifts),
    "L25c7 low genes 0.25 in all cells + composition + round-7 shifts": lambda s: T(s, mix=0.25, mix_low=0.3, shares=comp(N), **shifts),
    "L50c7 low genes 0.5 in all cells + composition + round-7 shifts": lambda s: T(s, mix=0.5, mix_low=0.3, shares=comp(N), **shifts),
    "LB45c7 low genes 1 in 45% of the cells + composition + round-7 shifts": lambda s: T(s, mix=1.0, mix_low=0.3, mix_cells=0.45, shares=comp(N), **shifts),
    "LB25c7 low genes 1 in 25% of the cells + composition + round-7 shifts": lambda s: T(s, mix=1.0, mix_low=0.3, mix_cells=0.25, shares=comp(N), **shifts),
    "LB45h low genes 0.5 in 45% of the cells + composition + round-7 shifts": lambda s: T(s, mix=0.5, mix_low=0.3, mix_cells=0.45, shares=comp(N), **shifts),
}
rows = []
for wname, make in W.items():
    for seed in (301, 302, 303):
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
