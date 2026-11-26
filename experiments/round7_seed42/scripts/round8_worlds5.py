"""Round 8, replication of the fourth set with a NEW random split (library 3 and the anchor nuclei are divided again into
a part for the hypothetical populations and a part for the files; new seeds for the populations), for the candidate of
10:53 (all added cells real + low-gene mixture 0.25). Worlds with and without the low-gene difference, three seeds each."""



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
tag = sys.argv[1] if len(sys.argv) > 1 else "w5"
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
    "A40: 40% raw nuclei, no correction": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.4, harmonise=0.0),
    "R7h: file recipe": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, **r7, **shifts),
    "R7 all synthetic": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, **{**r7, "real_share": 0.0}, **shifts),
    "R7 all real": lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, **{**r7, "real_share": 1.0}, **shifts),
}
allreal = {**r7, "real_share": 1.0}
F["candidate: all real + low genes 0.25"] = lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, mix=0.25, mix_low=0.3, **allreal, **shifts)
F["all real + low genes 0.25, measured molecules only"] = lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, mix=0.25, mix_low=0.3, mix_k="auto", **allreal, **shifts)
F["all real + low genes 0.35"] = lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, mix=0.35, mix_low=0.3, **allreal, **shifts)
F["all real + low genes 0.5"] = lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, mix=0.5, mix_low=0.3, **allreal, **shifts)
F["all real + low genes 0.5, below 0.1"] = lambda: predict7(e10, pool_f, anchor_f, 5000, 0.45, mix=0.5, mix_low=0.1, **allreal, **shifts)
files = {k: f() for k, f in F.items()}
print("files built", len(files), flush=True)
comp = lambda n: composition_shares(e10, pool_t, anchor_t, 0.45, 0.0, n, feasible_shares)
N = 2 * 638
T = lambda seed, **kw: predict7(e10, pool_t, anchor_t, N, 0.0, seed=seed, level="type", **kw)
SYN = dict(bwd=1.0, harmonise=1.0, pool_k="resample", level="type", k_prof=20)
W = {
    "T0 library 3 as it is": lambda s: T(s),
    "T1 expression moved (round-7 shifts)": lambda s: T(s, **shifts),
    "T2 composition moved (0.45)": lambda s: T(s, shares=comp(N)),
    "T3 composition + round-7 shifts": lambda s: T(s, shares=comp(N), **shifts),
    "S25c7 shared 0.25 + composition + round-7 shifts": lambda s: T(s, fwd=0.25, fwd_kind="dc", shares=comp(N), **shifts),
    "M20c7 mixture 0.2 + composition + round-7 shifts": lambda s: T(s, mix=0.2, shares=comp(N), **shifts),
    "T4 all-synthetic round-7 construction": lambda s: predict7(e10, pool_t, anchor_t, N, 0.45, seed=s, **SYN, **shifts),
    "L50 low genes 0.5": lambda s: T(s, mix=0.5, mix_low=0.3),
    "L25c7 low genes 0.25 + composition + round-7 shifts": lambda s: T(s, mix=0.25, mix_low=0.3, shares=comp(N), **shifts),
    "L50c7 low genes 0.5 + composition + round-7 shifts": lambda s: T(s, mix=0.5, mix_low=0.3, shares=comp(N), **shifts),
    "L100c7 low genes 1 + composition + round-7 shifts": lambda s: T(s, mix=1.0, mix_low=0.3, shares=comp(N), **shifts),
    "L50c7a low genes 0.5 (measured molecules) + composition + round-7 shifts": lambda s: T(s, mix=0.5, mix_low=0.3, mix_k="auto", shares=comp(N), **shifts),
}
rows = []
for wname, make in W.items():
    for seed in (201, 202, 203):
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
