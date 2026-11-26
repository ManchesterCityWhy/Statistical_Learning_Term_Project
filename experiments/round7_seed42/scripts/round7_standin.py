"""Round 7: which is the better stand-in for a real cell of the target stage - a real cell of the neighbouring
earlier stage, or a synthetic nucleus re-sequenced from the E14.5 anchor?

Both stand-ins are brought to the same lineage mean (estimated on a held-out third of the target-stage cells) and to
the same depth distribution, so what differs is the cell-level shape only.

    pairs     E9.5 -> E10.5 library 3 (without blood);  E8.5 -> E9.5
    lineages  CM, Endo, Epi (each scored separately against its own truth half / ceiling half)
    (i)   real cells of the earlier stage, thinned to the target depth, scaled gene-wise to the target lineage mean
    (ii)  synthetic nuclei: anchor nuclei of the lineage, 20 neighbours pooled, moved back to the target lineage
          mean, re-sequenced at the target depth (the round-7 form, lineage-level alignment)
    refs  real cells of the target stage itself (the held-out third); raw anchor nuclei (harmonised only)
Reference of the evaluator: the 10% E8.5 sample. Nothing here enters a prediction.
"""
import copy
import numpy as np, pandas as pd
from scipy import sparse
from experiments.round5_anchor import load_anchor
from experiments.round6_common import HEART, REAL, SEED, FastEval, Mem, depth_of, lineage_of, thin_rows
from experiments.round7_method import Anchor7, EPS, CLIP, _factors, scale_cells
from t1_pipeline.data import board_spec
pd.set_option("display.width", 250, "display.max_rows", 200)
_, genes = board_spec("T1:val")
ax, _ = load_anchor(genes, verbose=False)
e8 = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
ref = e8.dense(np.sort(np.random.default_rng(SEED).choice(e8.n_cells, round(.1 * e8.n_cells), replace=False)))
N = 500
rows_out = []
for pair, prev, cur, cur_rows in (("E9.5 -> E10.5 library 3", e9, e10, np.flatnonzero((e10.lib == "3") & (e10.labels != "Blood"))),
                                  ("E8.5 -> E9.5", e8, e9, np.flatnonzero(e9.labels != "Blood"))):
    rng = np.random.default_rng(SEED)
    lin_cur = lineage_of(cur.labels[cur_rows])
    lin_prev = lineage_of(prev.labels)
    # three disjoint parts of the target-stage cells: alignment part A, truth T, ceiling C
    perm = rng.permutation(len(cur_rows))
    part = np.empty(len(cur_rows), dtype=int)
    part[perm] = np.arange(len(cur_rows)) % 3
    a_rows = cur_rows[part == 0]
    # anchor context against the alignment part (lineage level): assignment, shared shift, lineage log-ratios
    anchor = Anchor7(ax)
    ctx = anchor.context(cur, a_rows, "min", "lineage")
    for L in ("CM", "Endo", "Epi"):
        A, T, C = (cur_rows[(part == k) & (lin_cur == L)] for k in (0, 1, 2))
        n = int(min(N, len(T), len(C), len(A)))
        if n < 150:
            print(pair, L, "skipped: too few cells", len(A), len(T), len(C)); continue
        T, C = rng.choice(T, n, replace=False), rng.choice(C, n, replace=False)
        ev = FastEval(ref, cur.dense(T), cur.dense(C))
        depth_a = depth_of(cur.X[A])
        mean_a = np.asarray(cur.X[A].expm1().mean(axis=0)).ravel()
        out = {}
        # reference: the target stage's own held-out cells
        out["target stage itself (held-out third)"] = cur.dense(rng.choice(A, n, replace=False))
        # (i) real cells of the earlier stage: thin to the target depth, scale to the target lineage mean
        p_rows = np.flatnonzero((lin_prev == L) & np.isin(prev.labels, list(HEART)))
        p_rows = rng.choice(p_rows, min(len(p_rows), 3 * n), replace=False)
        thinned = sparse.csr_matrix(thin_rows(prev.X[p_rows], depth_a, rng))
        mean_p = np.asarray(thinned.expm1().mean(axis=0)).ravel()
        lf = np.clip(np.log((mean_a + EPS) / (mean_p + EPS)), -CLIP, CLIP)
        pick = rng.choice(len(p_rows), n, replace=False)
        out["(i) real cells of the earlier stage, aligned"] = scale_cells(thinned[pick].toarray().astype(np.float32), lf)
        out["     the same, not aligned (thinned only)"] = thinned[pick].toarray().astype(np.float32)
        # (ii) synthetic nuclei of the lineage: pooled, moved back to the target lineage mean, re-sequenced
        cand = np.flatnonzero(ctx["lineage"] == L)
        picked = np.sort(rng.choice(cand, n, replace=len(cand) < n))
        lfn = np.tile(ctx["c"], (n, 1)) + _factors(ctx["near"][picked], ctx["lineage"][picked], ctx["d"], -1.0, ax.shape[1])
        depths = rng.choice(depth_a, n, replace=True)
        out["(ii) synthetic nuclei (round-7 form), aligned"] = anchor.resampled(picked, lfn, depths, k_prof=20, seed=SEED)
        out["     raw anchor nuclei (harmonised only)"] = scale_cells(ax[picked].toarray().astype(np.float32), np.tile(ctx["c"], (n, 1)))
        for name, cells in out.items():
            s = ev(cells)
            rows_out.append(dict(pair=pair, lineage=L, n=n, stand_in=name, score=s["score"], DES=s["skill_de_score"], DCS=s["skill_de_direction"],
                                 MMD=s["skill_mmd_u"], CSS=s["skill_variogram"], genes=float(np.median((cells > 0).sum(1)))))
        print(pair, L, "done, cells per set", n, "| earlier-stage cells available", len(p_rows), "| nuclei of the lineage", len(cand), flush=True)
frame = pd.DataFrame(rows_out)
out_dir = REAL / "runs" / "round7_standin"
out_dir.mkdir(parents=True, exist_ok=True)
frame.to_csv(out_dir / "standin.csv", index=False)
print(frame.round(3).to_string(index=False))
p = frame.pivot_table(index=["pair", "lineage"], columns="stand_in", values="score", sort=False)
d = p["(i) real cells of the earlier stage, aligned"] - p["(ii) synthetic nuclei (round-7 form), aligned"]
print("\n(i) minus (ii) by lineage:"); print(d.round(2).to_string())
print("\nmean over the three lineages:"); print(d.groupby(level="pair", sort=False).mean().round(2).to_string())
