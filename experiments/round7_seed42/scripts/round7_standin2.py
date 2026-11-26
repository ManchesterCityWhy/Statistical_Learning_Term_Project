"""Round 7: stand-in comparison with the cell-type composition controlled.

As round7_standin, but both stand-ins are drawn to the TYPE composition of the target lineage and aligned type by
type, so that only the cell-level shape differs:
    (i)   real cells of the earlier stage, thinned to the target depth, assigned to the nearest target-stage cell type
          (centroid correlation, the method used for the anchor nuclei), drawn to the target's type shares, scaled
          gene-wise to the target mean of the assigned type
    (ii)  synthetic nuclei: anchor nuclei assigned to the same types, drawn to the same type shares, 20 neighbours
          pooled, moved back to the target mean of the assigned type, re-sequenced at the target depth
Types with fewer than 30 target cells in the alignment part, or fewer than 15 candidates on either side, are left out
of both stand-ins and of the truth. Reference of the evaluator: the 10% E8.5 sample.
"""
import numpy as np, pandas as pd
from scipy import sparse
from experiments.round5_anchor import load_anchor
from experiments.round6_common import HEART, REAL, SEED, FastEval, Mem, centroid_transfer, depth_of, lineage_of, thin_rows
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
    lab_cur = cur.labels[cur_rows]
    lin_cur = lineage_of(lab_cur)
    lin_prev = lineage_of(prev.labels)
    perm = rng.permutation(len(cur_rows))
    part = np.empty(len(cur_rows), dtype=int)
    part[perm] = np.arange(len(cur_rows)) % 3
    a_rows = cur_rows[part == 0]
    anchor = Anchor7(ax)
    ctx = anchor.context(cur, a_rows, "min", "type")
    for L in ("CM", "Endo", "Epi"):
        inL = lin_cur == L
        A, T, C = (cur_rows[(part == k) & inL] for k in (0, 1, 2))
        depth_a = depth_of(cur.X[A])
        # earlier-stage cells of the lineage, thinned to the target depth, assigned to the target types of this lineage
        p_rows = np.flatnonzero((lin_prev == L) & np.isin(prev.labels, list(HEART)))
        thinned = sparse.csr_matrix(thin_rows(prev.X[p_rows], depth_a, rng))
        near_prev = centroid_transfer(cur.X[A], cur.labels[A], thinned)
        near_nuc, lin_nuc = ctx["near"], ctx["lineage"]
        types, counts = np.unique(cur.labels[A], return_counts=True)
        ok = [t for t, c in zip(types, counts) if c >= 30 and (near_prev == t).sum() >= 15 and ((near_nuc == t) & (lin_nuc == L)).sum() >= 15]
        if not ok:
            print(pair, L, "no usable type"); continue
        share = np.array([(cur.labels[A] == t).sum() for t in ok], dtype=float)
        share /= share.sum()
        keepT, keepC = T[np.isin(cur.labels[T], ok)], C[np.isin(cur.labels[C], ok)]
        n = int(min(N, len(keepT), len(keepC)))
        if n < 150:
            print(pair, L, "skipped: too few cells", len(keepT), len(keepC)); continue
        # truth and ceiling drawn to the same type shares as the stand-ins
        quota = np.floor(share * n).astype(int)
        quota[np.argmax(share)] += n - quota.sum()
        def draw(rows_, labels_, with_replacement=False):
            out = []
            for t, q in zip(ok, quota):
                cand = rows_[labels_ == t]
                out.append(rng.choice(cand, q, replace=with_replacement or len(cand) < q))
            return np.concatenate(out)
        tT, tC = draw(keepT, cur.labels[keepT]), draw(keepC, cur.labels[keepC])
        ev = FastEval(ref, cur.dense(tT), cur.dense(tC))
        sets = {"target stage itself (held-out third)": cur.dense(draw(A, cur.labels[A]))}
        # (i) real cells of the earlier stage
        idx = draw(np.arange(len(p_rows)), near_prev)
        cells = thinned[idx].toarray().astype(np.float32)
        lab_i = near_prev[idx]
        lf = np.zeros((len(idx), len(genes)), dtype=np.float32)
        for t in ok:
            mean_a = np.asarray(cur.X[A[cur.labels[A] == t]].expm1().mean(axis=0)).ravel()
            mean_p = np.asarray(thinned[np.flatnonzero(near_prev == t)].expm1().mean(axis=0)).ravel()
            lf[lab_i == t] = np.clip(np.log((mean_a + EPS) / (mean_p + EPS)), -CLIP, CLIP)
        sets["(i) real cells of the earlier stage, aligned by type"] = scale_cells(cells, lf)
        # (ii) synthetic nuclei
        nuc = draw(np.arange(ax.shape[0])[lin_nuc == L], near_nuc[lin_nuc == L])
        order = np.argsort(nuc)
        nuc = nuc[order]
        lfn = np.tile(ctx["c"], (len(nuc), 1)) + _factors(near_nuc[nuc], lin_nuc[nuc], ctx["d"], -1.0, ax.shape[1])
        sets["(ii) synthetic nuclei (round-7 form), aligned by type"] = anchor.resampled(nuc, lfn, rng.choice(depth_a, len(nuc), replace=True), k_prof=20, seed=SEED)
        for name, cells_ in sets.items():
            s = ev(cells_)
            rows_out.append(dict(pair=pair, lineage=L, n=n, types=len(ok), stand_in=name, score=s["score"], DES=s["skill_de_score"], DCS=s["skill_de_direction"],
                                 MMD=s["skill_mmd_u"], CSS=s["skill_variogram"]))
        print(pair, L, "done: cells", n, "types", ok, "shares", share.round(2).tolist(), flush=True)
frame = pd.DataFrame(rows_out)
out_dir = REAL / "runs" / "round7_standin"
out_dir.mkdir(parents=True, exist_ok=True)
frame.to_csv(out_dir / "standin_types.csv", index=False)
print(frame.round(3).to_string(index=False))
p = frame.pivot_table(index=["pair", "lineage"], columns="stand_in", values="score", sort=False)
d = p["(i) real cells of the earlier stage, aligned by type"] - p["(ii) synthetic nuclei (round-7 form), aligned by type"]
print("\n(i) minus (ii) by lineage:"); print(d.round(2).to_string())
print("\nmean over the three lineages:"); print(d.groupby(level="pair", sort=False).mean().round(2).to_string())
