"""Round 7: why do the E10.5-target proxies prefer un-pooled, shallow anchor nuclei? A control with shared lineages.

round7_clean removed two depth coincidences (the sparse blood cells of library 3, the deeper E8.5/E9.5 base) and
the round-6 form (raw nuclei) still beat the round-7 form (re-sequenced, moved back to the source's cell-type
means) by about 3 points. One candidate reason: the target E10.5 holds lineages the source stage does not have
(cushion mesenchyme and macrophages are 0% at E8.5/E9.5). Nuclei of such lineages are assigned to some other source
type and, when moved back, lose their own expression; raw nuclei keep it. The test is different: its source
(E10.5 library 3) already holds every lineage of the anchor.

Two targets, both with a depth-matched base (as in round7_clean), no developmental shifts, anchor share rho x elapsed:
    A   library 3 without blood (as round7_clean)
    B   library 3 restricted to the lineages the source has (CM, Endo, Epi, Other); the anchor nuclei are restricted
        the same way (lineage by their nearest library-3 cell type)
Forms: raw nuclei (round 6), re-sequenced nuclei moved back 1 / 0.5 / 0 (k_prof 20).

    python -m experiments.round7_clean2 --source E9.5|E8.5 [--split confirmation]
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
from scipy import sparse

from experiments.round5_anchor import load_anchor
from experiments.round5_common import h3_rows
from experiments.round6_common import HEART, REAL, SEED, FastEval, Log, Mem, depth_of, lineage_of, thin_rows
from experiments.round7_method import Anchor7, predict7
from t1_pipeline.data import board_spec

N = 2000
SHARED = ("CM", "Endo", "Epi", "Other")
RHO = 0.9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="E9.5")
    parser.add_argument("--split", default="development", choices=("development", "confirmation"))
    parser.add_argument("--tilt", type=float, default=0.0, help="also score the re-sequenced forms with this tilt (separate table)")
    args = parser.parse_args()
    a, b = ("development", "ceiling") if args.split == "development" else ("confirmation", "confirmation_ceiling")
    _, genes = board_spec("T1:val")
    ax, _ = load_anchor(genes, verbose=False)
    t0 = time.time()
    e8 = Mem(REAL / "data/T1/E8.5_RNA.h5ad", genes)
    e10 = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
    is9 = args.source == "E9.5"
    src = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes) if is9 else e8
    elapsed = .2 if is9 else 1 / 3
    ref = e8.dense(np.sort(np.random.default_rng(SEED).choice(e8.n_cells, round(.1 * e8.n_cells), replace=False)))
    h3 = h3_rows(e10)
    lib3 = np.flatnonzero(e10.lib == "3")
    # lineage of every anchor nucleus by its nearest library-3 cell type (what the test's own context would say)
    lin_anchor3 = Anchor7(ax).context(e10, lib3, "min", "type")["lineage"]
    pool = np.flatnonzero(np.isin(src.labels, list(HEART)) & (src.labels != "Blood"))
    lin_s = lineage_of(src.labels)
    print(f"source {args.source}: base pool {len(pool)} cells, lineage shares "
          + ", ".join(f"{k} {(lin_s[pool] == k).mean():.3f}" for k in np.unique(lin_s[pool]))
          + "; anchor lineage shares (by library-3 type) " + ", ".join(f"{k} {(lin_anchor3 == k).mean():.3f}" for k in np.unique(lin_anchor3)), flush=True)
    log = Log(REAL / "runs" / "round7_clean2" / f"scores{'_tilt' if args.tilt else ''}_{args.source}_{args.split}.csv")
    for target_name, keep in (("A: library 3 without blood", lambda lab: lab != "Blood"),
                              ("B: shared lineages only", lambda lab: np.isin(lineage_of(lab), SHARED) & (lab != "Blood"))):
        ta, tb = h3[a][keep(e10.labels[h3[a]])], h3[b][keep(e10.labels[h3[b]])]
        ev = FastEval(ref, e10.dense(ta), e10.dense(tb))
        tgt_rows = lib3[keep(e10.labels[lib3])]
        lin_t, depth_t = lineage_of(e10.labels[tgt_rows]), depth_of(e10.X[tgt_rows])
        # depth-matched copy of the source without a LIL matrix (four processes with two LIL copies each ran the
        # machine out of memory): thinned blocks are stacked and put back into the original row order
        rng = np.random.default_rng(SEED)
        blocks, order = [], []
        for lin in np.unique(lin_s[pool]):
            rows = pool[lin_s[pool] == lin]
            tgt = depth_t[lin_t == lin] if (lin_t == lin).sum() >= 50 else depth_t
            blocks.append(sparse.csr_matrix(thin_rows(src.X[rows], tgt, rng)))
            order.append(rows)
        rest = np.setdiff1d(np.arange(src.n_cells), pool)
        blocks.append(src.X[rest].tocsr())
        order.append(rest)
        order = np.concatenate(order)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        dm = copy.copy(src)
        dm.X = sparse.vstack(blocks, format="csr")[inverse]
        del blocks
        anchor_rows = None if target_name.startswith("A") else np.flatnonzero(np.isin(lin_anchor3, SHARED))
        print(f"[{target_name}] target {len(ta)} cells (median UMI {np.median(depth_of(e10.X[ta])):.0f}), lineage shares "
              + ", ".join(f"{k} {(lineage_of(e10.labels[ta]) == k).mean():.3f}" for k in np.unique(lineage_of(e10.labels[ta])))
              + f"; anchor nuclei allowed {ax.shape[0] if anchor_rows is None else len(anchor_rows)}; base median UMI {np.median(depth_of(dm.X[pool])):.0f}  {time.time() - t0:.0f}s", flush=True)
        anchor = Anchor7(ax)
        rs = dict(pool_k="resample", level="type", k_prof=20)
        cfgs = [("base only", 0.0, {}),
                ("raw nuclei (round 6 form)", RHO, {}),
                ("re-sequenced, moved back 1 (round 7 form)", RHO, dict(bwd=1.0, **rs)),
                ("re-sequenced, moved back 0.5", RHO, dict(bwd=.5, **rs)),
                ("re-sequenced, not moved back", RHO, dict(bwd=0.0, **rs))]
        if args.tilt:
            cfgs = [cfgs[0], cfgs[1], cfgs[2], cfgs[4],
                    (f"re-sequenced, moved back 1, tilt {args.tilt:g}", RHO, dict(bwd=1.0, tilt=args.tilt, **rs)),
                    (f"re-sequenced, not moved back, tilt {args.tilt:g}", RHO, dict(bwd=0.0, tilt=args.tilt, **rs))]
        for name, rho, kw in cfgs:
            cells = predict7(dm, pool, anchor, N, rho * elapsed, seed=SEED, anchor_rows=anchor_rows, **kw)
            log.add(ev, cells, target=target_name, config=name, f=round(rho * elapsed, 4))
    frame = pd.DataFrame(log.records)
    pd.set_option("display.width", 250)
    print(frame.pivot_table(index="config", columns="target", values="score", sort=False).round(2).to_string())
    for m in ("skill_de_score", "skill_de_direction", "skill_mmd_u", "skill_variogram"):
        print(m); print(frame.pivot_table(index="config", columns="target", values=m, sort=False).round(3).to_string())
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
