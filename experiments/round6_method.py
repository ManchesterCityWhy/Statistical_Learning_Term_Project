"""Round 6 method, one code path for validation and export.

    prediction = (1 - f) x cells of the source stage  +  f x real E14.5 nuclei (the external anchor)

The source part is drawn from a `pool` of source rows, either uniformly ("natural": the pool's own composition)
or to given lineage shares; an optional within-cell bulk trend can be applied to it. The anchor part is drawn
uniformly from the anchor nuclei; optionally the change shared by all lineages between anchor and source (the
other laboratory's nuclei preparation, ambient RNA, sex) is removed from it (`harmonise`).

Validation calls `predict` with a proxy's source stage; export calls the same function with E10.5.
"""
from __future__ import annotations

import hashlib

import numpy as np

from experiments.bulk_trend import apply_trend
from experiments.round6_anchor import common_shift, harmonise
from experiments.round6_common import LINEAGES, SEED, lineage_of


def lineage_shares(source, rows):
    lin = lineage_of(source.labels[np.asarray(rows)])
    return {k: float((lin == k).mean()) for k in LINEAGES}


def draw_base(source, pool, n, rng, shares=None, shallow_pool=None, shallow_share=0.0):
    """n distinct source rows from `pool`.

    pool=[(rows, share), ...]   library quotas, each part drawn uniformly (no lineage quotas in this mode)
    shares=None        uniform over the pool (its natural composition)
    shares={lineage:p} lineage quotas; uniform within a lineage
    shallow_pool       second pool (same lineage quotas as realised in the first draw); a share `shallow_share`
                       of every lineage is taken from it instead (depth hedge)
    """
    if isinstance(pool, (list, tuple)) and len(pool) and isinstance(pool[0], tuple):
        # library quotas: [(rows, share), ...], each part drawn uniformly
        counts = [int(round(n * share)) for _, share in pool]
        counts[int(np.argmax(counts))] += n - sum(counts)
        return np.sort(np.concatenate([np.asarray(r)[rng.choice(len(r), c, replace=False)] for (r, _), c in zip(pool, counts) if c > 0]))
    pool = np.asarray(pool)
    if shares is None and shallow_share == 0:
        return np.sort(pool[rng.choice(len(pool), n, replace=False)])
    lin = lineage_of(source.labels)
    if shares is None:
        shares = lineage_shares(source, pool)
    quota = {k: int(round(n * shares.get(k, 0.0))) for k in LINEAGES}
    biggest = max(quota, key=quota.get)
    quota[biggest] += n - sum(quota.values())
    rows = []
    for k, q in quota.items():
        if q <= 0:
            continue
        main = pool[lin[pool] == k]
        q2 = 0
        if shallow_pool is not None and shallow_share > 0:
            alt = np.asarray(shallow_pool)[lin[np.asarray(shallow_pool)] == k]
            q2 = min(int(round(q * shallow_share)), len(alt))
            if q2:
                rows.append(alt[rng.choice(len(alt), q2, replace=False)])
        q1 = q - q2
        if q1 > len(main):
            raise ValueError(f"lineage {k}: {q1} cells requested, {len(main)} available")
        rows.append(main[rng.choice(len(main), q1, replace=False)])
    return np.sort(np.concatenate(rows))


class Anchor:
    """The anchor nuclei plus what is needed to harmonise them against one source pool (cached per pool)."""

    def __init__(self, x):
        self.x = x.tocsr()
        self._shift = {}

    def shift(self, source, pool, key, mode="min"):
        """Shared anchor/source shift for one (source, pool). `key` is a readable label; the cache is keyed on the
        source object, the pool rows themselves and the mode, so a label can never return another pool's shift."""
        rows = np.ascontiguousarray(np.asarray(pool, dtype=np.int64))
        full = (id(source), hashlib.sha1(rows.tobytes()).hexdigest(), mode)
        if full not in self._shift:
            self._shift[full] = common_shift(source, rows, self.x, mode=mode)[0]
        return self._shift[full]


def predict(source, pool, anchor, n_total, f, *, seed=SEED, trend=None, trend_strength=0.0, shares=None,
            shallow_pool=None, shallow_share=0.0, harmonise_strength=0.0, shift_key=None, anchor_rows=None,
            harmonise_mode="min", return_rows=False):
    """Cells of the prediction (float32, n_total x genes). Base cells first, anchor nuclei last."""
    rng = np.random.default_rng(seed)
    k = int(round(n_total * f))
    rows = draw_base(source, pool, n_total - k, rng, shares, shallow_pool, shallow_share)
    cells = source.dense(rows)
    if trend_strength:
        cells = apply_trend(cells, trend, trend_strength / 5, False, seed)
    picked = np.array([], dtype=int)
    if k:
        candidates = np.arange(anchor.x.shape[0]) if anchor_rows is None else np.asarray(anchor_rows)
        picked = np.sort(candidates[rng.choice(len(candidates), k, replace=False)])
        nuclei = anchor.x[picked].toarray().astype(np.float32)
        if harmonise_strength:
            if shift_key is None:
                raise ValueError("harmonise_strength > 0 needs a shift_key naming the (source, pool) pair")
            parts = isinstance(pool, (list, tuple)) and len(pool) and isinstance(pool[0], tuple)
            shift_pool = np.concatenate([np.asarray(r) for r, _ in pool]) if parts else pool
            nuclei = harmonise(nuclei, anchor.shift(source, shift_pool, shift_key, harmonise_mode), harmonise_strength)
        cells = np.vstack([cells, nuclei])
    return (cells, rows, picked) if return_rows else cells
