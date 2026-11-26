"""Round 7 method: displacement interpolation between the source heart cells and the external E14.5 anchor.

Round 6 mixed unchanged source cells with harmonised anchor nuclei. Round 7 moves both populations towards
the position of the target on the source -> anchor path, lineage by lineage, before mixing them:

    prediction = (1 - f) x Forward_fwd(source cells)  +  f x Backward_bwd(Pool_k(harmonised anchor nuclei))

    lineage log-ratio   d_L[g] = log((mean anchor_L[g] + eps) / (mean source_L[g] + eps)) + c[g]   (counts per 10k;
                        c = the source/anchor log-ratio shared by all lineages, see round6_anchor; adding it is the
                        log-ratio the harmonised anchor would have, up to eps)
    Forward_fwd         source cell of lineage L: counts x exp(+fwd * d_L), renormalised to 10,000
    Backward_bwd        anchor nucleus of lineage L: counts x exp(h * c - bwd * d_L), renormalised
                        (c = shared anchor/source shift; with h = 1 and bwd = 1 the anchor lineage means equal the
                        source lineage means, with bwd = 0 this is the round-6 harmonised anchor)
    Pool_k              optional depth matching: the raw counts of every chosen anchor nucleus are summed with
                        those of its k-1 nearest anchor nuclei (the anchor was sequenced ~3.4x shallower than
                        the source heart library)

A displacement interpolation at position t on the path uses fwd = t and bwd = 1 - t: both parts then carry the
same lineage means, the geometric interpolation of source and anchor. Zeros stay zero in every cell.

Only the source stage (never a proxy target) and the anchor enter these estimates. `predict7` is the single code
path for validation and export.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import sparse

from experiments.round6_anchor import HARMONISE_LINEAGES, lineage_means
from experiments.round6_common import LINEAGES, SEED, centroid_transfer, depth_of, lineage_of
from experiments.round6_method import draw_base

EPS = 0.1
CLIP = 3.0
MIN_CELLS = 50
OWN_SHARE = 0.05
TILT_CLIP = 2.0


def scale_cells(cells, log_factor):
    """Scale counts gene-wise by exp(log_factor) (one row of factors, or one per cell) and renormalise to 10,000."""
    out = np.empty_like(cells, dtype=np.float32)
    per_cell = log_factor.ndim == 2
    for s in range(0, len(cells), 256):
        lf = log_factor[s:s + 256] if per_cell else log_factor
        c = np.expm1(cells[s:s + 256].astype(np.float64)) * np.exp(np.clip(lf, -CLIP, CLIP))
        tot = c.sum(axis=1, keepdims=True)
        out[s:s + 256] = np.log1p(c * np.divide(1e4, tot, out=np.ones_like(tot), where=tot > 0))
    return out


def soup_profile(genes, lo=10, hi=100):
    """Ambient-RNA profile of the anchor sample: summed counts of droplets with lo..hi UMIs that were not called
    as nuclei (the definition SoupX uses), in official gene order, normalised to sum 1."""
    import h5py
    import pandas as pd
    from experiments.round5_anchor import H5, METRICS
    metrics = pd.read_csv(METRICS)
    called = set(metrics.loc[metrics["is_cell"] == 1, "barcode"] if "is_cell" in metrics else metrics["barcode"])
    with h5py.File(H5, "r") as f:
        m = f["matrix"]
        barcodes = m["barcodes"][:].astype(str)
        names = m["features/name"][:].astype(str)
        kinds = m["features/feature_type"][:].astype(str)
        x = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()
    gex = np.flatnonzero(kinds == "Gene Expression")
    x = x[:, gex]
    tot = np.asarray(x.sum(axis=1)).ravel()
    empty = np.flatnonzero((tot >= lo) & (tot <= hi) & ~np.isin(barcodes, list(called)))
    counts = np.asarray(x[empty].sum(axis=0)).ravel().astype(np.float64)
    index = pd.Index(genes).get_indexer(names[gex])
    out = np.zeros(len(genes))
    np.add.at(out, index[index >= 0], counts[index >= 0])
    return out / out.sum(), {"empty_droplets": int(len(empty)), "umis": float(counts.sum())}


def leak_fractions(ms, ma, soup, top=400, specificity=0.03, min_genes=15):
    """Ambient fraction of every anchor lineage, from genes the lineage does not express in the source stage.

    For lineage L take the `top` most abundant soup genes whose source mean in L is below `specificity` x the
    largest source lineage mean (markers of other lineages). Whatever the anchor shows for them in L is ambient:
    rho_L = median(anchor mean_L / soup level). ms, ma: lineage -> mean counts per 10k; soup: profile summing to 1."""
    s = soup * 1e4
    top_genes = np.argsort(-s)[:top]
    best = np.max(np.stack(list(ms.values())), axis=0)
    rho, n_used = {}, {}
    for lin, mean_a in ma.items():
        if lin not in ms:
            continue
        genes = top_genes[(ms[lin][top_genes] <= specificity * best[top_genes]) & (best[top_genes] > 1.0)]
        if len(genes) >= min_genes:
            rho[lin] = float(np.clip(np.median(mean_a[genes] / s[genes]), 0.0, 0.8))
            n_used[lin] = int(len(genes))
    fallback = float(np.median(list(rho.values()))) if rho else 0.0
    return {lin: rho.get(lin, fallback) for lin in ma}, n_used


def remove_soup(mean, rho, soup):
    """Lineage mean (counts per 10k) without its ambient share, renormalised to 10,000."""
    out = np.maximum(mean - rho * soup * 1e4, 0.0)
    return out / out.sum() * 1e4


class Anchor7:
    """Anchor nuclei with cached lineage assignment, shared shift, lineage log-ratios and a neighbour graph.

    soup: ambient profile of the anchor sample (sums to 1). If given, the ambient share of every anchor lineage is
    estimated against the source stage and removed from the lineage means (and from re-sequenced nuclei)."""

    def __init__(self, x, soup=None):
        self.x = x.tocsr()
        self.depth = depth_of(self.x)
        self.soup = soup
        self._ctx = {}
        self._nn = None

    # -- depth matching ------------------------------------------------------------------------------------
    def neighbours(self, k_max=8, n_comp=40):
        """Indices of the k_max nearest anchor nuclei of every nucleus (itself first), in a truncated-SVD space."""
        if self._nn is None or self._nn.shape[1] < k_max:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.neighbors import NearestNeighbors
            if getattr(self, "_z", None) is None:
                svd = TruncatedSVD(n_components=n_comp, random_state=0)
                z = svd.fit_transform(self.x)
                self._svd, self._zraw = svd, z          # kept for `tilt` in resampled(); the graph below is unchanged
                self._z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
            self._nn = NearestNeighbors(n_neighbors=max(k_max, 16)).fit(self._z).kneighbors(self._z, return_distance=False)
        return self._nn

    def resampled(self, rows, log_factor, depths, k_prof=10, seed=SEED, rho=None, tilt=0.0):
        """Synthetic nuclei "re-sequenced" at the source's depth.

        For every chosen nucleus: raw counts of its k_prof nearest nuclei are summed, scaled gene-wise by
        exp(log_factor[row]) (harmonisation and backward shift) and normalised to a probability profile;
        `depths[row]` UMIs are then drawn from that profile (multinomial). Unlike scaling a single shallow
        nucleus, this gives the detection rates of a nucleus sequenced as deeply as the source cells.

        tilt > 0 restores what pooling averages away. The pooled profile is the mean of a neighbourhood, so
        nuclei drawn from it differ from one another far less than real cells of one type do. With tilt, the
        profile of nucleus i is multiplied gene-wise by exp(tilt x delta_i) / (neighbourhood mean of the same
        factor), where delta_i is the low-rank (truncated-SVD) reconstruction of how the nucleus itself differs
        from the mean of its neighbourhood in log1p units. The neighbourhood mean of the tilted profiles stays
        the pooled profile (so type means and the alignment are not moved); tilt x delta is clipped at +-TILT_CLIP."""
        rows = np.asarray(rows)
        nn = self.neighbours(k_prof)[:, :k_prof]
        rng = np.random.default_rng(seed)
        out = np.empty((len(rows), self.x.shape[1]), dtype=np.float32)
        for s in range(0, len(rows), 64):
            part = rows[s:s + 64]
            flat = nn[part].ravel()
            counts = np.expm1(self.x[flat].toarray().astype(np.float64)) * (self.depth[flat, None] / 1e4)
            prof = np.rint(counts).reshape(len(part), k_prof, -1).sum(axis=1)
            if rho is not None and self.soup is not None:      # ambient share of the pooled profile removed first
                prof = np.maximum(prof - rho[s:s + 64, None] * prof.sum(axis=1, keepdims=True) * self.soup[None, :], 0.0)
            if tilt:
                comp = self._svd.components_.astype(np.float64)
                zn = self._zraw[nn[part]].astype(np.float64)                      # (cells, k_prof, components)
                dev = zn - zn.mean(axis=1, keepdims=True)                         # every neighbour against the neighbourhood mean
                own = np.exp(np.clip(tilt * (dev[:, 0, :] @ comp), -TILT_CLIP, TILT_CLIP))   # neighbour 0 is the nucleus itself
                mean = np.zeros_like(own)
                for j in range(k_prof):
                    mean += np.exp(np.clip(tilt * (dev[:, j, :] @ comp), -TILT_CLIP, TILT_CLIP))
                prof *= own / (mean / k_prof)
            prof *= np.exp(np.clip(log_factor[s:s + 64], -CLIP, CLIP))
            prof /= prof.sum(axis=1, keepdims=True)
            n = np.maximum(np.rint(depths[s:s + 64]).astype(np.int64), 100)
            y = rng.multinomial(n, prof).astype(np.float64)
            out[s:s + 64] = np.log1p(y / n[:, None] * 1e4)
        return out

    def pooled(self, rows, k):
        """log1p(counts per 10k) of pooled nuclei: raw counts of each row summed with its k-1 nearest nuclei.

        k: one integer for all rows, or one integer per row (1 = the nucleus itself, unchanged)."""
        rows = np.asarray(rows)
        ks = np.full(len(rows), int(k)) if np.isscalar(k) else np.asarray(k, dtype=int)
        out = self.x[rows].toarray().astype(np.float32)
        if ks.max() <= 1:
            return out
        nn = self.neighbours(max(8, int(ks.max())))
        for kk in np.unique(ks[ks > 1]):
            idx = np.flatnonzero(ks == kk)
            for s in range(0, len(idx), 128):
                part = idx[s:s + 128]
                block = nn[rows[part], :kk]
                flat = block.ravel()
                counts = np.expm1(self.x[flat].toarray().astype(np.float64)) * (self.depth[flat, None] / 1e4)
                counts = np.rint(counts).reshape(len(part), kk, -1).sum(axis=1)
                tot = np.maximum(counts.sum(axis=1, keepdims=True), 1)
                out[part] = np.log1p(counts / tot * 1e4)
        return out

    def match_k(self, source, pool, ctx, k_max=6):
        """Per-nucleus pooling size that brings every anchor lineage to the median depth of the same source lineage."""
        rows = np.asarray(pool)
        lin_s = lineage_of(source.labels[rows])
        depth_s = depth_of(source.X[rows])
        overall = float(np.median(depth_s) / np.median(self.depth))
        ks = np.ones(self.x.shape[0], dtype=int)
        table = {}
        for lin in np.unique(ctx["lineage"]):
            a = ctx["lineage"] == lin
            ratio = float(np.median(depth_s[lin_s == lin]) / np.median(self.depth[a])) if (lin_s == lin).sum() >= MIN_CELLS else overall
            table[lin] = int(np.clip(round(ratio), 1, k_max))
            ks[a] = table[lin]
        return ks, table

    # -- what separates anchor and source, lineage by lineage ----------------------------------------------
    def context(self, source, pool, mode="min", level="lineage", prev=None, days_to_anchor=4.0):
        """Everything derived from one (source, pool): anchor lineages, shared shift c, lineage log-ratios d_L.

        prev = (previous stage, rows): adds the same-laboratory change of every lineage from the previous stage
        to the source ("vel", per day of the one-day step) and the two-witness developmental log-ratio "dev":
        a gene of a lineage is moved towards the anchor only if it also moved in that direction from the previous
        stage to the source, and by no more than `days_to_anchor` days at that earlier rate:
            dev_L[g] = sign x min(|d_L[g]|, days_to_anchor x |vel_L[g]|)   if sign(d_L[g]) = sign(vel_L[g]), else 0
        Differences between the two laboratories (ambient RNA, nuclear contamination, ribosomal content) do not
        show up in the same-laboratory step and are filtered out. Genes the lineage does not express itself in the
        source stage (below OWN_SHARE of the largest lineage mean) are never moved."""
        rows = np.ascontiguousarray(np.asarray(pool, dtype=np.int64))
        pkey = None if prev is None else (id(prev[0]), hashlib.sha1(np.ascontiguousarray(np.asarray(prev[1], dtype=np.int64)).tobytes()).hexdigest(), float(days_to_anchor))
        key = (id(source), hashlib.sha1(rows.tobytes()).hexdigest(), mode, level, pkey)
        if key in self._ctx:
            return self._ctx[key]
        labels = source.labels[rows]
        near = centroid_transfer(source.X[rows], labels, self.x)
        lin_a, lin_s = lineage_of(near), lineage_of(labels)
        ma, ms = lineage_means(self.x, lin_a, MIN_CELLS), lineage_means(source.X[rows], lin_s, MIN_CELLS)
        rho, rho_genes = {}, {}
        if self.soup is not None:
            rho, rho_genes = leak_fractions(ms, ma, self.soup)
            ma = {k: remove_soup(v, rho[k], self.soup) for k, v in ma.items()}
        used = [k for k in HARMONISE_LINEAGES if k in ma and k in ms]
        if len(used) < 3:
            raise ValueError(f"only {used} lineages are present on both sides")
        ratios = np.stack([np.log((ms[k] + EPS) / (ma[k] + EPS)) for k in used])
        sign = np.sign(np.median(ratios, axis=0))
        agree = (np.sign(ratios) == sign).all(axis=0) & (sign != 0)
        size = np.abs(ratios).min(axis=0) if mode == "min" else np.median(np.abs(ratios), axis=0)
        c = np.where(agree, sign * size, 0.0)                       # shared source/anchor shift (round-6 harmonise)
        # lineage log-ratio anchor' / source for every lineage present on both sides (also Mac etc. when large enough)
        d = {}
        for k in ma:
            if k in ms and k != "Other":
                d[k] = (-(np.log((ms[k] + EPS) / (ma[k] + EPS)) - c)).astype(np.float32)
        if level == "type":                                         # finer: per source cell type where both sides are large
            tm_a, tm_s = lineage_means(self.x, near, MIN_CELLS), lineage_means(source.X[rows], labels, MIN_CELLS)
            if self.soup is not None:
                tm_a = {t: remove_soup(v, rho.get(lineage_of([t])[0], 0.0), self.soup) for t, v in tm_a.items()}
            for t in tm_a:
                if t in tm_s and lineage_of([t])[0] != "Other":
                    d["type:" + t] = (-(np.log((tm_s[t] + EPS) / (tm_a[t] + EPS)) - c)).astype(np.float32)
        ctx = {"near": near, "lineage": lin_a, "c": c.astype(np.float32), "d": d, "used": used, "rho": rho, "rho_genes": rho_genes,
               "anchor_lineage_share": {k: float((lin_a == k).mean()) for k in np.unique(lin_a)}}
        if prev is not None:
            pstage, prows = prev
            prows = np.asarray(prows)
            mp = lineage_means(pstage.X[prows], lineage_of(pstage.labels[prows]), MIN_CELLS)
            vel, dev = {}, {}
            # a lineage is moved only in genes it expresses itself: genes whose source mean in the lineage is below
            # OWN_SHARE of the largest source lineage mean are another lineage's markers; any rise there is ambient
            # RNA or doublets (it shows in both witnesses for the same technical reason) and is not moved
            best = np.max(np.stack(list(ms.values())), axis=0)
            for k in d:
                if k.startswith("type:") or k not in mp or k not in ms:
                    continue
                v = np.log((ms[k] + EPS) / (mp[k] + EPS))
                same = (np.sign(v) == np.sign(d[k])) & (v != 0) & (ms[k] >= OWN_SHARE * best)
                vel[k] = v.astype(np.float32)
                dev[k] = np.where(same, np.sign(d[k]) * np.minimum(np.abs(d[k]), days_to_anchor * np.abs(v)), 0.0).astype(np.float32)
            ctx["vel"], ctx["dev"] = vel, dev
        self._ctx[key] = ctx
        return ctx


def composition_shares(source, main, anchor, tc, f, n, feasible=None):
    """Lineage quotas of the base so that the TOTAL composition of the file sits at position tc on the line from
    the source's lineage shares to the anchor's: total = (1 - tc) x source + tc x anchor; the f anchor nuclei
    come in the anchor's own composition, the base supplies the rest. tc = f reproduces the natural base."""
    from experiments.round6_method import lineage_shares
    pi_s = lineage_shares(source, main)
    pi_a = anchor.context(source, main)["anchor_lineage_share"]
    base = {k: max((1 - tc) * pi_s[k] + (tc - f) * pi_a.get(k, 0.0), 0.0) for k in LINEAGES}
    tot = sum(base.values())
    shares = {k: v / tot for k, v in base.items()}
    return feasible(source, main, shares, int(round(n * (1 - f)))) if feasible is not None else shares


def _factors(keys_type, keys_lineage, d, sign_strength, n_genes):
    """Per-cell log-factor rows: sign_strength * d[type] if present, else * d[lineage], else 0."""
    out = np.zeros((len(keys_lineage), n_genes), dtype=np.float32)
    for i, (t, lin) in enumerate(zip(keys_type, keys_lineage)):
        v = d.get("type:" + t, d.get(lin))
        if v is not None:
            out[i] = sign_strength * v
    return out


def predict7(source, pool, anchor, n_total, f, *, fwd=0.0, bwd=0.0, harmonise=1.0, mode="min", level="lineage",
             pool_k=1, anchor_rows=None, shares=None, seed=SEED, context_pool=None, return_info=False,
             prev=None, days_to_anchor=4.0, dev=0.0, k_prof=10, trend=None, trend_share=0.0, anchor_shift=None, tilt=0.0,
             depth_from="source"):
    """Cells of the prediction (float32, n_total x genes): shifted base cells first, shifted anchor nuclei last.

    pool          source rows the base is drawn from (uniformly, or to lineage `shares`)
    context_pool  source rows the lineage statistics are estimated on (default: `pool`)
    dev, trend_share  developmental shifts of the BASE cells (two-witness shift, within-cell bulk trend)
    tilt          strength of the individual low-rank deviation restored in re-sequenced nuclei (see Anchor7.resampled)
    depth_from    "source" (default: depth of a source cell of the same lineage) or "anchor" (the nucleus's own depth; analysis only)
    anchor_shift  share of those two shifts that the anchor nuclei receive as well. Default: 1 if the nuclei were
                  moved back completely (bwd = 1: they sit at the source's means, like the base, and move with it),
                  0 otherwise (nuclei that keep E14.5 expression are not pushed further forward).
    """
    rng = np.random.default_rng(seed)
    k = int(round(n_total * f))
    rows = draw_base(source, pool, n_total - k, rng, shares)
    cells = source.dense(rows)
    need_ctx = bool(k and (harmonise or bwd or pool_k == "resample")) or bool(fwd) or bool(dev)
    ckw = dict(prev=prev, days_to_anchor=days_to_anchor) if dev else {}
    ctx = anchor.context(source, pool if context_pool is None else context_pool, mode, level, **ckw) if need_ctx else None
    n_genes = cells.shape[1]
    if fwd:
        labels = source.labels[rows]
        cells = scale_cells(cells, _factors(labels, lineage_of(labels), ctx["d"], fwd, n_genes))
    if dev:   # two-witness developmental shift: `dev` = share of the source -> anchor path (e.g. 0.5 for E12.5)
        labels = source.labels[rows]
        cells = scale_cells(cells, _factors(labels, lineage_of(labels), ctx["dev"], dev, n_genes))
    a_shift = (1.0 if bwd >= 1.0 else 0.0) if anchor_shift is None else float(anchor_shift)
    if trend_share:
        # within-cell bulk trend (same-laboratory E9.5 -> E14.5 whole-heart log-ratio, round 2): every count is scaled
        # by exp(trend x trend_share), trend_share = share of the five-day bulk interval; row-wise, zeros stay zero
        from experiments.bulk_trend import apply_trend
        cells = apply_trend(cells, trend, trend_share, False, seed)
    picked = np.array([], dtype=int)
    if k:
        candidates = np.arange(anchor.x.shape[0]) if anchor_rows is None else np.asarray(anchor_rows)
        picked = np.sort(candidates[rng.choice(len(candidates), k, replace=False)])
        cpool = pool if context_pool is None else context_pool
        if pool_k == "resample":
            ctx = anchor.context(source, cpool, mode, level, **ckw)
            lf = np.tile(harmonise * ctx["c"], (k, 1)) if harmonise else np.zeros((k, n_genes), dtype=np.float32)
            if bwd:
                lf = lf + _factors(ctx["near"][picked], ctx["lineage"][picked], ctx["d"], -bwd, n_genes)
            if dev and a_shift:
                lf = lf + _factors(ctx["near"][picked], ctx["lineage"][picked], ctx["dev"], dev * a_shift, n_genes)
            # depth of every synthetic nucleus: a random source cell of the same lineage (whole pool if the lineage is rare)
            srows = np.asarray(cpool)
            depth_s, lin_s = depth_of(source.X[srows]), lineage_of(source.labels[srows])
            depths = np.empty(k)
            for lin in np.unique(ctx["lineage"][picked]):
                sel = np.flatnonzero(ctx["lineage"][picked] == lin)
                cand = depth_s[lin_s == lin] if (lin_s == lin).sum() >= MIN_CELLS and lin != "Other" else depth_s[lin_s != "Blood"]
                depths[sel] = rng.choice(cand, len(sel), replace=True)
            if depth_from == "anchor":      # analysis only: re-sequence every nucleus at its own (shallow) depth
                depths = anchor.depth[picked].astype(np.float64)
            rho_rows = np.array([ctx["rho"].get(lin, 0.0) for lin in ctx["lineage"][picked]]) if ctx.get("rho") else None
            nuclei = anchor.resampled(picked, lf, depths, k_prof=k_prof, seed=seed, rho=rho_rows, tilt=tilt)
        else:
            if pool_k == "match":
                ks, _ = anchor.match_k(source, cpool, anchor.context(source, cpool, mode, level))
                nuclei = anchor.pooled(picked, ks[picked])
            else:
                nuclei = anchor.pooled(picked, pool_k)
            if harmonise or bwd or (dev and a_shift):
                lf = np.tile(harmonise * ctx["c"], (k, 1)) if harmonise else np.zeros((k, n_genes), dtype=np.float32)
                if bwd:
                    lf = lf + _factors(ctx["near"][picked], ctx["lineage"][picked], ctx["d"], -bwd, n_genes)
                if dev and a_shift:
                    lf = lf + _factors(ctx["near"][picked], ctx["lineage"][picked], ctx["dev"], dev * a_shift, n_genes)
                nuclei = scale_cells(nuclei, lf)
        if trend_share and a_shift:
            nuclei = apply_trend(nuclei, trend, trend_share * a_shift, False, seed)
        cells = np.vstack([cells, nuclei])
    if return_info:
        return cells, {"base_rows": rows, "anchor_rows": picked, "f": float(f)}
    return cells
