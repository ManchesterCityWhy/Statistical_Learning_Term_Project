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


class SourceCounts:
    """UMI counts of the source pool with a neighbour graph: shifts real source cells in count space (round 8).

    scale_cells multiplies the counts a cell already has, so a gene that rises stays undetected wherever it was
    undetected, and a gene that falls keeps its detection rate. Here a falling gene loses molecules (binomial
    thinning of the cell's own counts, as in a shallower capture of the same cell) and a rising gene gains molecules
    (Poisson, rate (factor - 1) x the cell's expected count; the expected count is the mean of the cell's own count
    and what its k nearest source cells show at the cell's depth, so new molecules can land in cells that had none).
    The cell is then renormalised to 10,000 and log1p-transformed, like every released cell."""

    _cache = {}

    def __init__(self, source, pool, k=20, n_comp=40):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.neighbors import NearestNeighbors
        self.rows = np.asarray(pool, dtype=np.int64)
        x = source.X[self.rows].tocsr()
        depth = depth_of(x)
        c = x.astype(np.float64)
        c.data = np.rint(np.expm1(c.data) * np.repeat(depth, np.diff(c.indptr)) / 1e4)
        self.counts = c.tocsr()
        self.depth = np.asarray(self.counts.sum(axis=1)).ravel()
        z = TruncatedSVD(n_components=n_comp, random_state=0).fit_transform(x)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        self.nn = NearestNeighbors(n_neighbors=k).fit(z).kneighbors(z, return_distance=False)
        self.pos = np.full(int(source.X.shape[0]), -1, dtype=np.int64)
        self.pos[self.rows] = np.arange(len(self.rows))

    @classmethod
    def of(cls, source, pool, k=20):
        rows = np.ascontiguousarray(np.asarray(pool, dtype=np.int64))
        key = (id(source), hashlib.sha1(rows.tobytes()).hexdigest(), k)
        if key not in cls._cache:
            cls._cache[key] = cls(source, rows, k)
        return cls._cache[key]

    def shifted(self, rows, log_factor, seed=SEED):
        """log1p(counts per 10k) of the source cells `rows` after gene-wise factors exp(log_factor) (n x genes)."""
        rng = np.random.default_rng(seed + 20011)
        idx = self.pos[np.asarray(rows, dtype=np.int64)]
        assert (idx >= 0).all(), "count-space shift: a cell is not in the pool"
        n, n_pool = len(idx), self.counts.shape[0]
        out = np.empty((n, self.counts.shape[1]), dtype=np.float32)
        for s in range(0, n, 256):
            i = idx[s:s + 256]
            c = self.counts[i].toarray()
            f = np.exp(np.clip(log_factor[s:s + 256], -CLIP, CLIP).astype(np.float64))
            nb = self.nn[i]
            sel = sparse.csr_matrix((np.ones(nb.size), (np.repeat(np.arange(len(i)), nb.shape[1]), nb.ravel())), shape=(len(i), n_pool))
            expected = (sel @ self.counts).toarray() * (self.depth[i] / (sel @ self.depth))[:, None]
            down = rng.binomial(c.astype(np.int64), np.minimum(f, 1.0))
            up = rng.poisson(np.maximum(f - 1.0, 0.0) * 0.5 * (c + expected))
            new = np.where(f < 1.0, down, c + up).astype(np.float64)
            tot = new.sum(axis=1, keepdims=True)
            out[s:s + 256] = np.log1p(new * np.divide(1e4, tot, out=np.ones_like(tot), where=tot > 0))
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

    def pooled_counts(self, k):
        """Raw counts of every anchor nucleus summed with those of its k-1 nearest nuclei (dense float32, cached)."""
        cache = getattr(self, "_pooled", None)
        if cache is None or cache[0] != k:
            counts = self.x.astype(np.float64).tocsr()
            counts.data = np.rint(np.expm1(counts.data) * np.repeat(self.depth, np.diff(counts.indptr)) / 1e4)
            nn = self.neighbours(k)[:, :k]
            n = self.x.shape[0]
            sel = sparse.csr_matrix((np.ones(nn.size, dtype=np.float32), (np.repeat(np.arange(n), k), nn.ravel())), shape=(n, n))
            self._pooled = (k, np.asarray((sel @ counts).todense(), dtype=np.float32))
        return self._pooled[1]

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
        # guarded forward log-ratios (round 8): a lineage or cell type is moved only in genes it expresses itself in the
        # source stage (source mean at least OWN_SHARE of the largest source lineage mean); what the anchor shows of other
        # lineages' markers is ambient RNA or doublets. "dg" = lineage-specific part (d), "dr" = d plus the part shared by
        # all lineages (the whole anchor / source log-ratio).
        best = np.max(np.stack(list(ms.values())), axis=0)
        own = {k: ms[k] >= OWN_SHARE * best for k in ms}
        if level == "type":
            own.update({"type:" + t: tm_s[t] >= OWN_SHARE * best for t in tm_s})
        dg = {k: np.where(own[k], v, 0.0).astype(np.float32) for k, v in d.items() if k in own}
        dr = {k: np.where(own[k], v - c, 0.0).astype(np.float32) for k, v in d.items() if k in own}
        dc = {k: np.where(own[k], -c, 0.0).astype(np.float32) for k in d if k in own}          # the shared part alone
        means = dict(ms)
        if level == "type":
            means.update({"type:" + t: v for t, v in tm_s.items()})
        ctx = {"near": near, "lineage": lin_a, "c": c.astype(np.float32), "d": d, "dg": dg, "dr": dr, "dc": dc, "own": own, "source_means": means,
               "used": used, "rho": rho,
               "rho_genes": rho_genes, "anchor_lineage_share": {k: float((lin_a == k).mean()) for k in np.unique(lin_a)}}
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
            # the same two-witness rule applied to the WHOLE anchor / source ratio of the lineage (round 8): the part of
            # the ratio that all lineages share is no longer taken to be a difference between the laboratories; it has to
            # pass the same test (same direction as the lineage's own change from the previous stage, capped at
            # days_to_anchor days of that change)
            devw = {}
            for k in vel:
                raw = (d[k] - c).astype(np.float64)
                v = vel[k].astype(np.float64)
                same = (np.sign(v) == np.sign(raw)) & (v != 0) & (ms[k] >= OWN_SHARE * best)
                devw[k] = np.where(same, np.sign(raw) * np.minimum(np.abs(raw), days_to_anchor * np.abs(v)), 0.0).astype(np.float32)
            ctx["devw"] = devw
            # persistence-calibrated forward ratio "dm": the whole guarded ratio dr of a lineage, scaled by how much of it the
            # same-laboratory step from the previous stage realised per day. Regression through the origin of vel (one day)
            # on dr over the genes the lineage expresses; m = days_to_anchor x slope is 1 if the ratio were development at
            # a constant rate and 0 if it had nothing to do with the lineage's own change. Types take their lineage's m.
            m = {}
            for k in vel:
                sel = (ms[k] >= 0.2) & own[k]
                x, v = dr[k][sel].astype(np.float64), vel[k][sel].astype(np.float64)
                m[k] = float(np.clip(days_to_anchor * (v * x).sum() / max((x * x).sum(), 1e-12), 0.0, 1.5))
            ctx["m"] = m
            ctx["dm"] = {k: (m.get(lineage_of([k[5:]])[0] if k.startswith("type:") else k, 0.0) * v).astype(np.float32) for k, v in dr.items()}
        self._ctx[key] = ctx
        return ctx


def mixed_cells(source, rows, anchor, ctx, alpha, seed=SEED, k=20, guard=True, return_info=False, low=None, cells_share=1.0):
    """Real source cells in which a share `alpha` of the molecules is replaced by molecules of an anchor nucleus of the
    same cell type (round 8, "molecular mixture").

    Every source cell is paired with a random anchor nucleus assigned to its cell type (to its lineage if the type has
    fewer than MIN_CELLS nuclei; cells of the lineage "Other" and of lineages the anchor lacks are left as they are). The
    cell keeps each of its own UMIs with probability 1 - alpha (binomial thinning) and receives Poisson(alpha x depth x
    profile) UMIs from the partner's profile, i.e. the raw counts of the partner and its k - 1 nearest anchor nuclei,
    normalised to 1. With k = "auto" no profile is used: the cell receives measured molecules only, namely the UMIs of
    the partner and of as many of its nearest nuclei as are needed to supply alpha x depth of them (at most 20), each kept
    with the probability that brings their number down to alpha x depth (binomial thinning). With `guard`, anchor molecules of genes the cell type does not express itself in the source stage
    (below OWN_SHARE of the largest source lineage mean: other lineages' markers, which reach the anchor nuclei as ambient
    RNA) are dropped from the profile before it is normalised. The cell's depth is unchanged on average, its pseudobulk
    contribution is that of a (1 - alpha) : alpha mixture of the two populations, and it remains one cell.

    low (counts per 10k): the mixture is confined to the genes whose mean in the cell's own type (source stage) is below
    `low`. For these genes the cell keeps each of its molecules with probability 1 - alpha and receives anchor molecules
    at the rate alpha x depth x (share of the gene in the partner's whole profile); all other genes keep their counts.
    What the cell type already expresses is left to the same-laboratory shifts; genes it barely expresses at the source
    stage move to the share alpha of the way to the later level.
    Returns log1p(counts per 10k), float32."""
    rng = np.random.default_rng(seed + 30011)
    rows = np.asarray(rows, dtype=np.int64)
    x = source.X[rows].tocsr()
    depth = depth_of(x)
    labels = source.labels[rows]
    near, lin_a, own = ctx["near"], ctx["lineage"], ctx["own"]
    partner = np.full(len(rows), -1, dtype=np.int64)
    masks, lows = {}, {}
    for t in np.unique(labels):
        lin = lineage_of(np.array([t]))[0]
        if lin == "Other":
            continue
        cand = np.flatnonzero(near == t)
        if len(cand) < MIN_CELLS:
            cand = np.flatnonzero(lin_a == lin)
        if len(cand) < MIN_CELLS:
            continue
        own_t = own.get("type:" + t, own.get(lin))
        mean_t = ctx["source_means"].get("type:" + t, ctx["source_means"].get(lin))
        if own_t is None or mean_t is None:      # too few source cells of this type and lineage for a mean: left as they are
            continue
        idx = np.flatnonzero(labels == t)
        partner[idx] = rng.choice(cand, len(idx), replace=True)
        masks[t] = own_t
        if low is not None:
            lows[t] = mean_t < low
    if cells_share < 1.0:      # analysis only: the mixture is confined to a random share of the cells that have a partner
        eligible = np.flatnonzero(partner >= 0)
        drop = np.random.default_rng(seed + 40013).choice(eligible, int(round((1.0 - cells_share) * len(eligible))), replace=False)
        partner[drop] = -1
    real_molecules = isinstance(k, str) and k == "auto"
    if real_molecules:
        # only molecules that were measured: the partner's own UMIs, topped up with those of its nearest nuclei until
        # there are enough (at most 20 nuclei), then thinned binomially to alpha x depth
        counts_a = anchor.x.astype(np.float64).tocsr()
        counts_a.data = np.rint(np.expm1(counts_a.data) * np.repeat(anchor.depth, np.diff(counts_a.indptr)) / 1e4)
        nn = anchor.neighbours(20)[:, :20]
        cum = np.cumsum(anchor.depth[nn], axis=1)                       # depth of the first 1 .. 20 neighbours
    else:
        pooled = anchor.pooled_counts(k)
    out = np.empty((len(rows), x.shape[1]), dtype=np.float32)
    used_k = np.zeros(len(rows), dtype=np.int64)
    for s in range(0, len(rows), 256):
        sl = slice(s, s + 256)
        c = np.rint(np.expm1(x[sl].toarray().astype(np.float64)) * (depth[sl, None] / 1e4)).astype(np.int64)
        p = partner[sl]
        has = p >= 0
        new = c.astype(np.float64)
        if has.any() and alpha > 0:
            need = alpha * depth[sl][has]
            if real_molecules:
                kk = np.minimum((cum[p[has]] < 1.15 * need[:, None]).sum(axis=1) + 1, 20)      # 15% headroom for the guard
                used_k[np.flatnonzero(has) + s] = kk
                r = np.repeat(np.arange(len(kk)), kk)
                cidx = np.concatenate([nn[j, :m] for j, m in zip(p[has], kk)])
                sel = sparse.csr_matrix((np.ones(len(r)), (r, cidx)), shape=(len(kk), counts_a.shape[0]))
                prof = np.asarray((sel @ counts_a).todense())
            else:
                prof = pooled[p[has]].astype(np.float64)
            if guard:
                lab = labels[sl][has]
                for t in np.unique(lab):
                    if masks.get(t) is not None:
                        prof[lab == t] *= masks[t][None, :]
            tot_p = np.maximum(prof.sum(axis=1, keepdims=True), 1.0)
            if low is not None:        # low genes only: the mixture is confined to the genes the cell type has below `low`
                lab = labels[sl][has]
                lowm = np.stack([lows[t] for t in lab])
                prof *= lowm
                keep = np.where(lowm, rng.binomial(c[has], 1.0 - alpha), c[has])
            else:
                keep = rng.binomial(c[has], 1.0 - alpha)
            if real_molecules:
                add = rng.binomial(prof.astype(np.int64), np.minimum(need[:, None] / tot_p, 1.0))
            else:
                add = rng.poisson(need[:, None] * prof / tot_p)
            new[has] = keep + add
        tot = new.sum(axis=1, keepdims=True)
        out[sl] = np.log1p(new * np.divide(1e4, tot, out=np.ones_like(tot), where=tot > 0))
    if return_info:
        return out, {"partner": partner, "mixed_cells": int((partner >= 0).sum()), "nuclei_pooled": used_k}
    return out


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


def _real_standins(source, pool, used, types, rng):
    """One real source cell for every requested cell type: drawn without replacement from the pool cells of that type
    that are not used yet; if a type runs out, from unused cells of the same lineage; only then with replacement."""
    pool = np.asarray(pool)
    free = np.setdiff1d(pool, used)
    labels = source.labels[free]
    lin = lineage_of(labels)
    types = np.asarray(types)
    out = np.empty(len(types), dtype=np.int64)
    taken = np.zeros(len(free), dtype=bool)
    order = rng.permutation(len(types))
    pending = []
    for t in np.unique(types):
        want = order[types[order] == t]
        have = rng.permutation(np.flatnonzero((labels == t) & ~taken))
        m = min(len(want), len(have))
        out[want[:m]] = free[have[:m]]
        taken[have[:m]] = True
        pending += list(want[m:])
    for i in pending:
        lineage = lineage_of(np.array([types[i]]))[0]
        have = np.flatnonzero((lin == lineage) & ~taken)
        if len(have):
            j = rng.choice(have)
            out[i] = free[j]
            taken[j] = True
        else:
            cand = pool[source.labels[pool] == types[i]]
            cand = cand if len(cand) else pool[lineage_of(source.labels[pool]) == lineage]
            out[i] = rng.choice(cand if len(cand) else pool)
    return out


def predict7(source, pool, anchor, n_total, f, *, fwd=0.0, bwd=0.0, harmonise=1.0, mode="min", level="lineage",
             pool_k=1, anchor_rows=None, shares=None, seed=SEED, context_pool=None, return_info=False,
             prev=None, days_to_anchor=4.0, dev=0.0, k_prof=10, trend=None, trend_share=0.0, anchor_shift=None, tilt=0.0,
             depth_from="source", real_share=0.0, fwd_kind="d", shift_mode="scale", dev_kind="dev", mix=0.0, mix_guard=True, mix_k=20, mix_low=None, mix_cells=1.0):
    """Cells of the prediction (float32, n_total x genes): shifted base cells first, shifted anchor nuclei last.

    mix           share of the molecules of every real cell (base and stand-ins) replaced by molecules of an anchor nucleus of
                  the same cell type before the other shifts are applied (see mixed_cells); mix_guard, mix_k as there
    dev_kind      "dev" (two-witness shift on the lineage-specific part of the ratio, round 7) or "devw" (on the whole ratio)
    shift_mode    "scale" (round 7: the counts a cell has are multiplied, zeros stay zero) or "counts" (round 8: the shifts
                  of a real cell are applied to its UMI counts, see SourceCounts; synthetic nuclei get all shifts before
                  they are drawn)
    fwd_kind      which log-ratio `fwd` moves the base along: "d" (lineage-specific part, round 7 experiments), "dg" (the
                  same, only in genes the lineage or type expresses itself), "dr" (whole anchor / source ratio, same guard).
                  With "dg" / "dr", nuclei that were moved back completely (bwd = 1) move forward with the base.

    pool          source rows the base is drawn from (uniformly, or to lineage `shares`)
    context_pool  source rows the lineage statistics are estimated on (default: `pool`)
    dev, trend_share  developmental shifts of the BASE cells (two-witness shift, within-cell bulk trend)
    tilt          strength of the individual low-rank deviation restored in re-sequenced nuclei (see Anchor7.resampled)
    depth_from    "source" (default: depth of a source cell of the same lineage) or "anchor" (the nucleus's own depth; analysis only)
    real_share    share of the picked anchor nuclei that are NOT synthesised but replaced by a real source cell of the cell
                  type the nucleus is assigned to (see _real_standins). These cells are part of the base from then on: they
                  receive the developmental shifts like every base cell. 0 = round-7 file, 1 = no synthetic nucleus at all;
                  the cell-type composition of the result is the same for every value.
    anchor_shift  share of those two shifts that the anchor nuclei receive as well. Default: 1 if the nuclei were
                  moved back completely (bwd = 1: they sit at the source's means, like the base, and move with it),
                  0 otherwise (nuclei that keep E14.5 expression are not pushed further forward).
    """
    rng = np.random.default_rng(seed)
    k = int(round(n_total * f))
    rows = draw_base(source, pool, n_total - k, rng, shares)
    need_ctx = bool(k and (harmonise or bwd or pool_k == "resample" or real_share)) or bool(fwd) or bool(dev) or bool(mix)
    ckw = dict(prev=prev, days_to_anchor=days_to_anchor) if (dev or fwd_kind == "dm") else {}
    ctx = anchor.context(source, pool if context_pool is None else context_pool, mode, level, **ckw) if need_ctx else None
    picked_given, stand_for, n_base = None, np.array([], dtype=int), len(rows)
    if k and real_share:
        # the nuclei are picked exactly as in the synthetic case (the generator is in the same state: nothing else
        # draws from it between the base and the nuclei); a share of them is then represented by real source cells
        candidates = np.arange(anchor.x.shape[0]) if anchor_rows is None else np.asarray(anchor_rows)
        picked_all = np.sort(candidates[rng.choice(len(candidates), k, replace=False)])
        r2 = np.random.default_rng(seed + 10007)
        take = np.sort(r2.choice(k, int(round(real_share * k)), replace=False))
        stand_for = picked_all[take]
        rows = np.concatenate([rows, _real_standins(source, pool, rows, ctx["near"][stand_for], r2)])
        picked_given = picked_all[np.setdiff1d(np.arange(k), take)]
        k = len(picked_given)
    n_genes = source.X.shape[1]
    counts_mode = shift_mode == "counts"
    a_shift = (1.0 if bwd >= 1.0 else 0.0) if anchor_shift is None else float(anchor_shift)
    if counts_mode and (fwd or dev or trend_share):
        labels = source.labels[rows]
        lf = np.zeros((len(rows), n_genes), dtype=np.float32)
        if fwd:
            lf += _factors(labels, lineage_of(labels), ctx[fwd_kind], fwd, n_genes)
        if dev:
            lf += _factors(labels, lineage_of(labels), ctx[dev_kind], dev, n_genes)
        if trend_share:
            lf += (np.asarray(trend, dtype=np.float32) * np.float32(trend_share))[None, :]
        cells = SourceCounts.of(source, pool).shifted(rows, lf, seed)
    else:
        assert not (mix and counts_mode), "the molecular mixture is implemented for shift_mode = 'scale'"
        cells = mixed_cells(source, rows, anchor, ctx, mix, seed, k=mix_k, guard=mix_guard, low=mix_low, cells_share=mix_cells) if mix else source.dense(rows)
        if fwd:
            labels = source.labels[rows]
            cells = scale_cells(cells, _factors(labels, lineage_of(labels), ctx[fwd_kind], fwd, n_genes))
        if dev:   # two-witness developmental shift: `dev` = share of the source -> anchor path (e.g. 0.5 for E12.5)
            labels = source.labels[rows]
            cells = scale_cells(cells, _factors(labels, lineage_of(labels), ctx[dev_kind], dev, n_genes))
        if trend_share:
            # within-cell bulk trend (same-laboratory E9.5 -> E14.5 whole-heart log-ratio, round 2): every count is scaled
            # by exp(trend x trend_share), trend_share = share of the five-day bulk interval; row-wise, zeros stay zero
            from experiments.bulk_trend import apply_trend
            cells = apply_trend(cells, trend, trend_share, False, seed)
    picked = np.array([], dtype=int)
    if k:
        if picked_given is None:
            candidates = np.arange(anchor.x.shape[0]) if anchor_rows is None else np.asarray(anchor_rows)
            picked = np.sort(candidates[rng.choice(len(candidates), k, replace=False)])
        else:
            picked = picked_given
        cpool = pool if context_pool is None else context_pool
        if pool_k == "resample":
            ctx = anchor.context(source, cpool, mode, level, **ckw)
            lf = np.tile(harmonise * ctx["c"], (k, 1)) if harmonise else np.zeros((k, n_genes), dtype=np.float32)
            if bwd:
                lf = lf + _factors(ctx["near"][picked], ctx["lineage"][picked], ctx["d"], -bwd, n_genes)
            if dev and a_shift:
                lf = lf + _factors(ctx["near"][picked], ctx["lineage"][picked], ctx[dev_kind], dev * a_shift, n_genes)
            if fwd and a_shift and fwd_kind != "d":
                lf = lf + _factors(ctx["near"][picked], ctx["lineage"][picked], ctx[fwd_kind], fwd * a_shift, n_genes)
            if counts_mode and trend_share and a_shift:      # the bulk trend enters before the nucleus is drawn
                lf = lf + (np.asarray(trend, dtype=np.float32) * np.float32(trend_share * a_shift))[None, :]
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
                    lf = lf + _factors(ctx["near"][picked], ctx["lineage"][picked], ctx[dev_kind], dev * a_shift, n_genes)
                nuclei = scale_cells(nuclei, lf)
        if trend_share and a_shift and not (counts_mode and pool_k == "resample"):
            from experiments.bulk_trend import apply_trend
            nuclei = apply_trend(nuclei, trend, trend_share * a_shift, False, seed)
        cells = np.vstack([cells, nuclei])
    if return_info:
        return cells, {"base_rows": rows, "anchor_rows": picked, "f": float(f), "n_drawn_base": n_base, "stand_in_for": stand_for}
    return cells
