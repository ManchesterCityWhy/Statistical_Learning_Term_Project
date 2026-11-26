"""Checks of the thinned export (afternoon of 11-26), in a fresh process, read-only.

    python -m experiments.round8_verify9 <prediction dir name> <thin> <mix> <round-8 dir name>
"""
import hashlib
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd

from experiments.bulk_trend import load_trend
from experiments.round5_anchor import load_anchor
from experiments.round6_common import REAL, Mem, depth_of, lineage_of
from experiments.round7_export import build, thinned_source
from t1_pipeline.data import board_spec, file_hash, validate_prediction

name, thin, mix, old = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
d = REAL / "predictions" / name
man = json.load(open(d / "manifest.json"))
path = d / "E12.5_prediction.h5ad"
out = {}
out["sha256_matches_manifest"] = file_hash(path) == man["file_sha256"]
out["sha256"] = file_hash(path)
out["bytes"] = path.stat().st_size
out["bytes_match_manifest"] = path.stat().st_size == man["file_bytes"]
spec, genes = board_spec("T1:test")
a = ad.read_h5ad(path)
x = np.asarray(a.X.todense() if hasattr(a.X, "todense") else a.X, dtype=np.float32)
out["shape"] = list(x.shape)
out["gene_order_official"] = list(a.var_names.astype(str)) == list(genes)
out["finite_nonnegative"] = bool(np.isfinite(x).all() and (x >= 0).all())
lib = np.expm1(x.astype(np.float64)).sum(axis=1)
out["row_sum_min_max"] = [float(lib.min()), float(lib.max())]
out["distinct_rows"] = int(len(np.unique(x, axis=0)))
out["max_value"] = float(x.max())
out["validation"] = validate_prediction(path, "T1:test")
out["cells_sha256_matches_manifest"] = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest() == man["prediction_cells_sha256"]
prov = np.load(d / "provenance.npz")
rows, partner = prov["real_rows"], prov["mixture_partner"]
source = Mem(REAL / "data/T1/E10.5_RNA.h5ad", genes)
out["all_rows_from_library_3"] = bool((source.lib[rows] == "3").all())
out["distinct_source_rows"] = int(len(np.unique(rows)))
lin = lineage_of(source.labels[rows])
out["lineage_shares"] = pd.Series(lin).value_counts(normalize=True).round(4).to_dict()
out["unmixed_cells"] = int((partner < 0).sum())
# fresh rebuild
src_t, info = thinned_source(source, thin)
out["thinning_info_matches_manifest"] = info == man.get("thinning")
out["thinning_info"] = info
e9 = Mem(REAL / "data/T1/E9.5_RNA.h5ad", genes)
ax, _ = load_anchor(genes, verbose=False)
trend, _ = load_trend(genes)
again = build(src_t, ax, 0.45, 0.0, 5000, level="type", dev=0.5, prev=(e9, np.arange(e9.n_cells)), days=4.0, trend=trend, trend_share=0.4, k_prof=20,
              real_share=1.0, mix=mix, mix_low=0.3, mix_k=20)
out["fresh_rebuild_identical"] = bool(np.array_equal(again, x))
# thinning visible in the file: unmixed cells keep exactly the non-zero pattern of their thinned source cell, which is a
# subset of the pattern of the original cell
un = np.flatnonzero(partner < 0)
t_nz = (src_t.X[rows[un]] > 0).toarray()
o_nz = (source.X[rows[un]] > 0).toarray()
f_nz = x[un] > 0
out["unmixed_pattern_equals_thinned_source"] = bool((f_nz == t_nz).all())
out["unmixed_pattern_subset_of_original"] = bool((~f_nz | o_nz).all())
out["unmixed_genes_detected_ratio_file_over_original"] = float(f_nz.sum() / o_nz.sum())
d0, d1 = depth_of(source.X[rows]), depth_of(src_t.X[rows])
out["median_umi_original_rows"] = float(np.median(d0))
out["median_umi_thinned_rows"] = float(np.median(d1))
out["umi_ratio_quantiles_5_50_95"] = [float(v) for v in np.quantile(d1 / d0, [.05, .5, .95])]
out["median_genes_detected_file"] = float(np.median((x > 0).sum(axis=1)))
# against the round-8 file
man8 = json.load(open(REAL / "predictions" / old / "manifest.json"))
prov8 = np.load(REAL / "predictions" / old / "provenance.npz")
out["same_base_rows_as_round8"] = bool(np.array_equal(prov8["real_rows"][:2750], rows[:2750]))
out["stand_in_rows_shared_with_round8"] = int(len(np.intersect1d(prov8["real_rows"][2750:], rows[2750:])))
out["rows_shared_with_round8_total"] = int(len(np.intersect1d(prov8["real_rows"], rows)))
x8 = ad.read_h5ad(REAL / "predictions" / old / "E12.5_prediction.h5ad")
x8 = np.asarray(x8.X.todense() if hasattr(x8.X, "todense") else x8.X, dtype=np.float32)
out["median_genes_detected_round8"] = float(np.median((x8 > 0).sum(axis=1)))
out["rows_identical_to_round8_file"] = int((x == x8).all(axis=1).sum()) if x.shape == x8.shape else None
out["manifest_fields_differing_from_round8"] = sorted(k for k in set(man) | set(man8) if json.dumps(man.get(k), sort_keys=True) != json.dumps(man8.get(k), sort_keys=True))
out["code_hashes_differing_from_round8"] = sorted(k for k in man["code_sha256"] if man["code_sha256"][k] != man8["code_sha256"].get(k))
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
json.dump(out, open(REAL / "logs" / f"r9_verify_{name[-24:]}.json", "w"), indent=1, default=str)
