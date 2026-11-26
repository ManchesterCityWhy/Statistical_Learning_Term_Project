"""Round 7: rebuild the first official submission (h1 = experiments/export_composition.py, h = 1) on this machine.

Same statements as export_composition.py for h = 1 (same seed, same order of random draws); additionally keeps
the sampled E10.5 rows. Output: runs/round7_h1/h1.npz (rows, cells) and a regenerated h5ad whose SHA-256 is
compared with the manifest of the file the team submitted.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import nnls
from t1_pipeline.data import Stage, board_spec, file_hash, validate_prediction
from t1_pipeline.cli import make_prediction_file
from experiments.bulk_trend import load_trend, apply_trend

SUBMITTED_SHA = "d6f5ac379e31919494367552a40b376ff2f735d80135ea2a7a22dfec0aa34ed1"
rng = np.random.default_rng(42)
cfg = json.loads(Path("experiments/round2_seed42/final_audit_seed42.json").read_text())["configuration"] if Path("experiments/round2_seed42/final_audit_seed42.json").exists() else {"strength": 3, "bulk": 3}
_, genes = board_spec("T1:test")
b = Stage.open(Path("data/T1/E10.5_RNA.h5ad"), genes)
trend, ext = load_trend(genes)
labels = b.labels
types = np.unique(labels)
means, p = [], []
for lab in types:
    ids = np.flatnonzero(labels == lab)
    p.append(len(ids) / b.n_cells)
    total = np.zeros(len(genes))
    for chunk in np.array_split(ids, max(1, len(ids) // 128)):
        total += np.expm1(b.dense(chunk)).sum(axis=0, dtype=np.float64)
    means.append(total / len(ids))
profiles = np.stack(means, axis=1)
p = np.array(p)
base = profiles @ p
keep = (base > .1) & (np.std(profiles, axis=1) > .1)
design = profiles[keep] / np.sqrt(base[keep, None] + 1)
h = 1
desired = base * np.exp(np.clip(trend * cfg["strength"] * h, -2, 2))
desired *= base.sum() / desired.sum()
q = nnls(np.vstack([design, 10 * np.eye(len(p))]), np.r_[desired[keep] / np.sqrt(base[keep] + 1), 10 * p])[0]
q /= q.sum()
weights = np.zeros(b.n_cells)
for lab, prob in zip(types, q):
    weights[labels == lab] = prob / (labels == lab).sum()
rows = rng.choice(b.n_cells, 2000, replace=False, p=weights / weights.sum())
pred = apply_trend(b.dense(rows), trend, cfg["bulk"] * h / 5, False, 42)
out = Path("runs/round7_h1")
out.mkdir(parents=True, exist_ok=True)
path = out / "E12.5_prediction_h1_rebuilt.h5ad"
if path.exists():
    path.unlink()
make_prediction_file(path, pred, genes)
sha = file_hash(path)
np.savez_compressed(out / "h1.npz", rows=rows, cells=pred.astype(np.float32))
props = {t: round(float(v), 4) for t, v in zip(types, q) if v > 0}
lib = np.array([n.rsplit("_", 1)[-1] for n in __import__("anndata").read_h5ad("data/T1/E10.5_RNA.h5ad", backed="r").obs_names])
info = {"sha256_rebuilt": sha, "sha256_submitted": SUBMITTED_SHA, "identical": sha == SUBMITTED_SHA, "bytes": path.stat().st_size,
        "proportions": props, "library_share": {k: round(float((lib[rows] == k).mean()), 4) for k in "123"},
        "validation": validate_prediction(path, "T1:test")}
(out / "info.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
