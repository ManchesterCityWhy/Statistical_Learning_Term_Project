"""Public specification snapshots and bounded-memory AnnData reads."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import anndata as ad
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "specs" / "t1"
PANEL_URL = "https://virtualembryo.ai/challenge/panels/"
TRAIN_SHA256 = {
    "E8.5": "8eab2d0ccaa89f92861b09816b76b6a731b8ed6b5995e4af196d9ed8ff76e504",
    "E9.5": "0296e0043842f944a663f3c11ac1542d1bbee733c1cd657bd56f75af65958361",
}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_hash(genes: list[str]) -> str:
    # Official genes_sha256 hashes ordered names joined by LF, without a final LF.
    return hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sync_specs() -> dict:
    def download(name):
        req = Request(PANEL_URL + name, headers={"User-Agent": "T1-course-project/1.0"})
        with urlopen(req, timeout=30) as response:
            return response.read()

    index = json.loads(download("index.json"))
    boards = {key: index[key] for key in ("T1:val", "T1:test")}
    panels = {}
    # Validate everything before replacing the last known specification.
    for board in boards.values():
        name = board["genes_file"]
        blob = download(name)
        genes = blob.decode("utf-8").splitlines()
        digest = panel_hash(genes)
        if len(genes) != board["n_genes"] or len(set(genes)) != len(genes):
            raise ValueError(f"Invalid official gene panel: {name}")
        if not digest.startswith(board["genes_sha256"]):
            raise ValueError(f"Official panel checksum mismatch: {name}")
        panels[name] = blob
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    for name, blob in panels.items():
        (SPEC_DIR / name).write_bytes(blob)
    result = {"source": PANEL_URL + "index.json", "retrieved_at": datetime.now(timezone.utc).isoformat(), "boards": boards}
    write_json(SPEC_DIR / "boards.json", result)
    return result


def board_spec(key: str) -> tuple[dict, list[str]]:
    path = SPEC_DIR / "boards.json"
    if not path.exists():
        raise FileNotFoundError("Run: python -m t1_pipeline sync-specs")
    board = json.loads(path.read_text(encoding="utf-8"))["boards"][key]
    panel = SPEC_DIR / board["genes_file"]
    genes = panel.read_text(encoding="utf-8").splitlines()
    if not panel_hash(genes).startswith(board["genes_sha256"]):
        raise ValueError("The local gene panel changed; run sync-specs again")
    return board, genes


@dataclass
class Stage:
    path: Path
    data: ad.AnnData
    order: np.ndarray

    @classmethod
    def open(cls, path: Path, genes: list[str]):
        data = ad.read_h5ad(path, backed="r")
        actual = data.var_names.astype(str)
        if not actual.is_unique or len(actual) != len(genes) or set(actual) != set(genes):
            data.file.close()
            raise ValueError(f"{path.name}: missing, extra, or duplicate genes")
        if data.n_obs < 2:
            data.file.close()
            raise ValueError(f"{path.name}: fewer than two cells")
        return cls(path, data, actual.get_indexer(genes))

    @property
    def n_cells(self):
        return self.data.n_obs

    @property
    def labels(self):
        if "celltype" not in self.data.obs:
            return None
        labels = self.data.obs["celltype"]
        if labels.isna().any():
            raise ValueError(f"{self.path.name}: missing celltype annotations")
        return labels.astype(str).to_numpy()

    def matrix(self, indices):
        indices = np.asarray(indices, dtype=int)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Stage reads must use unique row indices")
        sort = np.argsort(indices)
        matrix = self.data[indices[sort], :].X
        matrix = matrix[:, self.order]
        inverse = np.argsort(sort)
        return matrix[inverse]

    def dense(self, indices):
        matrix = self.matrix(indices)
        return np.asarray(matrix.toarray() if sparse.issparse(matrix) else matrix, dtype=np.float32)

    def mean(self, indices=None, batch_size=256):
        indices = np.arange(self.n_cells) if indices is None else np.asarray(indices)
        if not len(indices):
            raise ValueError("Cannot average an empty cell population")
        total = np.zeros(len(self.order), dtype=np.float64)
        for start in range(0, len(indices), batch_size):
            matrix = self.matrix(indices[start:start + batch_size])
            values = matrix.data if sparse.issparse(matrix) else matrix
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{self.path.name}: .X contains non-finite or negative values")
            total += np.asarray(matrix.sum(axis=0, dtype=np.float64)).ravel()
        return total / len(indices)

    def inspect(self):
        detected = []
        minimum, maximum = float("inf"), 0.0
        for start in range(0, self.n_cells, 256):
            matrix = self.matrix(np.arange(start, min(start + 256, self.n_cells)))
            values = matrix.data if sparse.issparse(matrix) else matrix
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{self.path.name}: .X contains non-finite or negative values")
            minimum = min(minimum, float(values.min(initial=0)))
            maximum = max(maximum, float(values.max(initial=0)))
            detected.extend(np.asarray((matrix > 0).sum(axis=1)).ravel().tolist())
        labels = self.labels
        composition = {} if labels is None else {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        return {"file": self.path.name, "sha256": file_hash(self.path), "cells": self.n_cells,
                "genes": len(self.order), "minimum": minimum, "maximum": maximum,
                "nonzero_fraction": float(np.mean(detected) / len(self.order)),
                "detected_genes_quantiles": np.quantile(detected, [0, .25, .5, .75, 1]).tolist(),
                "celltype_counts": composition,
                "normalization": "Expected released log1p values; no transformation applied. File validation cannot prove normalization."}

    def close(self):
        self.data.file.close()


def validate_prediction(path: Path, board: str):
    spec, genes = board_spec(board)
    data = ad.read_h5ad(path, backed="r")
    try:
        if list(data.var_names.astype(str)) != genes:
            raise ValueError("Prediction genes or order do not match the official board")
        if not spec["min_cells"] <= data.n_obs <= spec["max_cells"]:
            raise ValueError(f"Cell count must be in [{spec['min_cells']}, {spec['max_cells']}]")
        if path.stat().st_size > 1_200_000_000:
            raise ValueError("Prediction exceeds the 1200 MB upload limit")
        for start in range(0, data.n_obs, 256):
            x = data[start:start + 256].X
            values = x.data if sparse.issparse(x) else x
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Prediction contains non-finite or negative expression")
        return {"board": board, "cells": data.n_obs, "genes": data.n_vars,
                "sha256": file_hash(path), "bytes": path.stat().st_size,
                "format_valid": True, "normalization_verified": False}
    finally:
        data.file.close()
