"""Synthetic checks only: none of these numbers are biological experiment results."""
import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from t1_pipeline import cli
from t1_pipeline.data import Stage
from t1_pipeline.evaluation import calibrated_score, metric_panel
from t1_pipeline.model import Extrapolator, extrapolate


def test_public_gene_panel_hash_ignores_final_newline_but_checks_order(tmp_path, monkeypatch):
    import t1_pipeline.data as module

    genes = ["Xkr4", "Gm1992", "Gm19938"]
    panel = tmp_path / "genes.txt"
    panel.write_text("\n".join(genes) + "\n")
    (tmp_path / "boards.json").write_text(json.dumps({"boards": {"T1:test": {
        "genes_file": "genes.txt", "genes_sha256": module.panel_hash(genes)[:16]}}}))
    monkeypatch.setattr(module, "SPEC_DIR", tmp_path)
    assert module.board_spec("T1:test")[1] == genes
    panel.write_text("\n".join(reversed(genes)) + "\n")
    with pytest.raises(ValueError, match="changed"):
        module.board_spec("T1:test")


@pytest.fixture
def stages(tmp_path):
    genes = ["a", "b", "c", "d"]
    a = np.array([[1, 0, 1, 0], [2, 0, 1, 0], [1, 0, 2, 0], [2, 0, 2, 0]], dtype=np.float32)
    b = np.array([[2, 0, 1, 0], [3, 0, 1, 0], [0, 2, 0, 1], [0, 3, 0, 1]], dtype=np.float32)
    result = []
    for i, matrix in enumerate((a, b)):
        # Source gene order differs deliberately; Stage must align columns correctly.
        order = np.array([2, 0, 3, 1])
        data = ad.AnnData(X=sparse.csr_matrix(matrix[:, order]),
                          obs=pd.DataFrame({"celltype": ["A", "A", "A", "A"] if i == 0 else ["A", "A", "new", "new"]}, index=[f"c{j}" for j in range(4)]),
                          var=pd.DataFrame(index=np.array(genes)[order]))
        path = tmp_path / f"stage{i}.h5ad"
        data.write_h5ad(path)
        result.append(Stage.open(path, genes))
    yield result, a, b
    for stage in result:
        stage.close()


def test_backed_read_keeps_requested_row_order_and_aligns_genes(stages):
    (a, _), expected, _ = stages
    np.testing.assert_array_equal(a.dense([3, 0, 2]), expected[[3, 0, 2]])
    np.testing.assert_allclose(a.mean(batch_size=2), expected.mean(axis=0))


def test_alpha_zero_matches_copy_and_pca_preserves_cell_residuals(stages):
    (a, b), _, _ = stages
    model = Extrapolator(a, b, pca_cells_per_stage=4)
    model.fit_pca([1, 2])
    copy, _ = model.predict({"method": "copy_last"}, 4, seed=42, h=1)
    zero, _ = model.predict({"method": "pca_shift", "dimensions": 2, "alpha": 0}, 4, seed=42, h=2)
    np.testing.assert_array_equal(copy, zero)
    assert 0 <= model.diagnostics["1"]["retained_delta_squared_norm"] <= 1 + 1e-6


def test_type_baseline_does_not_move_a_newly_observed_type(stages):
    (a, b), _, _ = stages
    model = Extrapolator(a, b)
    copy, _ = model.predict({"method": "copy_last"}, 4, 42, h=2)
    shifted, _ = model.predict({"method": "pseudobulk_shift", "alpha": 1}, 4, 42, h=2)
    novel = copy[:, 1] > 0
    np.testing.assert_array_equal(copy[novel], shifted[novel])
    np.testing.assert_array_equal(shifted[~novel, 0], copy[~novel, 0] + 2)


def test_zero_preservation_and_two_day_multiplier():
    x = np.array([[0, 2], [1, 0]], dtype=np.float32)
    y, _ = extrapolate(x, [.5, .5], alpha=.5, h=2, preserve_zeros=True)
    np.testing.assert_array_equal(y, [[0, 2.5], [1.5, 0]])
    z, clipped = extrapolate(x, [-2, -2], alpha=1, h=1)
    assert clipped == .75
    assert np.all(z >= 0)


def test_real_veckit_metrics_and_calibration_on_synthetic_distributions():
    rng = np.random.default_rng(42)
    reference = rng.gamma(2, 1, (64, 64)).astype(np.float32)
    target = np.maximum(reference + np.linspace(-1, 2, 64), 0).astype(np.float32)
    floor = metric_panel(reference, target, reference, seed=42)
    perfect = metric_panel(target, target, reference, seed=42)
    assert all(value is not None for value in floor.values())
    assert perfect["de_score"] > floor["de_score"]
    assert perfect["de_direction"] > floor["de_direction"]
    assert perfect["variogram"] == pytest.approx(0, abs=1e-10)
    result = calibrated_score(floor, floor, perfect)
    assert result["local_score"] == pytest.approx(50)
    assert calibrated_score(perfect, floor, perfect)["local_score"] == pytest.approx(100)
    missing = dict(perfect, de_score=None)
    assert calibrated_score(missing, floor, perfect)["local_score"] == pytest.approx(75)


def test_rejects_missing_genes_and_negative_input(tmp_path):
    path = tmp_path / "bad.h5ad"
    ad.AnnData(X=np.array([[0, -1], [1, 1]], dtype=np.float32), var=pd.DataFrame(index=["a", "b"])).write_h5ad(path)
    with pytest.raises(ValueError, match="genes"):
        Stage.open(path, ["a", "c"])
    stage = Stage.open(path, ["a", "b"])
    try:
        with pytest.raises(ValueError, match="negative"):
            stage.mean()
    finally:
        stage.close()


def test_prediction_round_trip_has_no_submitted_labels(tmp_path):
    path = tmp_path / "pred.h5ad"
    cli.make_prediction_file(path, np.ones((3, 4), dtype=np.float32), ["a", "b", "c", "d"])
    loaded = ad.read_h5ad(path)
    assert loaded.X.dtype == np.float32
    assert "celltype" not in loaded.obs
    assert list(loaded.var_names) == ["a", "b", "c", "d"]
    with pytest.raises(FileExistsError):
        cli.make_prediction_file(path, loaded.X, list(loaded.var_names))


def test_final_prediction_uses_only_e95_e105_and_h2(stages, tmp_path, monkeypatch):
    (a, b), _, _ = stages
    genes = ["a", "b", "c", "d"]
    opened = []
    frozen = {"candidate": {"name": "copy", "method": "copy_last"}, "data": {},
              "pca_cells_per_stage": 4, "pca_seed": 42, "prediction_cells": 3,
              "prediction_seed": 42, "input_sha256": {"E9.5": "a", "E10.5": "b"},
              "development_code_sha256": {}, "veckit_commit": cli.VECKIT_COMMIT}
    monkeypatch.setattr(cli, "load_config", lambda _: frozen)
    monkeypatch.setattr(cli, "board_spec", lambda _: ({"min_cells": 2, "max_cells": 4}, genes))
    monkeypatch.setattr(cli, "run_manifest", lambda _: {"code_sha256": {}})
    monkeypatch.setattr(cli, "file_hash", lambda p: "a" if p == a.path else "b")
    def open_fake(stack, config, names, panel):
        opened.extend(names)
        return {"E9.5": a, "E10.5": b}
    monkeypatch.setattr(cli, "open_stages", open_fake)
    monkeypatch.setattr(cli, "validate_prediction", lambda *args: {"format_valid": True})
    out = tmp_path / "final"
    cli.predict_final(SimpleNamespace(selection="unused", output=str(out)))
    manifest = json.loads((out / "manifest.json").read_text())
    assert opened == ["E9.5", "E10.5"]
    assert manifest["h"] == 2
    assert manifest["status"] == "prediction_ready_not_submitted"


def test_synthetic_benchmark_freeze_and_export_end_to_end(tmp_path, monkeypatch):
    import t1_pipeline.data as data_module

    rng = np.random.default_rng(19)
    genes = [f"gene{i}" for i in range(64)]
    delta = np.linspace(-.8, 1.2, 64).astype(np.float32)
    base = rng.gamma(2, .5, (128, 64)).astype(np.float32)
    paths = {}
    for step, name in enumerate(["E8.5", "E9.5", "E10.5"]):
        matrix = np.maximum(base + step * delta, 0)
        path = tmp_path / f"{name}.h5ad"
        ad.AnnData(X=sparse.csr_matrix(matrix), var=pd.DataFrame(index=genes)).write_h5ad(path)
        paths[name] = str(path)
    config = {"data": paths, "prediction_cells": 64, "pca_cells_per_stage": 64,
              "pca_seed": 42, "evaluation_seed": 42, "evaluation_stage_fraction": 1,
              "seeds": [42], "candidates": [{"name": "copy", "method": "copy_last"},
                   {"name": "pca", "method": "pca_shift", "dimensions": 3, "alpha": .5}]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    schema = lambda _: ({"min_cells": 16, "max_cells": 128}, genes)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "board_spec", schema)
    monkeypatch.setattr(data_module, "board_spec", schema)
    # Only this synthetic test bypasses the real release checksum requirement.
    monkeypatch.setattr(cli, "check_hash", lambda *args: None)
    run = tmp_path / "run"
    cli.benchmark(SimpleNamespace(config=str(config_path), output=str(run), keep_predictions=False))
    assert json.loads((run / "manifest.json").read_text())["status"] == "complete"
    assert len(pd.read_csv(run / "metrics.csv")) == 2
    selection = tmp_path / "frozen.json"
    cli.freeze(SimpleNamespace(run=str(run), candidate="pca", seed=42, output=str(selection)))
    out = tmp_path / "export"
    cli.predict_final(SimpleNamespace(selection=str(selection), output=str(out)))
    final = ad.read_h5ad(out / "E12.5_prediction.h5ad")
    assert final.shape == (64, 64)
    assert np.isfinite(final.X).all() and np.all(final.X >= 0)
    assert "celltype" not in final.obs
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["h"] == 2 and manifest["file_validation"]["format_valid"]
