"""Model registry: candidates, promotion archive and rollback."""

import json

import pytest

from src import model_registry


def _install_tmp_layout(tmp_path, monkeypatch):
    root = tmp_path / "models"
    monkeypatch.setattr(model_registry, "MODELS_ROOT", root)
    monkeypatch.setattr(model_registry, "PRODUCTION_DIR", root / "production")
    monkeypatch.setattr(model_registry, "CANDIDATES_DIR", root / "candidates")
    monkeypatch.setattr(model_registry, "ARCHIVE_DIR", root / "archive")
    monkeypatch.setattr(model_registry, "THRESHOLDS_PATH", root / "thresholds.json")
    model_registry.ensure_layout()
    return root


def _write_candidate(root, version, payload, thresholds=None):
    target = root / "candidates" / version
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.pkl").write_bytes(payload)
    (target / "metadata.json").write_text(
        json.dumps({"model_version": version}), encoding="utf-8"
    )
    (target / "metrics.json").write_text(
        json.dumps({"f1": 0.9, "samples": 5}), encoding="utf-8"
    )
    if thresholds is not None:
        (target / "thresholds.json").write_text(
            json.dumps(thresholds), encoding="utf-8"
        )
    return target


def test_candidate_never_touches_production_paths(tmp_path, monkeypatch):
    root = _install_tmp_layout(tmp_path, monkeypatch)

    model_registry.save_candidate(
        {"weights": [1, 2]},
        {"model_version": "c1"},
        "c1",
        metrics={"f1": 0.5},
        dump=lambda _model, path: path.write_bytes(b"candidate-bytes"),
    )

    assert not (root / "production" / "model.pkl").exists()
    assert not (root / "model.pkl").exists()
    assert model_registry.list_candidates() == ["c1"]
    assert model_registry.get_candidate("c1")["metrics"] == {"f1": 0.5}


def test_duplicate_candidate_version_is_rejected(tmp_path, monkeypatch):
    _install_tmp_layout(tmp_path, monkeypatch)
    writer = lambda _model, path: path.write_bytes(b"x")

    model_registry.save_candidate({}, {"model_version": "c1"}, "c1", dump=writer)

    with pytest.raises(model_registry.ModelRegistryError, match="already exists"):
        model_registry.save_candidate({}, {"model_version": "c1"}, "c1", dump=writer)


def test_promotion_archives_previous_production_and_installs_thresholds(
    tmp_path, monkeypatch
):
    root = _install_tmp_layout(tmp_path, monkeypatch)
    _write_candidate(root, "c1", b"one", {"low": 0.3, "high": 0.7})
    _write_candidate(root, "c2", b"two", {"low": 0.4, "high": 0.8})

    first = model_registry.install_production("c1")
    assert (root / "production" / "model.pkl").read_bytes() == b"one"
    assert (root / "model.pkl").read_bytes() == b"one"
    assert json.loads((root / "thresholds.json").read_text(encoding="utf-8"))["low"] == 0.3

    second = model_registry.install_production("c2", gates={"passed": True})
    assert (root / "production" / "model.pkl").read_bytes() == b"two"
    assert json.loads((root / "thresholds.json").read_text(encoding="utf-8"))["low"] == 0.4
    assert first["previous_archive"] is None
    assert second["previous_archive"] is not None
    assert second["previous_archive"].endswith("__c1")
    assert model_registry.latest_archive().endswith("__c1")
    assert second["gates"] == {"passed": True}


def test_rollback_restores_the_previous_production_model(tmp_path, monkeypatch):
    root = _install_tmp_layout(tmp_path, monkeypatch)
    _write_candidate(root, "c1", b"one", {"low": 0.3, "high": 0.7})
    _write_candidate(root, "c2", b"two", {"low": 0.4, "high": 0.8})

    model_registry.install_production("c1")
    model_registry.install_production("c2")
    record = model_registry.rollback(reason="quality_regression")

    assert (root / "production" / "model.pkl").read_bytes() == b"one"
    assert (root / "model.pkl").read_bytes() == b"one"
    assert json.loads((root / "thresholds.json").read_text(encoding="utf-8"))["low"] == 0.3
    assert record["reason"] == "quality_regression"
    assert record["replaced_archive"] is not None


def test_rollback_without_archive_fails_loudly(tmp_path, monkeypatch):
    root = _install_tmp_layout(tmp_path, monkeypatch)
    _write_candidate(root, "c1", b"one")
    model_registry.install_production("c1")

    # Nothing was archived because there was no prior production model.
    assert model_registry.list_archive() == []
    with pytest.raises(model_registry.ModelRegistryError, match="No archived model"):
        model_registry.rollback()


def test_missing_candidate_raises(tmp_path, monkeypatch):
    _install_tmp_layout(tmp_path, monkeypatch)

    with pytest.raises(model_registry.ModelRegistryError, match="Candidate not found"):
        model_registry.get_candidate("nope")
