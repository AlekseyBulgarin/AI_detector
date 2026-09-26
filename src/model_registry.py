"""Model registry.

``models/model.pkl`` is no longer the source of truth. Artifacts live in
``models/production``, ``models/candidates`` and ``models/archive``; the flat
legacy paths are kept only as a compatibility mirror updated during promotion.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    ARCHIVE_DIR,
    CANDIDATES_DIR,
    MODELS_ROOT,
    PRODUCTION_DIR,
    THRESHOLDS_PATH,
)


class ModelRegistryError(RuntimeError):
    """Raised when a registry operation cannot be completed safely."""


def _timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_layout():
    for path in (PRODUCTION_DIR, CANDIDATES_DIR, ARCHIVE_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "production": PRODUCTION_DIR,
        "candidates": CANDIDATES_DIR,
        "archive": ARCHIVE_DIR,
    }


def candidate_dir(version):
    return CANDIDATES_DIR / version


def save_candidate(model, metadata, version, metrics=None, dump=None):
    """Persist a trained candidate. Never touches production paths."""
    ensure_layout()
    target = candidate_dir(version)
    if target.exists():
        raise ModelRegistryError(f"Candidate already exists: {version}")
    target.mkdir(parents=True, exist_ok=True)
    writer = dump or _default_dump
    writer(model, target / "model.pkl")
    (target / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if metrics is not None:
        (target / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return target


def _default_dump(model, path):
    import joblib

    joblib.dump(model, path)


def read_metadata(path):
    path = Path(path)
    if not path.exists():
        raise ModelRegistryError(f"Metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_candidate(version):
    target = candidate_dir(version)
    if not (target / "model.pkl").exists():
        raise ModelRegistryError(f"Candidate not found: {version}")
    metadata = read_metadata(target / "metadata.json")
    metrics = None
    if (target / "metrics.json").exists():
        metrics = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
    return {
        "version": version,
        "directory": target,
        "model_path": target / "model.pkl",
        "metadata": metadata,
        "metrics": metrics,
    }


def list_candidates():
    if not CANDIDATES_DIR.exists():
        return []
    return sorted(
        path.name
        for path in CANDIDATES_DIR.iterdir()
        if path.is_dir() and (path / "model.pkl").exists()
    )


def list_archive():
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(path.name for path in ARCHIVE_DIR.iterdir() if path.is_dir())


def get_production():
    model_path = PRODUCTION_DIR / "model.pkl"
    if not model_path.exists() and (MODELS_ROOT / "model.pkl").exists():
        model_path = MODELS_ROOT / "model.pkl"
    if not model_path.exists():
        raise ModelRegistryError("No production model is registered")
    metadata = {}
    for candidate in (PRODUCTION_DIR / "metadata.json", MODELS_ROOT / "metadata.json"):
        if candidate.exists():
            metadata = read_metadata(candidate)
            break
    metrics = None
    if (PRODUCTION_DIR / "metrics.json").exists():
        metrics = json.loads(
            (PRODUCTION_DIR / "metrics.json").read_text(encoding="utf-8")
        )
    return {
        "version": metadata.get("model_version", "unknown"),
        "model_path": model_path,
        "metadata": metadata,
        "metrics": metrics,
    }


def _copy_artifacts(source_dir, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("model.pkl", "metadata.json", "metrics.json", "thresholds.json"):
        origin = Path(source_dir) / name
        if origin.exists():
            shutil.copy2(origin, destination / name)


def archive_production(reason="manual_archive"):
    """Copy the current production artifacts into the archive. Non-destructive."""
    if not (PRODUCTION_DIR / "model.pkl").exists() and not (
        MODELS_ROOT / "model.pkl"
    ).exists():
        return None
    ensure_layout()
    production = get_production()
    label = f"{_timestamp()}__{production['version']}"
    destination = ARCHIVE_DIR / label
    source_dir = (
        PRODUCTION_DIR if (PRODUCTION_DIR / "model.pkl").exists() else MODELS_ROOT
    )
    _copy_artifacts(source_dir, destination)
    (destination / "archive.json").write_text(
        json.dumps(
            {
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "version": production["version"],
                "reason": reason,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def install_production(version, promoted_by="manual", gates=None):
    """Promote a candidate. Archives the previous production first."""
    candidate = get_candidate(version)
    archived = archive_production(reason=f"promoted_{version}")
    ensure_layout()
    _copy_artifacts(candidate["directory"], PRODUCTION_DIR)
    # Compatibility mirror for the existing loader path.
    shutil.copy2(candidate["model_path"], MODELS_ROOT / "model.pkl")
    shutil.copy2(candidate["directory"] / "metadata.json", MODELS_ROOT / "metadata.json")
    thresholds = candidate["directory"] / "thresholds.json"
    if thresholds.exists():
        shutil.copy2(thresholds, THRESHOLDS_PATH)
    record = {
        "version": version,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promoted_by": promoted_by,
        "gates": gates or {},
        "previous_archive": archived.name if archived else None,
    }
    (PRODUCTION_DIR / "promotion.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def latest_archive():
    entries = list_archive()
    if not entries:
        return None
    return entries[-1]


def rollback(reason="manual_rollback"):
    """Restore the most recent archived production model."""
    label = latest_archive()
    if label is None:
        raise ModelRegistryError("No archived model is available for rollback")
    source = ARCHIVE_DIR / label
    if not (source / "model.pkl").exists():
        raise ModelRegistryError(f"Archive is incomplete: {label}")
    ensure_layout()
    current = None
    if (PRODUCTION_DIR / "model.pkl").exists():
        current = archive_production(reason=f"rolled_back_from_{reason}")
    _copy_artifacts(source, PRODUCTION_DIR)
    shutil.copy2(source / "model.pkl", MODELS_ROOT / "model.pkl")
    if (source / "metadata.json").exists():
        shutil.copy2(source / "metadata.json", MODELS_ROOT / "metadata.json")
    if (PRODUCTION_DIR / "thresholds.json").exists():
        shutil.copy2(PRODUCTION_DIR / "thresholds.json", THRESHOLDS_PATH)
    record = {
        "restored_version": label,
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "replaced_archive": current.name if current else None,
    }
    (PRODUCTION_DIR / "rollback.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record
