"""Immutable dataset registry.

Approved feedback is never written back into ``data/raw``. Instead a new,
versioned snapshot is assembled from the raw corpus plus reviewed feedback and
frozen with a manifest. Old versions are never mutated.
"""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT
from src.dataset_quality import build_quality_report, write_reports
from src.splitter import group_split, split_manifest


DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "manifests"
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAINABLE_LABELS = {"human", "ai"}


class DatasetRegistryError(RuntimeError):
    """Raised when a dataset version cannot be created safely."""


def ensure_layout():
    for path in (
        DATASETS_DIR,
        MANIFESTS_DIR,
        PROCESSED_DIR,
        FEEDBACK_DIR / "pending",
        FEEDBACK_DIR / "approved",
        PROJECT_ROOT / "data" / "raw" / "human",
        PROJECT_ROOT / "data" / "raw" / "ai",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return DATASETS_DIR


def next_version(root=None):
    root = Path(root or DATASETS_DIR)
    existing = []
    for path in root.glob("v*"):
        suffix = path.name[1:]
        if suffix.isdigit():
            existing.append(int(suffix))
    return f"v{max(existing, default=0) + 1}"


def _record_digest(records):
    payload = "\n".join(
        sorted(f"{record.get('id')}|{record.get('hash')}" for record in records)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_for(records, dataset_version, split_summary=None, sources=None):
    labels = Counter(record.get("label") for record in records)
    source_names = sorted(
        {str(record.get("source", "unknown")) for record in records}
    )
    duplicate_count = sum(
        1
        for count in Counter(record.get("normalized_hash") for record in records).values()
        if count > 1
    )
    counts = split_summary["counts"] if split_summary else {}
    return {
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(records),
        "human_samples": labels.get("human", 0),
        "ai_samples": labels.get("ai", 0),
        "ai_assisted_samples": labels.get("ai_assisted", 0),
        "feedback_samples": sum(
            1
            for record in records
            if str(record.get("source", "")).startswith("approved_feedback")
        ),
        "sources": sources or source_names,
        "duplicate_count": duplicate_count,
        "train_count": counts.get("train", 0),
        "validation_count": counts.get("validation", 0),
        "test_count": counts.get("frozen_test", 0),
        "trainable_samples": sum(
            1 for record in records if record.get("label") in TRAINABLE_LABELS
        ),
        "records_digest": _record_digest(records),
    }


def create_dataset_version(
    records,
    version=None,
    output_root=None,
    include_feedback=True,
    proportions=None,
    seed=42,
    quality_report_path=None,
    write_quality=True,
):
    """Freeze records into a new immutable dataset version.

    Returns ``(manifest, directory)``. Raises when quality gates fail, so an
    unusable dataset can never become a registry entry by accident.
    """
    ensure_layout()
    root = Path(output_root or DATASETS_DIR)
    dataset_version = version or next_version(root)

    selected = []
    for record in records:
        is_feedback = str(record.get("source", "")).startswith("approved_feedback")
        if is_feedback and not include_feedback:
            continue
        selected.append(dict(record))

    if not selected:
        raise DatasetRegistryError("No records available for a dataset version")

    assignment, split_summary = group_split(selected, proportions=proportions, seed=seed)
    report = build_quality_report(selected)
    if write_quality:
        write_reports(report)

    version_dir = root / dataset_version
    if version_dir.exists():
        raise DatasetRegistryError(f"Dataset version already exists: {dataset_version}")
    version_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for record in selected:
        payload = {
            "id": record.get("id"),
            "label": record.get("label"),
            "text": record.get("text"),
            "hash": record.get("hash"),
            "normalized_hash": record.get("normalized_hash"),
            "source": record.get("source", "raw"),
            "split": assignment[record.get("id")],
            "group": record.get("group"),
            "characters": record.get("characters"),
            "words": record.get("words"),
            "sentences": record.get("sentences"),
            "trainable": record.get("label") in TRAINABLE_LABELS,
        }
        for field in ("author", "topic", "prompt", "generation_session", "consent", "review_status"):
            if field in record:
                payload[field] = record[field]
        lines.append(json.dumps(payload, ensure_ascii=False))

    (version_dir / "samples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = manifest_for(selected, dataset_version, split_summary)
    manifest["quality_gates"] = report["quality_gates"]["passed"]
    manifest["quality_warnings"] = report.get("warnings", [])
    manifest["split_summary"] = split_summary
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (version_dir / "split.json").write_text(
        json.dumps(
            split_manifest(selected, assignment, split_summary, dataset_version),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / f"{dataset_version}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MANIFESTS_DIR / "current.txt").write_text(dataset_version, encoding="utf-8")
    return manifest, version_dir


def load_dataset_version(dataset_version, root=None):
    root = Path(root or DATASETS_DIR)
    version_dir = root / dataset_version
    manifest_path = version_dir / "manifest.json"
    if not manifest_path.exists():
        raise DatasetRegistryError(f"Unknown dataset version: {dataset_version}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for line in (version_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return manifest, records


def list_versions(root=None):
    root = Path(root or DATASETS_DIR)
    if not root.exists():
        return []
    return sorted(path.name for path in root.glob("v*") if path.is_dir())


def current_version():
    try:
        return (MANIFESTS_DIR / "current.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def purge_staging(root=None):
    """Remove exported staging files without touching registered versions."""
    root = Path(root or FEEDBACK_DIR)
    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.name != ".gitkeep":
            path.unlink()
            removed += 1
    return removed
