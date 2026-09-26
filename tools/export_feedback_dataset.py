import csv
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import list_approved_feedback  # noqa: E402
from src.feedback_validator import normalize_text, text_statistics  # noqa: E402
from src.privacy import can_export_for_training  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "feedback" / "approved"


def export_feedback_dataset(output_root=None, database_path=None):
    """Export approved, consented human/AI feedback into a staging folder.

    The export is only a staging step. Samples enter training exclusively via a
    versioned dataset registry entry, never by being written back into
    ``data/raw``.
    """
    destination = Path(output_root or DEFAULT_OUTPUT)
    destination.mkdir(parents=True, exist_ok=True)
    records = list_approved_feedback(database_path)
    for label in ("human", "ai"):
        class_dir = destination / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for old_file in class_dir.glob("feedback_*.txt"):
            old_file.unlink()

    metadata_path = destination / "metadata.csv"
    fields = [
        "feedback_id",
        "label",
        "filename",
        "text_hash",
        "prediction_label",
        "prediction_probability",
        "model_version",
        "created_at",
        "consent",
        "review_status",
        "priority_category",
        "priority_score",
    ]
    metadata = []
    for item in records:
        label = item["user_label"]
        if not can_export_for_training(
            item.get("allow_training"),
            bool(item.get("text_content")),
            item.get("status"),
            item.get("user_label"),
        ):
            continue
        filename = f"feedback_{item['id']}.txt"
        path = destination / label / filename
        text = item["text_content"].strip() + "\n"
        path.write_text(text, encoding="utf-8")
        metadata.append(
            {
                "feedback_id": item["id"],
                "label": label,
                "filename": str(path.relative_to(destination)),
                "text_hash": item["text_hash"],
                "prediction_label": item["prediction_label"],
                "prediction_probability": item["prediction_probability"],
                "model_version": item["model_version"],
                "created_at": item["created_at"],
                "consent": True,
                "review_status": "approved",
                "priority_category": item.get("priority_category", "edge_case"),
                "priority_score": item.get("priority_score", 0),
            }
        )
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)
    return metadata_path, metadata


def load_exported_records(output_root=None):
    """Read the staged export back into registry-ready records."""
    destination = Path(output_root or DEFAULT_OUTPUT)
    records = []
    if not destination.exists():
        return records
    for label in ("human", "ai"):
        for path in sorted((destination / label).glob("feedback_*.txt")):
            text = path.read_text(encoding="utf-8")
            stats = text_statistics(text)
            records.append(
                {
                    "id": f"feedback_{path.stem.replace('feedback_', '')}",
                    "label": label,
                    "label_id": 1 if label == "ai" else 0,
                    "filename": str(path),
                    "text": text,
                    "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "normalized_hash": hashlib.sha256(
                        normalize_text(text).encode("utf-8")
                    ).hexdigest(),
                    "characters": stats["characters"],
                    "words": stats["words"],
                    "sentences": stats["sentences"],
                    "source": "approved_feedback",
                    "consent": True,
                    "review_status": "approved",
                }
            )
    return records


if __name__ == "__main__":
    path, rows = export_feedback_dataset()
    print(f"Exported {len(rows)} approved feedback samples to {path}")
