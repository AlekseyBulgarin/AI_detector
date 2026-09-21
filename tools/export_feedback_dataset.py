import csv
from pathlib import Path

from src.database import list_approved_feedback


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def export_feedback_dataset(output_root=None, database_path=None):
    """Export only approved, consented human/AI feedback into class folders."""
    destination = Path(output_root or PROJECT_ROOT / "data" / "raw" / "feedback")
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
    ]
    metadata = []
    for item in records:
        label = item["user_label"]
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
            }
        )
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)
    return metadata_path, metadata


if __name__ == "__main__":
    path, records = export_feedback_dataset()
    print(f"Exported {len(records)} approved feedback samples to {path}")
