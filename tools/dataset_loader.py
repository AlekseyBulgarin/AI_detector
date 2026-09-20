import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path


CLASS_LABELS = {"human": 0, "ai": 1}
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"[.!?]+")


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be loaded safely."""


def normalize_text(text):
    """Normalize text for duplicate detection without changing model input."""
    return " ".join(text.lower().split())


def _text_statistics(text):
    words = WORD_PATTERN.findall(text.lower())
    sentences = [part for part in SENTENCE_PATTERN.split(text) if part.strip()]
    return len(text), len(words), len(sentences)


def _record(label, path, text):
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    normalized_hash = hashlib.sha256(
        normalize_text(text).encode("utf-8")
    ).hexdigest()
    characters, words, sentences = _text_statistics(text)
    return {
        "id": f"{label}_{path.stem}",
        "label": label,
        "label_id": CLASS_LABELS[label],
        "filename": str(path),
        "text": text,
        "hash": raw_hash,
        "normalized_hash": normalized_hash,
        "characters": characters,
        "words": words,
        "sentences": sentences,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_dataset(records):
    """Return validation problems without deleting or changing records."""
    errors = []
    if not records:
        errors.append("Dataset is empty")
        return errors

    labels = {record.get("label") for record in records}
    missing_labels = set(CLASS_LABELS) - labels
    if missing_labels:
        errors.append(f"Missing class labels: {sorted(missing_labels)}")

    seen_ids = set()
    for record in records:
        record_id = record.get("id")
        if record_id in seen_ids:
            errors.append(f"Duplicate record id: {record_id}")
        seen_ids.add(record_id)
        if record.get("label") not in CLASS_LABELS:
            errors.append(f"Unknown label for {record.get('filename')}")
        if not record.get("text", "").strip():
            errors.append(f"Empty text: {record.get('filename')}")

    return errors


def load_dataset(data_root="data/raw", include_duplicates=False):
    """Load raw text files in deterministic order.

    Exact and normalized duplicates are filtered by default. The analyzer can
    request all records to report duplicate problems instead.
    """
    root = Path(data_root)
    records = []
    seen_hashes = set()
    seen_normalized_hashes = set()
    read_errors = []

    for label in sorted(CLASS_LABELS):
        class_dir = root / label
        for path in sorted(class_dir.glob("*.txt"), key=lambda item: item.name):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as exc:
                read_errors.append(f"{path}: {exc}")
                continue

            record = _record(label, path, text)
            if not include_duplicates:
                if (
                    record["hash"] in seen_hashes
                    or record["normalized_hash"] in seen_normalized_hashes
                ):
                    continue
                seen_hashes.add(record["hash"])
                seen_normalized_hashes.add(record["normalized_hash"])
            records.append(record)

    if read_errors:
        raise DatasetValidationError("Unable to read dataset files: " + "; ".join(read_errors))

    errors = validate_dataset(records)
    if errors:
        raise DatasetValidationError("; ".join(errors))
    return records


def create_metadata(records, output_path="data/metadata/metadata.csv"):
    """Write the requested metadata fields and return the output path."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "label",
        "filename",
        "hash",
        "normalized_hash",
        "characters",
        "words",
        "sentences",
        "created_at",
    ]
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields} for record in records)
    return destination
