import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from tools.dataset_loader import create_metadata, load_dataset, validate_dataset
except ImportError:
    from dataset_loader import create_metadata, load_dataset, validate_dataset


def _stats(records, field):
    values = [record[field] for record in records]
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
    }


def _duplicate_groups(records, field):
    groups = defaultdict(list)
    for record in records:
        groups[record[field]].append(record["filename"])
    return [files for files in groups.values() if len(files) > 1]


def analyze_dataset(data_root="data/raw", report_path="reports/dataset_quality.json"):
    all_records = load_dataset(data_root, include_duplicates=True)
    quality_errors = validate_dataset(all_records)
    by_class = {
        label: [record for record in all_records if record["label"] == label]
        for label in ("human", "ai")
    }
    report = {
        "total_samples": len(all_records),
        "samples_per_class": {label: len(records) for label, records in by_class.items()},
        "statistics": {
            "characters": _stats(all_records, "characters"),
            "words": _stats(all_records, "words"),
            "sentences": _stats(all_records, "sentences"),
            "by_class": {
                label: {
                    "characters": _stats(records, "characters"),
                    "words": _stats(records, "words"),
                    "sentences": _stats(records, "sentences"),
                }
                for label, records in by_class.items()
            },
        },
        "duplicates": {
            "exact": _duplicate_groups(all_records, "hash"),
            "normalized": _duplicate_groups(all_records, "normalized_hash"),
        },
        "validation_errors": quality_errors,
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    create_metadata(all_records)
    return report


if __name__ == "__main__":
    result = analyze_dataset()
    print(json.dumps(result, ensure_ascii=False, indent=2))
