import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tools.dataset_loader import create_metadata, load_dataset
except ImportError:
    from dataset_loader import create_metadata, load_dataset

from src.dataset_quality import build_quality_report, write_reports  # noqa: E402


def analyze_dataset(data_root="data/raw", report_path="reports/dataset_quality.json"):
    """Analyze the raw corpus and write the JSON + Markdown quality reports."""
    all_records = load_dataset(data_root, include_duplicates=True)
    report = build_quality_report(all_records)
    json_path = Path(report_path)
    md_path = json_path.with_suffix(".md")
    write_reports(report, json_path, md_path)
    create_metadata(all_records)
    return report


if __name__ == "__main__":
    result = analyze_dataset()
    print(json.dumps(result["quality_gates"], ensure_ascii=False, indent=2))
