"""Build a frozen dataset version from raw data plus approved feedback.

Usage:
    python tools/build_dataset_version.py
    python tools/build_dataset_version.py --no-feedback --version raw-only
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_registry import (  # noqa: E402
    DatasetRegistryError,
    create_dataset_version,
    ensure_layout,
)
from tools.dataset_loader import load_dataset  # noqa: E402
from tools.export_feedback_dataset import load_exported_records  # noqa: E402


def collect_records(include_feedback=True):
    ensure_layout()
    records = []
    for record in load_dataset(PROJECT_ROOT / "data" / "raw"):
        record["source"] = "raw"
        records.append(record)
    if include_feedback:
        records.extend(load_exported_records())
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create a versioned dataset")
    parser.add_argument("--version", help="Explicit version, for example v3")
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Freeze only the original raw corpus",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Print the manifest")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = collect_records(include_feedback=not args.no_feedback)
    try:
        manifest, directory = create_dataset_version(
            records,
            version=args.version,
            include_feedback=not args.no_feedback,
            seed=args.seed,
        )
    except DatasetRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(
            f"Created {manifest['dataset_version']} at {directory} "
            f"({manifest['total_samples']} samples, "
            f"{manifest['train_count']}/{manifest['validation_count']}/"
            f"{manifest['test_count']} split, "
            f"gates={'PASS' if manifest['quality_gates'] else 'FAIL'})"
        )
        for warning in manifest.get("quality_warnings", []):
            print(f"WARNING: {warning}")
    return 0 if manifest["quality_gates"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
