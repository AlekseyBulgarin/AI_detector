"""Roll production back to the most recently archived model.

Usage:
    python tools/rollback_model.py
    python tools/rollback_model.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR  # noqa: E402
from src.model_registry import (  # noqa: E402
    ModelRegistryError,
    get_production,
    latest_archive,
    rollback,
)


def rollback_production(dry_run=False, reason="manual_rollback"):
    label = latest_archive()
    if label is None:
        raise ModelRegistryError("No archived model is available for rollback")
    current = None
    try:
        current = get_production()["version"]
    except ModelRegistryError:
        current = None

    if dry_run:
        return {"rolled_back": False, "dry_run": True, "target": label, "current": current}

    record = rollback(reason=reason)
    log_path = REPORTS_DIR / "rollbacks.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"rolled_back": True, "record": record, "target": label, "current": current}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Roll back the production model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reason", default="manual_rollback")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = rollback_production(dry_run=args.dry_run, reason=args.reason)
    except ModelRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if result.get("dry_run"):
        print(f"Dry run. Would restore {result['target']} over {result['current']}.")
        return 0
    print(f"Rolled back to {result['target']} (was {result['current']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
