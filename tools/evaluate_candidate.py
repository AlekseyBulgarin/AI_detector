"""Re-evaluate a stored candidate against the frozen test set.

Usage:
    python tools/evaluate_candidate.py --version v2
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402

from src.config import THRESHOLDS_PATH  # noqa: E402
from src.evaluation import (  # noqa: E402
    calibrate_thresholds,
    evaluate_arrays,
    segment_report,
    write_evaluation,
)
from src.model_registry import ModelRegistryError, get_candidate  # noqa: E402


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def frozen_records_for(candidate):
    from src.dataset_registry import load_dataset_version
    from src.splitter import group_split
    from train_model import load_training_records

    dataset_version = candidate["metadata"].get("dataset_version")
    try:
        _manifest, records = load_dataset_version(dataset_version)
        return [record for record in records if record.get("split") == "frozen_test"], dataset_version
    except Exception:
        pass

    records = load_training_records(
        include_feedback=bool(candidate["metadata"].get("feedback_samples")),
        dataset_version=None,
    )
    assignment, _summary = group_split(records)
    return [
        record for record in records if assignment[record["id"]] == "frozen_test"
    ], dataset_version or "raw-corpus-v0"


def evaluate_candidate(version):
    candidate = get_candidate(version)
    model = joblib.load(candidate["model_path"])
    records, dataset_version = frozen_records_for(candidate)
    return _evaluate(version, model, records, dataset_version, candidate)


def evaluate_production():
    """Baseline the current production model on the frozen test set."""
    from src.model_registry import get_production

    production = get_production()
    model = joblib.load(production["model_path"])
    records, dataset_version = frozen_records_for(production)
    return _evaluate(
        production["version"], model, records, dataset_version, production
    )


def _quality_warnings(source, dataset_version):
    """Prefer the frozen dataset manifest over anything recomputed."""
    if dataset_version:
        try:
            from src.dataset_registry import load_dataset_version

            manifest, _records = load_dataset_version(dataset_version)
            warnings = manifest.get("quality_warnings")
            if warnings is not None:
                return warnings
        except Exception:  # noqa: BLE001 - fall back to stored metrics
            pass
    return (
        (source.get("metrics") or {}).get("quality_warnings")
        or (source.get("metadata") or {}).get("quality_warnings")
        or []
    )


def _evaluate(version, model, records, dataset_version, source):
    if not records:
        raise ModelRegistryError(f"No frozen test samples available for {version}")

    labels = [1 if record["label"] == "ai" else 0 for record in records]
    texts = [record["text"] for record in records]
    predictions = [int(value) for value in model.predict(texts)]
    probabilities = [float(value) for value in model.predict_proba(texts)[:, 1]]
    metrics = evaluate_arrays(labels, predictions, probabilities)
    metadata = source.get("metadata", {})
    is_candidate = bool(source.get("directory"))
    directory = Path(source.get("directory") or Path(source["model_path"]).parent)
    thresholds_path = directory / "thresholds.json" if is_candidate else THRESHOLDS_PATH
    thresholds = _read_json(thresholds_path)
    if thresholds is None and is_candidate:
        # No training-time thresholds yet: fit them once and persist them next
        # to the candidate so later evaluations reuse the same decision layer.
        thresholds = calibrate_thresholds(
            labels, probabilities, source="frozen_test_roc_recalibrated"
        )
        thresholds_path.write_text(
            json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if thresholds is None:
        # Production thresholds are owned by the promotion flow and are never
        # refitted in place; report what evaluation would produce instead.
        thresholds = calibrate_thresholds(
            labels, probabilities, source="frozen_test_roc_recalibrated"
        )
    report = {
        "model_version": version,
        "dataset_version": dataset_version,
        "metrics": metrics,
        "thresholds": thresholds,
        "segments": segment_report(records, labels, predictions, probabilities),
        "split_summary": metadata.get("split_summary")
        or (source.get("metrics") or {}).get("split_summary"),
        "quality_warnings": _quality_warnings(source, dataset_version),
        "sample_size_warning": (
            None
            if len(records) >= 30
            else (
                f"Frozen test set holds only {len(records)} samples: metrics are "
                "directional and must not be presented as reliable accuracy."
            )
        ),
    }
    path = write_evaluation(report, version)
    return path, report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a candidate model")
    parser.add_argument("--version", help="Candidate version")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Evaluate the current production model instead of a candidate",
    )
    args = parser.parse_args(argv)
    if not args.production and not args.version:
        parser.error("either --version or --production is required")
    if args.production and args.version:
        parser.error("--version and --production are mutually exclusive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.production:
            path, report = evaluate_production()
        else:
            path, report = evaluate_candidate(args.version)
    except (ModelRegistryError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    metrics = report["metrics"]
    print(f"Evaluation written: {path}")
    print(
        f"accuracy={metrics['accuracy']:.3f} precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
        f"roc_auc={metrics['roc_auc']:.3f} fpr={metrics['false_positive_rate']:.3f} "
        f"fnr={metrics['false_negative_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
