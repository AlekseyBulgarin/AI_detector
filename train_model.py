"""Train a candidate model.

This script can only ever produce a *candidate*. It never replaces the
production model: promotion is a separate, explicitly human action performed by
``tools/promote_model.py`` after evaluation and gate checks.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.dataset_quality import build_quality_report
from src.evaluation import (
    calibrate_thresholds,
    evaluate_arrays,
    segment_report,
    write_evaluation,
)
from src.model_registry import ModelRegistryError, save_candidate
from src.splitter import group_key, group_split
from src.versions import DEFAULT_DATASET_VERSION, FEATURE_VERSION
from tools.dataset_loader import load_dataset


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent


def git_commit():
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def cross_validate_model(model, texts, labels):
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        texts,
        labels,
        cv=folds,
        scoring=("accuracy", "precision", "recall", "f1", "roc_auc"),
        error_score="raise",
    )
    return {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"])),
        }
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
    }


def format_metrics(metrics):
    matrix = metrics["confusion_matrix"]
    return (
        f"Accuracy: {metrics['accuracy']:.3f}; "
        f"Precision: {metrics['precision']:.3f}; "
        f"Recall: {metrics['recall']:.3f}; "
        f"F1: {metrics['f1']:.3f}; "
        f"ROC-AUC: {metrics['roc_auc']:.3f}; "
        f"FPR: {metrics['false_positive_rate']:.3f}; "
        f"FNR: {metrics['false_negative_rate']:.3f}; "
        f"Confusion matrix: {matrix}"
    )


def load_training_records(include_feedback=False, dataset_version=None):
    """Load training records from the registry or from the raw corpus."""
    if dataset_version:
        from src.dataset_registry import load_dataset_version

        _manifest, records = load_dataset_version(dataset_version)
        return records

    original = load_dataset(PROJECT_ROOT / "data" / "raw")
    for record in original:
        record["source"] = "original"
    if not include_feedback:
        return original

    from tools.export_feedback_dataset import load_exported_records

    feedback = load_exported_records()
    seen_hashes = {record["hash"] for record in original}
    seen_normalized = {record["normalized_hash"] for record in original}
    for record in feedback:
        if record["hash"] in seen_hashes or record["normalized_hash"] in seen_normalized:
            continue
        original.append(record)
        seen_hashes.add(record["hash"])
        seen_normalized.add(record["normalized_hash"])
    return original


def _stored_split(records):
    """Reuse the split recorded in a dataset manifest when one is present."""
    names = ("train", "validation", "frozen_test")
    if not records or any(record.get("split") not in names for record in records):
        return None
    buckets = {name: [] for name in names}
    for record in records:
        buckets[record["split"]].append(record)
    if not buckets["train"] or not buckets["frozen_test"]:
        return None
    groups = {group_key(record) for record in records}
    summary = {
        "counts": {name: len(buckets[name]) for name in names},
        "by_label": {
            name: {
                label: sum(1 for record in buckets[name] if record.get("label") == label)
                for label in ("human", "ai")
            }
            for name in names
        },
        "group_count": len(groups),
        "proportions": {"train": 0.7, "validation": 0.15, "frozen_test": 0.15},
        "feedback_in_frozen_test": any(
            str(record.get("source", "")).startswith("approved_feedback")
            for record in buckets["frozen_test"]
        ),
        "source": "dataset_manifest",
    }
    return buckets, summary


def make_split(records, seed=RANDOM_STATE):
    stored = _stored_split(records)
    if stored is not None:
        return stored
    assignment, summary = group_split(
        records,
        proportions={"train": 0.7, "validation": 0.15, "frozen_test": 0.15},
        seed=seed,
    )
    buckets = {"train": [], "validation": [], "frozen_test": []}
    for record in records:
        buckets[assignment[record["id"]]].append(record)
    if not buckets["frozen_test"]:
        buckets["frozen_test"] = buckets["validation"] or buckets["train"][-1:]
    if not buckets["train"]:
        raise ValueError("Training split is empty")
    if not buckets["validation"]:
        buckets["validation"] = buckets["train"][-1:]
    return buckets, summary


def dataset_quality_warnings(records, dataset_version=None):
    """Read the frozen dataset manifest warnings instead of recomputing them."""
    if dataset_version:
        try:
            from src.dataset_registry import load_dataset_version

            manifest, _ = load_dataset_version(dataset_version)
            warnings = manifest.get("quality_warnings")
            if warnings is not None:
                return warnings
        except Exception:  # noqa: BLE001 - fall back to a fresh analysis
            pass
    return build_quality_report(records).get("warnings", [])


def _labels(records):
    return [1 if record["label"] == "ai" else 0 for record in records]


def _texts(records):
    return [record["text"] for record in records]


def train_candidate(
    records,
    model_version,
    dataset_version=DEFAULT_DATASET_VERSION,
    seed=RANDOM_STATE,
    evaluate=True,
):
    """Fit candidate models, select one, evaluate it, and store it as a candidate."""
    from src.ml_pipeline import build_models

    buckets, split_summary = make_split(records, seed=seed)
    train_records = buckets["train"]
    validation_records = buckets["validation"]
    test_records = buckets["frozen_test"]

    train_texts = _texts(train_records)
    train_labels = np.array(_labels(train_records))

    results = {}
    fitted_models = {}
    for name, model in build_models().items():
        cross_validation = cross_validate_model(model, train_texts, train_labels)
        model.fit(train_texts, train_labels)
        validation_metrics = evaluate_arrays(
            _labels(validation_records),
            model.predict(_texts(validation_records)),
            model.predict_proba(_texts(validation_records))[:, 1],
        )
        results[name] = {
            "validation_metrics": validation_metrics,
            "cross_validation": cross_validation,
        }
        fitted_models[name] = model

    best_name = max(
        results,
        key=lambda name: (
            results[name]["cross_validation"]["f1"]["mean"],
            results[name]["cross_validation"]["roc_auc"]["mean"],
        ),
    )
    best_model = fitted_models[best_name]

    frozen_metrics = evaluate_arrays(
        _labels(test_records),
        best_model.predict(_texts(test_records)),
        best_model.predict_proba(_texts(test_records))[:, 1],
    )
    frozen_probabilities = best_model.predict_proba(_texts(test_records))[:, 1]
    # Thresholds are fitted on the validation split only: tuning them on the
    # frozen test set would leak test information into the decision layer.
    calibration_records = validation_records or train_records
    calibration_labels = _labels(calibration_records)
    calibration_probabilities = best_model.predict_proba(
        _texts(calibration_records)
    )[:, 1]
    thresholds = calibrate_thresholds(
        calibration_labels,
        calibration_probabilities,
        source="validation_roc" if validation_records else "train_roc",
    )
    segments = segment_report(
        test_records,
        _labels(test_records),
        [int(value) for value in best_model.predict(_texts(test_records))],
        [float(value) for value in frozen_probabilities],
    )

    created_at = datetime.now(timezone.utc).isoformat()
    feedback_samples = sum(
        1
        for record in records
        if str(record.get("source", "")).startswith("approved_feedback")
    )
    metadata = {
        "version": model_version,
        "model_version": model_version,
        "selected_model": best_name,
        "algorithm": type(best_model.named_steps["classifier"]).__name__,
        "dataset_version": dataset_version,
        "feature_version": FEATURE_VERSION,
        "dataset_size": len(records),
        "feedback_samples": feedback_samples,
        "train_size": len(train_records),
        "validation_size": len(validation_records),
        "test_size": len(test_records),
        "split_summary": split_summary,
        "sklearn_version": sklearn.__version__,
        "git_commit": git_commit(),
        "created_at": created_at,
        "training_date": created_at,
        "random_state": seed,
        "stage": "candidate",
        "metrics": results,
    }
    metrics = {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "frozen_test": frozen_metrics,
        "validation": results[best_name]["validation_metrics"],
        "thresholds": thresholds,
        "segments": segments,
        "split_summary": split_summary,
        "quality_warnings": dataset_quality_warnings(records, dataset_version),
        "candidates": {
            name: {
                "f1": values["cross_validation"]["f1"]["mean"],
                "roc_auc": values["cross_validation"]["roc_auc"]["mean"],
            }
            for name, values in results.items()
        },
    }

    directory = save_candidate(best_model, metadata, model_version, metrics=metrics)
    evaluation = {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "evaluated_at": created_at,
        "selected_model": best_name,
        "metrics": frozen_metrics,
        "validation_metrics": results[best_name]["validation_metrics"],
        "cross_validation": results[best_name]["cross_validation"],
        "thresholds": thresholds,
        "segments": segments,
        "split_summary": split_summary,
        "quality_warnings": dataset_quality_warnings(records, dataset_version),
    }
    if evaluate:
        write_evaluation(evaluation, model_version)
        (directory / "thresholds.json").write_text(
            json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "version": model_version,
        "directory": directory,
        "metadata": metadata,
        "metrics": metrics,
        "evaluation": evaluation,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an AI Detector candidate")
    parser.add_argument(
        "--include-feedback",
        action="store_true",
        help="Include exported approved and consented feedback samples",
    )
    parser.add_argument("--model-version", help="Candidate version, for example v3")
    parser.add_argument(
        "--dataset",
        help="Registry dataset version to train from, for example v1",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_version = args.model_version or datetime.now(timezone.utc).strftime(
        "candidate-%Y%m%d%H%M%S"
    )
    records = load_training_records(args.include_feedback, args.dataset)
    dataset_version = args.dataset or DEFAULT_DATASET_VERSION
    try:
        result = train_candidate(
            records, model_version, dataset_version=dataset_version, seed=args.seed
        )
    except ModelRegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Selected model: {result['metadata']['selected_model']}")
    print(f"Frozen test: {format_metrics(result['evaluation']['metrics'])}")
    print(f"Candidate saved: {result['directory']}")
    print("Candidate is NOT promoted. Run tools/promote_model.py after review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
