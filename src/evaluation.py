"""Evaluation platform.

Every candidate is scored against the same frozen evaluation dataset with the
same metric definitions, so production-vs-candidate deltas are meaningful.

Class convention throughout this module: label ``1`` is AI, label ``0`` is
human. A false positive is therefore human text classified as AI, which is the
most damaging error for a teacher and is reported separately.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import EVALUATION_DIR, REPORTS_DIR


LENGTH_BUCKETS = ((0, 100, "0-99"), (100, 250, "100-249"), (250, 500, "250-499"), (500, 10**9, "500+"))
CONFIDENCE_BUCKETS = ((0.0, 0.35, "0-35"), (0.35, 0.65, "35-65"), (0.65, 0.85, "65-85"), (0.85, 1.01, "85-100"))


def evaluate_arrays(labels, predictions, probabilities):
    """Core metrics shared by every evaluation path."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities, dtype=float)

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    auc = (
        float(roc_auc_score(labels, probabilities))
        if len(set(labels.tolist())) > 1
        else 0.0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": matrix.tolist(),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "samples": int(len(labels)),
    }


def evaluate_model(model, texts, labels):
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)[:, 1]
    return evaluate_arrays(labels, predictions, probabilities)


def length_bucket(characters):
    for low, high, name in LENGTH_BUCKETS:
        if low <= characters < high:
            return name
    return "unknown"


def confidence_bucket(probability):
    for low, high, name in CONFIDENCE_BUCKETS:
        if low <= probability < high:
            return name
    return "unknown"


def _bucket_report(keys, labels, predictions, probabilities):
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(str(key), []).append(index)
    report = {}
    for key, indexes in sorted(groups.items()):
        report[key] = evaluate_arrays(
            [labels[index] for index in indexes],
            [predictions[index] for index in indexes],
            [probabilities[index] for index in indexes],
        )
    return report


def segment_report(records, labels, predictions, probabilities):
    """Break metrics down by length, topic, source and confidence range."""
    labels = list(labels)
    predictions = list(predictions)
    probabilities = [float(value) for value in probabilities]

    length_keys = [
        length_bucket(record.get("characters", len(str(record.get("text", "")))))
        for record in records
    ]
    topic_keys = [record.get("topic") or "unknown" for record in records]
    source_keys = [record.get("source") or "unknown" for record in records]
    confidence_keys = [confidence_bucket(value) for value in probabilities]

    return {
        "by_length": _bucket_report(length_keys, labels, predictions, probabilities),
        "by_topic": _bucket_report(topic_keys, labels, predictions, probabilities),
        "by_source": _bucket_report(source_keys, labels, predictions, probabilities),
        "by_confidence": _bucket_report(
            confidence_keys, labels, predictions, probabilities
        ),
    }


def calibrate_thresholds(labels, probabilities, alpha=0.05, source="frozen_test_roc"):
    """Derive abstention thresholds from measured data instead of guessing.

    ``low`` is the largest score at which declaring "human" still keeps the
    false-negative rate at or below ``alpha``. ``high`` is the smallest score at
    which declaring "AI" keeps the false-positive rate at or below ``alpha``.

    ``source`` records which split the thresholds were fitted on. Fitting on the
    validation split keeps the frozen test set untouched by any threshold choice.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) == 0 or len(set(labels.tolist())) < 2:
        return {
            "low": 0.35,
            "high": 0.65,
            "alpha": alpha,
            "source": "fallback_insufficient_data",
            "samples": int(len(labels)),
        }

    candidates = sorted(set(probabilities.tolist()))
    low = 0.0
    high = None
    for threshold in candidates:
        ai_mask = labels == 1
        total_ai = int(np.sum(ai_mask))
        missed = int(np.sum(probabilities[ai_mask] <= threshold))
        miss_rate = missed / total_ai if total_ai else 0.0
        if miss_rate <= alpha:
            low = threshold

    human_mask = labels == 0
    total_human = int(np.sum(human_mask))
    for threshold in candidates:
        false_positives = int(np.sum(probabilities[human_mask] >= threshold))
        fpr = false_positives / total_human if total_human else 0.0
        if fpr <= alpha:
            high = threshold
            break
    if high is None:
        # Alpha cannot be met at any threshold: keep the most selective option
        # rather than silently relaxing the constraint.
        high = max(candidates)

    if low > high:
        high = low
    return {
        "low": round(float(low), 4),
        "high": round(float(high), 4),
        "alpha": alpha,
        "source": source,
        "samples": int(len(labels)),
        "calculated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_thresholds(thresholds, path=None):
    destination = Path(path or REPORTS_DIR / "thresholds.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def write_evaluation(report, version, directory=None):
    destination = Path(directory or EVALUATION_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{version}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def evaluation_path(version, directory=None):
    return Path(directory or EVALUATION_DIR) / f"{version}.json"


def load_evaluation(version, directory=None):
    path = evaluation_path(version, directory)
    if not path.exists():
        raise FileNotFoundError(f"No evaluation for {version}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_evaluation(
    model,
    records,
    labels,
    dataset_version,
    model_version,
    alpha=0.05,
):
    """Evaluate a model on a fixed record set and return a full report."""
    texts = [record.get("text", "") for record in records]
    predictions = [int(value) for value in model.predict(texts)]
    probabilities = [float(value) for value in model.predict_proba(texts)[:, 1]]
    metrics = evaluate_arrays(labels, predictions, probabilities)
    thresholds = calibrate_thresholds(labels, probabilities, alpha=alpha)
    return {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "thresholds": thresholds,
        "segments": segment_report(records, labels, predictions, probabilities),
        "predictions": [
            {
                "id": record.get("id"),
                "label": int(label),
                "prediction": prediction,
                "probability": probability,
            }
            for record, label, prediction, probability in zip(
                records, labels, predictions, probabilities
            )
        ],
    }
