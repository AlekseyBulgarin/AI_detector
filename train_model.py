import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.ml_pipeline import build_models, evaluate_model
from tools.dataset_loader import load_dataset


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent


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
        f"Confusion matrix: {matrix}"
    )


def write_report(results, output_path):
    lines = [
        "# Model Comparison",
        "",
        "The final test set was held out before model fitting. Cross-validation was run only on the training portion.",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, result in results.items():
        metrics = result["test_metrics"]
        lines.append(
            f"| {name} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['roc_auc']:.3f} | "
            f"`{metrics['confusion_matrix']}` |"
        )
    lines.extend(["", "## Cross-validation", ""])
    for name, result in results.items():
        lines.append(f"### {name}")
        for metric, values in result["cross_validation"].items():
            lines.append(f"- {metric}: {values['mean']:.3f} +/- {values['std']:.3f}")
        lines.append("")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def load_training_records(include_feedback=False):
    """Load original data plus exported approved feedback only when requested."""
    original = load_dataset(PROJECT_ROOT / "data" / "raw")
    for record in original:
        record["source"] = "original"
    if not include_feedback:
        return original

    feedback_root = PROJECT_ROOT / "data" / "raw" / "feedback"
    if not feedback_root.exists():
        return original
    feedback = load_dataset(feedback_root)
    seen_hashes = {record["hash"] for record in original}
    seen_normalized = {record["normalized_hash"] for record in original}
    for record in feedback:
        if record["hash"] in seen_hashes or record["normalized_hash"] in seen_normalized:
            continue
        record["source"] = "approved_feedback"
        original.append(record)
        seen_hashes.add(record["hash"])
        seen_normalized.add(record["normalized_hash"])
    return original


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a versioned AI Detector model")
    parser.add_argument(
        "--include-feedback",
        action="store_true",
        help="Include only exported approved and consented feedback samples",
    )
    parser.add_argument("--model-version", help="Model version, for example phase2-v2")
    parser.add_argument("--output-model", type=Path, help="Optional output model path")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Copy the versioned artifact and metadata to the active model paths",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_version = args.model_version or ("phase2-v2" if args.include_feedback else "phase2-v1")
    records = load_training_records(args.include_feedback)
    texts = np.array([record["text"] for record in records], dtype=object)
    labels = np.array([record["label_id"] for record in records])

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    results = {}
    fitted_models = {}
    for name, model in build_models().items():
        cross_validation = cross_validate_model(model, train_texts, train_labels)
        model.fit(train_texts, train_labels)
        test_metrics = evaluate_model(model, test_texts, test_labels)
        results[name] = {
            "test_metrics": test_metrics,
            "cross_validation": cross_validation,
        }
        fitted_models[name] = model
        print(f"{name}: {format_metrics(test_metrics)}")

    best_name = max(
        results,
        key=lambda name: (
            results[name]["cross_validation"]["f1"]["mean"],
            results[name]["cross_validation"]["roc_auc"]["mean"],
        ),
    )
    version_suffix = model_version.removeprefix("phase2-")
    model_path = args.output_model or PROJECT_ROOT / "models" / f"model_{version_suffix}.pkl"
    metadata_path = PROJECT_ROOT / "models" / f"metadata_{version_suffix}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_name], model_path)

    created_at = datetime.now(timezone.utc).isoformat()
    feedback_samples = sum(record.get("source") == "approved_feedback" for record in records)
    metadata = {
        "version": model_version,
        "model_version": model_version,
        "selected_model": best_name,
        "dataset_size": len(records),
        "feedback_samples": feedback_samples,
        "dataset_version": f"original+approved-feedback-{feedback_samples}",
        "train_size": len(train_labels),
        "test_size": len(test_labels),
        "features_version": "linguistic-v1+tfidf-v1",
        "sklearn_version": sklearn.__version__,
        "training_date": created_at,
        "created_at": created_at,
        "random_state": RANDOM_STATE,
        "metrics": results,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(results, PROJECT_ROOT / "reports" / "model_comparison.md")
    if args.promote:
        shutil.copyfile(model_path, PROJECT_ROOT / "models" / "model.pkl")
        shutil.copyfile(metadata_path, PROJECT_ROOT / "models" / "metadata.json")
    print(f"Selected model: {best_name}")
    print(f"Saved model: {model_path}")
    return metadata


if __name__ == "__main__":
    main()
