import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
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


def main():
    records = load_dataset(PROJECT_ROOT / "data" / "raw")
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
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    metadata_path = PROJECT_ROOT / "models" / "metadata.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_name], model_path)

    metadata = {
        "model_version": "phase2-v1",
        "selected_model": best_name,
        "dataset_size": len(records),
        "train_size": len(train_labels),
        "test_size": len(test_labels),
        "features_version": "linguistic-v1+tfidf-v1",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "metrics": results,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(results, PROJECT_ROOT / "reports" / "model_comparison.md")
    print(f"Selected model: {best_name}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
