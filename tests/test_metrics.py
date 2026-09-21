import pytest
import numpy as np

from src.ml_pipeline import evaluate_model, evaluate_predictions


def test_evaluate_predictions_returns_required_metrics_and_confusion_matrix():
    metrics = evaluate_predictions(
        labels=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        probabilities=[0.1, 0.7, 0.8, 0.9],
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
    }
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert all(0 <= metrics[name] <= 1 for name in ("accuracy", "precision", "recall", "f1", "roc_auc"))


def test_evaluate_model_uses_fitted_model_predictions():
    class FakeModel:
        def predict(self, texts):
            assert texts == ["a", "b"]
            return [0, 1]

        def predict_proba(self, texts):
            assert texts == ["a", "b"]
            return np.array([[0.9, 0.1], [0.2, 0.8]])

    metrics = evaluate_model(FakeModel(), ["a", "b"], [0, 1])

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
