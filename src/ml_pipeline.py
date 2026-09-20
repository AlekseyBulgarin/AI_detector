import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.features import extract_features


LINGUISTIC_COLUMNS = [
    "avg_sentence_length",
    "unique_ratio",
    "stopword_ratio",
    "punctuation_ratio",
    "word_length_std",
]


class TextFeatureFrame(BaseEstimator, TransformerMixin):
    """Convert raw texts into a frame for ColumnTransformer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = [str(text) for text in X]
        features = [extract_features(text) for text in texts]
        frame = pd.DataFrame(features, columns=LINGUISTIC_COLUMNS)
        frame.insert(0, "text", texts)
        return frame


def _classifier():
    return LogisticRegression(max_iter=1000, random_state=42)


def _pipeline(transformers):
    return Pipeline(
        [
            ("prepare", TextFeatureFrame()),
            ("features", ColumnTransformer(transformers)),
            ("classifier", _classifier()),
        ]
    )


def build_models():
    """Return all Phase 2 candidate models with unfitted vectorizers."""
    linguistic = ("linguistic", "passthrough", LINGUISTIC_COLUMNS)
    word = (
        "word_tfidf",
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        "text",
    )
    character = (
        "char_tfidf",
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True),
        "text",
    )

    return {
        "baseline": _pipeline([linguistic]),
        "word_tfidf": _pipeline([word]),
        "char_tfidf": _pipeline([character]),
        "combined": _pipeline([word, character, linguistic]),
    }


def evaluate_predictions(labels, predictions, probabilities):
    """Return the common classification metrics for a fitted model."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def evaluate_model(model, texts, labels):
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)[:, 1]
    return evaluate_predictions(labels, predictions, probabilities)
