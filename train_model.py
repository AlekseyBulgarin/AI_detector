import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features import extract_features


RANDOM_STATE = 42


def load_texts(path, label, features, labels, seen_texts):
    if not os.path.exists(path):
        print(f"⚠️ Папка {path} не найдена")
        return

    filenames = [filename for filename in os.listdir(path) if filename.endswith(".txt")]
    for filename in filenames:
        file_path = os.path.join(path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            if text in seen_texts:
                print(f"⚠️ Пропущен дубликат: {file_path}")
                continue
            seen_texts.add(text)
            features.append(extract_features(text))
            labels.append(label)

    print(f"✅ Обработано {len(filenames)} текстов из {path}")


def evaluate_model(model, dataset_name, features, labels):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "confusion_matrix": matrix,
    }

    print(f"\n📊 Метрики ({dataset_name})")
    print(f"  Accuracy:  {metrics['accuracy'] * 100:.1f}%")
    print(f"  Precision: {metrics['precision'] * 100:.1f}%")
    print(f"  Recall:    {metrics['recall'] * 100:.1f}%")
    print(f"  F1-score:  {metrics['f1'] * 100:.1f}%")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
    print("  Confusion matrix [human, AI]:")
    print(matrix)
    return metrics


X, y = [], []
seen_texts = set()
load_texts("data/raw/human", 0, X, y, seen_texts)
load_texts("data/raw/ai", 1, X, y, seen_texts)

if not X:
    print("❌ Нет данных для обучения! Положи тексты в папки data/raw/human/ и data/raw/ai/")
    raise SystemExit(1)

X = np.array(X)
y = np.array(y)

if len(np.unique(y)) < 2:
    print("❌ Для обучения нужны примеры обоих классов")
    raise SystemExit(1)

# Keep the test set untouched until the final evaluation.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=RANDOM_STATE,
    stratify=y,
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

model = LogisticRegression(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

evaluate_model(model, "validation", X_validation, y_validation)
evaluate_model(model, "test", X_test, y_test)
print(
    f"\nℹ️ Размеры выборок: train={len(y_train)}, "
    f"validation={len(y_validation)}, test={len(y_test)}"
)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("✅ Модель сохранена в models/model.pkl")
