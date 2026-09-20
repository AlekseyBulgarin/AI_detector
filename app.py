from flask import Flask, render_template, request, jsonify
import os
import joblib
import json
from pathlib import Path

from src.database import initialize_database
from src.feedback import record_analysis, record_feedback
from src.features import extract_features


app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
initialize_database()


MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
model = None
model_version = "legacy"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded")
else:
    print("Model not found. Run train_model.py first.")

if MODEL_METADATA_PATH.exists():
    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
        model_version = json.load(file).get("model_version", model_version)


def predict_probability(text):
    """Возвращает вероятность того, что текст написан ИИ (0-100)"""
    if model is None:
        return 50

    if hasattr(model, "named_steps") and "prepare" in model.named_steps:
        prob = model.predict_proba([text])[0][1]
    else:
        features = extract_features(text)
        prob = model.predict_proba([features])[0][1]
    return round(prob * 100, 1)


def prediction_label(probability):
    return "ai" if probability >= 50 else "human"


@app.route("/")
def index():
    return render_template(
        "index.html",
        user_text=None,
        probability=None,
        analysis_id=None,
        error=None,
    )


@app.route("/check", methods=["POST"])
def check():
    text = request.form.get("user_text", "")

    if len(text.strip()) < 20:
        return render_template(
            "index.html",
            user_text=text,
            probability=None,
            analysis_id=None,
            error="❌ Текст слишком короткий (минимум 20 символов)",
        )

    probability = predict_probability(text)
    analysis_id = record_analysis(
        text,
        prediction_label(probability),
        probability,
        model_version,
    )

    return render_template(
        "index.html",
        user_text=text,
        probability=probability,
        analysis_id=analysis_id,
        error=None,
    )


@app.route("/api/check", methods=["POST"])
def api_check():
    """API endpoint для проверки через JSON (для расширения)"""
    data = request.get_json()
    text = data.get("text", "")

    if len(text.strip()) < 20:
        return jsonify({"error": "Текст слишком короткий", "probability": None})

    probability = predict_probability(text)
    analysis_id = record_analysis(
        text,
        prediction_label(probability),
        probability,
        model_version,
    )
    return jsonify({"probability": probability, "analysis_id": analysis_id})


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(silent=True) or {}
    analysis_id = data.get("analysis_id")
    label = data.get("label")
    if not analysis_id or label not in {"human", "ai", "unsure"}:
        return jsonify({"error": "analysis_id and a valid label are required"}), 400

    try:
        feedback_id = record_feedback(analysis_id, label)
    except Exception as exc:
        if "FOREIGN KEY" in str(exc):
            return jsonify({"error": "Analysis not found"}), 404
        raise
    return jsonify({"id": feedback_id, "status": "pending"}), 201


if __name__ == "__main__":
    app.run(debug=True)
