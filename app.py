from flask import Flask, render_template, request, jsonify
import os
import joblib
import json
import logging
import sklearn

from src.config import FLASK_ENV, MODEL_METADATA_PATH, MODEL_PATH
from src.database import AnalysisNotFoundError, initialize_database
from src.feedback import record_analysis, record_feedback
from src.features import extract_features


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
initialize_database()


model = None
model_version = "legacy"
model_load_error = None

if MODEL_METADATA_PATH.exists():
    try:
        with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
            model_metadata = json.load(file)
            model_version = model_metadata.get("model_version", model_version)
            expected_sklearn = model_metadata.get("sklearn_version")
            if expected_sklearn and expected_sklearn != sklearn.__version__:
                logger.warning(
                    "Model was trained with scikit-learn %s; runtime has %s",
                    expected_sklearn,
                    sklearn.__version__,
                )
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read model metadata: %s", exc)

if MODEL_PATH.exists():
    logger.info("event=model_loading path=%s", MODEL_PATH)
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("event=model_loaded version=%s", model_version)
    except Exception as exc:
        model_load_error = str(exc)
        logger.exception("Model could not be loaded")
else:
    model_load_error = "Model artifact is missing"
    logger.error("event=model_missing path=%s", MODEL_PATH)

logger.info(
    "event=application_started environment=%s model_available=%s",
    FLASK_ENV,
    model is not None,
)


def predict_probability(text):
    """Возвращает вероятность того, что текст написан ИИ (0-100)"""
    if model is None:
        raise RuntimeError(model_load_error or "Model is unavailable")

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


@app.route("/health")
def health():
    if model is None:
        return jsonify({"status": "unavailable", "model_loaded": False}), 503
    return jsonify(
        {
            "status": "ok",
            "model_loaded": True,
            "model_version": model_version,
        }
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

    try:
        probability = predict_probability(text)
    except Exception:
        logger.exception("Prediction failed for form request")
        return render_template(
            "index.html",
            user_text=text,
            probability=None,
            analysis_id=None,
            error="Модель временно недоступна. Попробуйте позже.",
        ), 503
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
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str):
        return jsonify({"error": "Поле text должно быть строкой", "probability": None}), 400

    if len(text.strip()) < 20:
        return jsonify({"error": "Текст слишком короткий", "probability": None}), 400

    try:
        probability = predict_probability(text)
    except Exception:
        logger.exception("Prediction failed for API request")
        return jsonify({"error": "Модель временно недоступна", "probability": None}), 503
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
    except AnalysisNotFoundError:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify({"id": feedback_id, "status": "pending"}), 201


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=FLASK_ENV == "development",
    )
