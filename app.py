from collections import OrderedDict
from hashlib import sha256
from time import perf_counter

from flask import Flask, g, jsonify, redirect, render_template, request, url_for
import os
import joblib
import json
import logging
import sklearn
import hmac
from datetime import datetime, timezone

from src.config import FLASK_ENV, MODEL_METADATA_PATH, MODEL_PATH
from src.database import (
    AnalysisNotFoundError,
    find_analysis_by_hash,
    initialize_database,
    list_feedback,
    update_feedback_status,
)
from src.feedback import record_analysis, submit_feedback
from src.feedback_validator import FeedbackValidationError
from src.features import extract_features


app = Flask(__name__)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
initialize_database()

ANALYSIS_CACHE_SIZE = 256
analysis_cache = OrderedDict()


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


@app.before_request
def start_request_timer():
    g.request_started_at = perf_counter()


@app.after_request
def log_request_timing(response):
    started_at = getattr(g, "request_started_at", None)
    if started_at is not None:
        duration_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "event=request_completed method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )
    return response


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


def _analysis_cache_key(text):
    text_hash = sha256(text.encode("utf-8")).hexdigest()
    return model_version, text_hash


def _remember_analysis(cache_key, analysis):
    analysis_cache[cache_key] = analysis
    analysis_cache.move_to_end(cache_key)
    if len(analysis_cache) > ANALYSIS_CACHE_SIZE:
        analysis_cache.popitem(last=False)


def analyze_text(text):
    """Reuse recent or persisted results before running model inference."""
    cache_key = _analysis_cache_key(text)
    cached = analysis_cache.get(cache_key)
    if cached is not None:
        analysis_cache.move_to_end(cache_key)
        return cached

    _, text_hash = cache_key
    persisted = find_analysis_by_hash(text_hash, model_version)
    if persisted is not None and persisted.get("text_content"):
        _remember_analysis(cache_key, persisted)
        return persisted

    probability = predict_probability(text)
    analysis = {
        "id": record_analysis(
            text,
            prediction_label(probability),
            probability,
            model_version,
        ),
        "prediction": prediction_label(probability),
        "probability": probability,
        "model_version": model_version,
    }
    _remember_analysis(cache_key, analysis)
    return analysis


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
        analysis = analyze_text(text)
    except Exception:
        logger.exception("Prediction failed for form request")
        return render_template(
            "index.html",
            user_text=text,
            probability=None,
            analysis_id=None,
            error="Модель временно недоступна. Попробуйте позже.",
        ), 503
    return render_template(
        "index.html",
        user_text=text,
        probability=analysis["probability"],
        analysis_id=analysis["id"],
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
        analysis = analyze_text(text)
    except Exception:
        logger.exception("Prediction failed for API request")
        return jsonify({"error": "Модель временно недоступна", "probability": None}), 503
    return jsonify({"probability": analysis["probability"], "analysis_id": analysis["id"]})


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(silent=True) or {}
    analysis_id = data.get("analysis_id")
    label = data.get("label")
    allow_training = data.get("allow_training", False)
    if not analysis_id or label not in {"human", "ai", "unsure"}:
        return jsonify({"error": "analysis_id and a valid label are required"}), 400
    if not isinstance(allow_training, bool):
        return jsonify({"error": "allow_training must be a boolean"}), 400

    try:
        result = submit_feedback(analysis_id, label, allow_training)
    except AnalysisNotFoundError:
        return jsonify({"error": "Analysis not found"}), 404
    except FeedbackValidationError as exc:
        return jsonify({"success": False, "error": str(exc), "status": "invalid"}), 400
    return jsonify(result), 201 if result["status"] == "pending" else 200


def _admin_authorized():
    if not ADMIN_TOKEN:
        return True
    supplied = request.headers.get("X-Admin-Token") or request.cookies.get("admin_token", "")
    return hmac.compare_digest(supplied, ADMIN_TOKEN)


@app.route("/admin/feedback", methods=["GET", "POST"])
def admin_feedback():
    if request.method == "POST" and ADMIN_TOKEN:
        supplied = request.form.get("token", "")
        if hmac.compare_digest(supplied, ADMIN_TOKEN):
            response = render_template("admin_feedback.html", feedback=list_feedback("pending"))
            response = app.make_response(response)
            response.set_cookie("admin_token", ADMIN_TOKEN, httponly=True, samesite="Lax")
            return response
        return render_template("admin_feedback.html", feedback=None, login_error=True), 401
    if not _admin_authorized():
        return render_template("admin_feedback.html", feedback=None, login_required=True), 401
    return render_template("admin_feedback.html", feedback=list_feedback("pending"))


@app.route("/admin/feedback/<int:feedback_id>", methods=["POST"])
def admin_review_feedback(feedback_id):
    if not _admin_authorized():
        return jsonify({"error": "Admin authorization required"}), 401
    status = request.form.get("status") or (request.get_json(silent=True) or {}).get("status")
    try:
        update_feedback_status(feedback_id, status, datetime.now(timezone.utc).isoformat())
    except ValueError:
        return jsonify({"error": "Invalid review status"}), 400
    except LookupError:
        return jsonify({"error": "Feedback not found"}), 404
    if request.is_json:
        return jsonify({"success": True, "feedback_id": feedback_id, "status": status})
    return redirect(url_for("admin_feedback"))


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=FLASK_ENV == "development",
    )
