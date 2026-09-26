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

from src.config import FLASK_ENV, MODEL_METADATA_PATH, MODEL_PATH, STATIC_CACHE_SECONDS
from src.database import (
    AnalysisNotFoundError,
    find_analysis_by_hash,
    initialize_database,
    list_feedback,
    update_feedback_status,
)
from src.decision import classify as decide
from src.feedback import record_analysis, submit_feedback
from src.feedback_validator import FeedbackValidationError
from src.features import extract_features
from src.monitoring import compute_production_stats
from src.privacy import VALID_CONFIDENCE, VALID_USER_LABELS, should_store_text
from src.versions import DEFAULT_DATASET_VERSION, FEATURE_VERSION


app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = STATIC_CACHE_SECONDS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
initialize_database()

ANALYSIS_CACHE_SIZE = 256
analysis_cache = OrderedDict()

# Stage timings of the most recent prediction, read right after predict_probability.
_LAST_TIMING = {}


def _admin_token():
    return os.environ.get("ADMIN_TOKEN", "").strip()


model = None
model_version = "legacy"
model_load_error = None
model_load_ms = 0.0

model_load_started = perf_counter()
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
        model_load_ms = round((perf_counter() - model_load_started) * 1000, 3)
        logger.info("event=model_loaded version=%s", model_version)
    except Exception as exc:
        model_load_error = str(exc)
        logger.exception("Model could not be loaded")
else:
    model_load_error = "Model artifact is missing"
    logger.error("event=model_missing path=%s", MODEL_PATH)

logger.info(
    "event=application_started environment=%s model_available=%s load_ms=%.3f",
    FLASK_ENV,
    model is not None,
    model_load_ms,
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


def _elapsed_ms(started):
    return round((perf_counter() - started) * 1000, 3)


def predict_probability(text):
    """Возвращает вероятность того, что текст написан ИИ (0-100)"""
    if model is None:
        raise RuntimeError(model_load_error or "Model is unavailable")

    _LAST_TIMING.clear()
    if hasattr(model, "named_steps") and "prepare" in model.named_steps:
        started = perf_counter()
        frame = model.named_steps["prepare"].transform([text])
        matrix = model.named_steps["features"].transform(frame)
        feature_ms = _elapsed_ms(started)
        started = perf_counter()
        prob = model.named_steps["classifier"].predict_proba(matrix)[0][1]
        prediction_ms = _elapsed_ms(started)
    else:
        started = perf_counter()
        features = extract_features(text)
        feature_ms = _elapsed_ms(started)
        started = perf_counter()
        prob = model.predict_proba([features])[0][1]
        prediction_ms = _elapsed_ms(started)

    _LAST_TIMING["feature_ms"] = feature_ms
    _LAST_TIMING["prediction_ms"] = prediction_ms
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


def _payload_from_row(row):
    probability = round(float(row["probability"]), 1)
    return {
        "id": row["id"],
        "analysis_id": row["id"],
        "prediction": row["prediction"],
        "probability": probability,
        "model_version": row["model_version"],
        "dataset_version": row.get("dataset_version") or DEFAULT_DATASET_VERSION,
        "feature_version": row.get("feature_version") or FEATURE_VERSION,
        "decision": decide(probability),
        "reused": True,
    }


def analyze_text(text, store_text=None):
    """Reuse recent or persisted results before running model inference."""
    effective_store = should_store_text(store_text)
    cache_key = _analysis_cache_key(text)
    cached = analysis_cache.get(cache_key)
    if cached is not None:
        analysis_cache.move_to_end(cache_key)
        return dict(cached)

    _, text_hash = cache_key
    persisted = find_analysis_by_hash(text_hash, model_version)
    if persisted is not None:
        reused = _payload_from_row(persisted)
        _remember_analysis(cache_key, reused)
        return dict(reused)

    _LAST_TIMING.clear()
    prediction_started = perf_counter()
    probability = predict_probability(text)
    model_stage_ms = _LAST_TIMING.copy() or {
        "feature_ms": None,
        "prediction_ms": None,
    }
    model_stage_ms["model_total_ms"] = _elapsed_ms(prediction_started)

    db_started = perf_counter()
    analysis_id = record_analysis(
        text,
        prediction_label(probability),
        probability,
        model_version,
        DEFAULT_DATASET_VERSION,
        FEATURE_VERSION,
        model_stage_ms["model_total_ms"],
        effective_store,
    )
    db_write_ms = _elapsed_ms(db_started)

    analysis = {
        "id": analysis_id,
        "analysis_id": analysis_id,
        "prediction": prediction_label(probability),
        "probability": probability,
        "model_version": model_version,
        "dataset_version": DEFAULT_DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "decision": decide(probability),
        "store_text": effective_store,
        "reused": False,
        "timing": {
            **model_stage_ms,
            "db_write_ms": db_write_ms,
        },
    }
    _remember_analysis(cache_key, analysis)
    return dict(analysis)


def _with_request_latency(payload):
    started_at = getattr(g, "request_started_at", None)
    if started_at is not None:
        payload["latency_ms"] = _elapsed_ms(started_at)
    return payload


def _index_context(**overrides):
    context = {
        "user_text": None,
        "probability": None,
        "analysis": None,
        "decision": None,
        "timing": None,
        "error": None,
        "store_text_default": should_store_text(),
        "model_version": model_version,
        "dataset_version": DEFAULT_DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "static_cache_seconds": STATIC_CACHE_SECONDS,
    }
    context.update(overrides)
    if not context.get("analysis_id") and isinstance(context.get("analysis"), dict):
        context["analysis_id"] = context["analysis"].get("analysis_id")
    return context


@app.route("/")
def index():
    return render_template("index.html", **_index_context())


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


@app.route("/api/metrics")
def api_metrics():
    payload = compute_production_stats()
    payload["model"] = {
        "version": model_version,
        "loaded": model is not None,
        "load_ms": model_load_ms,
        "feature_version": FEATURE_VERSION,
        "error": model_load_error,
    }
    return jsonify(_with_request_latency(payload))


@app.route("/check", methods=["POST"])
def check():
    text = request.form.get("user_text", "")
    store_text = request.form.get("store_text") == "1"

    if len(text.strip()) < 20:
        return render_template(
            "index.html",
            **_index_context(
                user_text=text,
                error="❌ Текст слишком короткий (минимум 20 символов)",
                store_text_default=should_store_text(store_text),
            ),
        )

    try:
        analysis = analyze_text(text, store_text=store_text)
    except Exception:
        logger.exception("Prediction failed for form request")
        return render_template(
            "index.html",
            **_index_context(
                user_text=text,
                error="Модель временно недоступна. Попробуйте позже.",
                store_text_default=should_store_text(store_text),
            ),
        ), 503

    timing = dict(analysis.get("timing") or {})
    started_at = getattr(g, "request_started_at", None)
    if started_at is not None:
        timing["total_ms"] = _elapsed_ms(started_at)
    return render_template(
        "index.html",
        **_index_context(
            user_text=text,
            probability=analysis["probability"],
            analysis=analysis,
            decision=analysis["decision"],
            timing=timing,
            store_text_default=analysis.get("store_text", should_store_text()),
        ),
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

    store_text = data.get("store_text")
    if store_text is not None and not isinstance(store_text, bool):
        return jsonify({"error": "store_text must be a boolean", "probability": None}), 400

    try:
        analysis = analyze_text(text, store_text=store_text)
    except Exception:
        logger.exception("Prediction failed for API request")
        return jsonify({"error": "Модель временно недоступна", "probability": None}), 503

    payload = {
        "probability": analysis["probability"],
        "analysis_id": analysis["id"],
        "prediction": analysis["prediction"],
        "model_version": analysis["model_version"],
        "dataset_version": analysis["dataset_version"],
        "feature_version": analysis["feature_version"],
        "decision": analysis["decision"],
        "reused": analysis["reused"],
        "store_text": analysis.get("store_text", should_store_text(store_text)),
        "timing": analysis.get("timing"),
    }
    return jsonify(_with_request_latency(payload))


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json(silent=True) or {}
    analysis_id = data.get("analysis_id")
    label = data.get("label")

    if not analysis_id or label not in VALID_USER_LABELS:
        return jsonify({"error": "analysis_id and a valid label are required"}), 400

    allow_training = data.get("allow_training", data.get("consent_for_training", False))
    if not isinstance(allow_training, bool):
        return jsonify({"error": "allow_training must be a boolean"}), 400

    confidence = data.get("user_confidence", data.get("confidence"))
    if confidence is not None and confidence not in VALID_CONFIDENCE:
        return jsonify({"error": "user_confidence must be certain or not_sure"}), 400

    text = data.get("text")
    if text is not None and not isinstance(text, str):
        return jsonify({"error": "text must be a string"}), 400

    extra_flags = []
    if data.get("prediction_incorrect") is True:
        extra_flags.append("prediction_incorrect")

    try:
        result = submit_feedback(
            analysis_id, label, allow_training, confidence, text, extra_flags
        )
    except AnalysisNotFoundError:
        return jsonify({"error": "Analysis not found"}), 404
    except FeedbackValidationError as exc:
        return jsonify({"success": False, "error": str(exc), "status": "invalid"}), 400
    return jsonify(result), 201 if result["status"] == "pending" else 200


ADMIN_FILTERS = {
    "all": {},
    "pending": {"status": "pending"},
    "needs_review": {"status": "needs_review"},
    "approved": {"status": "approved"},
    "rejected": {"status": "rejected"},
    "high_disagreement": {"category": "high_disagreement"},
    "false_positives": {"category": "false_positive_candidate"},
    "false_negatives": {"category": "false_negative_candidate"},
    "ai_assisted": {"category": "ai_assisted"},
}


def _admin_authorized():
    token = _admin_token()
    if not token:
        return False
    supplied = (
        request.headers.get("X-Admin-Token")
        or request.cookies.get("admin_token", "")
        or ""
    )
    return hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))


@app.route("/admin/login", methods=["POST"])
def admin_login():
    token = _admin_token()
    supplied = request.form.get("token", "")
    if not token or not hmac.compare_digest(
        supplied.encode("utf-8"), token.encode("utf-8")
    ):
        return render_template(
            "admin_feedback.html", feedback=None, login_error=True
        ), 401
    response = app.make_response(redirect(url_for("admin_feedback")))
    response.set_cookie("admin_token", token, httponly=True, samesite="Lax")
    return response


@app.route("/admin/feedback", methods=["GET"])
def admin_feedback():
    token = _admin_token()
    if not token:
        return render_template(
            "admin_feedback.html", feedback=None, disabled=True
        ), 403
    if not _admin_authorized():
        return render_template(
            "admin_feedback.html", feedback=None, login_required=True
        ), 401

    active_filter = request.args.get("filter", "pending")
    if active_filter not in ADMIN_FILTERS:
        active_filter = "pending"
    rows = list_feedback(**ADMIN_FILTERS[active_filter])
    return render_template(
        "admin_feedback.html",
        feedback=rows,
        active_filter=active_filter,
        filters=list(ADMIN_FILTERS),
        stats=compute_production_stats(),
    )


@app.route("/admin/feedback/<int:feedback_id>", methods=["POST"])
def admin_review_feedback(feedback_id):
    if not _admin_token():
        return jsonify({"error": "Admin access is not configured"}), 403
    if not _admin_authorized():
        return jsonify({"error": "Admin authorization required"}), 401

    payload = request.get_json(silent=True) or {}
    status = request.form.get("status") or payload.get("status")
    note = request.form.get("note") or payload.get("note")
    try:
        update_feedback_status(
            feedback_id,
            status,
            datetime.now(timezone.utc).isoformat(),
            reviewer_note=note,
        )
    except ValueError:
        return jsonify({"error": "Invalid review status"}), 400
    except LookupError:
        return jsonify({"error": "Feedback not found"}), 404
    if request.is_json:
        return jsonify({"success": True, "feedback_id": feedback_id, "status": status})
    return redirect(url_for("admin_feedback", filter=request.form.get("filter", "pending")))


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=FLASK_ENV == "development",
    )
