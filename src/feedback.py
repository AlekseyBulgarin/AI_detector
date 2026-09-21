import hashlib
import uuid
from datetime import datetime, timezone

from src.database import (
    AnalysisNotFoundError,
    find_feedback_for_analysis,
    get_analysis,
    insert_analysis,
    insert_feedback,
)
from src.feedback_validator import FeedbackValidationError, validate_feedback


VALID_LABELS = {"human", "ai", "unsure"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def record_analysis(text, prediction, probability, model_version):
    analysis_id = str(uuid.uuid4())
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    insert_analysis(
        analysis_id,
        text_hash,
        prediction,
        probability,
        model_version,
        _now(),
        text,
    )
    return analysis_id


def submit_feedback(analysis_id, label, allow_training=False):
    if label not in VALID_LABELS:
        raise FeedbackValidationError("Feedback label must be human, ai, or unsure")
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise AnalysisNotFoundError(analysis_id)
    validate_feedback(analysis.get("text_content"), label, allow_training)
    existing = find_feedback_for_analysis(analysis_id)
    if existing is not None:
        return {
            "success": False,
            "feedback_id": existing["id"],
            "id": existing["id"],
            "status": "duplicate",
        }
    feedback_id = insert_feedback(
        analysis_id=analysis_id,
        text_hash=analysis["text_hash"],
        text_content=analysis["text_content"],
        prediction_label=analysis["prediction"],
        prediction_probability=analysis["probability"],
        model_version=analysis["model_version"],
        user_label=label,
        allow_training=allow_training,
        created_at=_now(),
    )
    return {"success": True, "feedback_id": feedback_id, "id": feedback_id, "status": "pending"}


def record_feedback(analysis_id, label):
    """Compatibility helper returning the numeric feedback id."""
    return submit_feedback(analysis_id, label)["feedback_id"]
