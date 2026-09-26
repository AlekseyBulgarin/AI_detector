import hashlib
import uuid
from datetime import datetime, timezone

from src.database import (
    AnalysisNotFoundError,
    find_feedback_by_hash,
    find_feedback_for_analysis_exact,
    get_analysis,
    insert_analysis,
    insert_feedback,
    list_recent_feedback_texts,
)
from src.feedback_validator import (
    ACCEPTED_RESULT,
    FLAGGED_RESULT,
    FeedbackValidationError,
    evaluate_submission,
    normalize_text,
    validate_feedback,
)
from src.priority import score_feedback
from src.versions import DEFAULT_DATASET_VERSION, FEATURE_VERSION


VALID_LABELS = {"human", "ai", "ai_assisted", "unsure"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_analysis(
    text,
    prediction,
    probability,
    model_version,
    dataset_version=DEFAULT_DATASET_VERSION,
    feature_version=FEATURE_VERSION,
    latency_ms=None,
    store_text=True,
):
    """Persist one prediction event.

    When ``store_text`` is False only the SHA-256 hash, metadata and the
    prediction are retained; the raw text is never written to disk.
    """
    analysis_id = str(uuid.uuid4())
    text_length = len(text)
    word_count = len(text.split())
    insert_analysis(
        analysis_id,
        _hash(text),
        prediction,
        probability,
        model_version,
        _now(),
        text if store_text else None,
        normalized_hash=_hash(normalize_text(text)),
        text_length=text_length,
        word_count=word_count,
        dataset_version=dataset_version,
        feature_version=feature_version,
        latency_ms=latency_ms,
        store_text=store_text,
    )
    return analysis_id


def _existing_entries(exclude_analysis_id, limit=500):
    entries = []
    for item in list_recent_feedback_texts(limit):
        if item.get("analysis_id") == exclude_analysis_id:
            continue
        entries.append((item["id"], item["text_content"]))
    return entries


def submit_feedback(
    analysis_id,
    label,
    allow_training=False,
    user_confidence=None,
    text=None,
    extra_flags=None,
):
    """Record human feedback without ever touching the production model.

    Feedback is always stored separately from the model. It can only become
    training data after a human approves it and training consent was given.
    """
    if label not in VALID_LABELS:
        raise FeedbackValidationError(
            "Feedback label must be human, ai, ai_assisted, or unsure"
        )
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise AnalysisNotFoundError(analysis_id)

    retained = analysis.get("text_content")
    effective_text = retained if retained else text

    if effective_text:
        validate_feedback(effective_text, label, allow_training, user_confidence)
    else:
        # Metadata can still be validated when raw text was not retained.
        if not isinstance(allow_training, bool):
            raise FeedbackValidationError("allow_training must be a boolean")
        if user_confidence is not None and user_confidence not in {
            "certain",
            "not_sure",
        }:
            raise FeedbackValidationError(
                "user_confidence must be certain or not_sure"
            )
        if allow_training and label not in {"human", "ai"}:
            raise FeedbackValidationError(
                "Training consent is only meaningful for human or ai labels"
            )

    existing = find_feedback_for_analysis_exact(analysis_id)
    if existing is not None:
        return {
            "success": False,
            "feedback_id": existing["id"],
            "id": existing["id"],
            "status": "duplicate",
        }

    text_hash = analysis["text_hash"]
    flags = []
    status = "pending"
    duplicate_of = None
    validation_result = ACCEPTED_RESULT

    if effective_text:
        outcome = evaluate_submission(
            effective_text,
            label,
            existing_entries=_existing_entries(analysis_id),
        )
        flags = outcome.flags
        duplicate_of = outcome.duplicate_of
        validation_result = outcome.result
    else:
        flags = ["text_not_retained"]
        validation_result = FLAGGED_RESULT

    for flag in extra_flags or []:
        if flag not in flags:
            flags.append(flag)

    if duplicate_of is None:
        hash_match = find_feedback_by_hash(text_hash, exclude_analysis_id=analysis_id)
        if hash_match is not None:
            duplicate_of = hash_match["id"]
            if "exact_duplicate" not in flags:
                flags.append("exact_duplicate")
            validation_result = "duplicate"

    if validation_result == "duplicate":
        status = "duplicate"

    mining = score_feedback(
        analysis["prediction"],
        analysis["probability"],
        label,
        user_confidence,
    )

    feedback_id = insert_feedback(
        analysis_id=analysis_id,
        text_hash=text_hash,
        text_content=effective_text,
        prediction_label=analysis["prediction"],
        prediction_probability=analysis["probability"],
        model_version=analysis["model_version"],
        user_label=label,
        allow_training=allow_training,
        created_at=_now(),
        normalized_hash=_hash(normalize_text(effective_text or "")),
        user_confidence=user_confidence,
        validation_flags=flags,
        validation_result=validation_result,
        priority_category=mining["priority_category"],
        hard_case_categories=mining["categories"],
        priority_score=mining["priority_score"],
        status=status,
        duplicate_of=duplicate_of,
    )
    return {
        "success": status != "duplicate",
        "feedback_id": feedback_id,
        "id": feedback_id,
        "status": status,
        "priority_category": mining["priority_category"],
        "priority_score": mining["priority_score"],
        "categories": mining["categories"],
    }


def record_feedback(analysis_id, label):
    """Compatibility helper returning the numeric feedback id."""
    return submit_feedback(analysis_id, label)["feedback_id"]
