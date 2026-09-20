import hashlib
import uuid
from datetime import datetime, timezone

from src.database import insert_analysis, insert_feedback


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
    )
    return analysis_id


def record_feedback(analysis_id, label):
    if label not in VALID_LABELS:
        raise ValueError("Feedback label must be human, ai, or unsure")
    return insert_feedback(analysis_id, label, _now())
