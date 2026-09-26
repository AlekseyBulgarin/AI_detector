"""Central privacy and consent policy for text retention.

Two independent controls exist and must never be conflated:

``store_text``
    Whether the raw analyzed text is retained with the analysis event. This is
    an operational retention decision made at prediction time.

``consent_for_training``
    Whether an explicitly reviewed feedback sample may later be exported into a
    training dataset. This is a separate, affirmative user decision made after
    prediction.

Feedback about a prediction is always allowed. Permission to reuse the raw text
for training is not implied by submitting feedback.
"""

import os

VALID_USER_LABELS = {"human", "ai", "ai_assisted", "unsure"}
TRAINABLE_LABELS = {"human", "ai"}
VALID_CONFIDENCE = {"certain", "not_sure"}


def _env_bool(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def default_store_text():
    """Project privacy policy: raw analysis text is retained by default.

    Retention is required for the human review queue and for duplicate checks.
    It never implies training consent. Set ``STORE_TEXT_DEFAULT=false`` to stop
    retaining raw analysis text; feedback can still carry its own text when the
    user grants training consent.
    """
    return _env_bool("STORE_TEXT_DEFAULT", True)


def should_store_text(requested=None):
    """Resolve the effective ``store_text`` flag for a prediction request."""
    if isinstance(requested, bool):
        return requested
    return default_store_text()


def can_export_for_training(allow_training, text_available, status, label):
    """Return True only when every privacy and quality precondition holds."""
    if status != "approved":
        return False
    if not allow_training:
        return False
    if not text_available:
        return False
    if label not in TRAINABLE_LABELS:
        return False
    return True
