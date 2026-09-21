import re


VALID_LABELS = {"human", "ai", "unsure"}
MAX_TEXT_LENGTH = 30000
MIN_TEXT_LENGTH = 20
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class FeedbackValidationError(ValueError):
    """Raised when feedback is unsafe or cannot be used for review."""


def validate_feedback(text, user_label, allow_training=False):
    """Validate feedback before it is persisted or considered for training."""
    if user_label not in VALID_LABELS:
        raise FeedbackValidationError("Feedback label must be human, ai, or unsure")
    if not isinstance(text, str) or not text.strip():
        raise FeedbackValidationError("The analyzed text is empty")
    if len(text.strip()) < MIN_TEXT_LENGTH:
        raise FeedbackValidationError("The analyzed text is too short")
    if len(text) > MAX_TEXT_LENGTH:
        raise FeedbackValidationError("The analyzed text is too long")
    if CONTROL_CHARACTER_PATTERN.search(text):
        raise FeedbackValidationError("The analyzed text contains invalid control characters")
    if len(set(text.strip())) == 1:
        raise FeedbackValidationError("The analyzed text is suspiciously repetitive")
    if not isinstance(allow_training, bool):
        raise FeedbackValidationError("allow_training must be a boolean")
    return True
