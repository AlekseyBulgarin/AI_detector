"""Feedback validation.

Validation never destroys records. Hard failures prevent creation and are
returned to the caller as an API error. Anything recoverable is persisted on
the feedback row as ``validation_flags`` plus a ``validation_result`` so the
human reviewer can see exactly why a submission is questionable.
"""

import re
import unicodedata


VALID_LABELS = {"human", "ai", "ai_assisted", "unsure"}
TRAINABLE_LABELS = {"human", "ai"}
VALID_CONFIDENCE = {"certain", "not_sure"}
MAX_TEXT_LENGTH = 30000
MIN_TEXT_LENGTH = 20
NEAR_DUPLICATE_THRESHOLD = 0.85
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REJECT_RESULT = "rejected"
FLAGGED_RESULT = "flagged"
ACCEPTED_RESULT = "accepted"


class FeedbackValidationError(ValueError):
    """Raised when feedback is unsafe or cannot be persisted at all."""


class ValidationOutcome:
    """Result of validating one feedback submission."""

    def __init__(self, flags=None, result=ACCEPTED_RESULT, duplicate_of=None):
        self.flags = list(flags or [])
        self.result = result
        self.duplicate_of = duplicate_of

    @property
    def is_duplicate(self):
        return self.result == "duplicate"

    def as_dict(self):
        return {
            "flags": self.flags,
            "result": self.result,
            "duplicate_of": self.duplicate_of,
        }


def normalize_text(text):
    """Aggressive normalization used only for duplicate comparison."""
    lowered = unicodedata.normalize("NFKC", text or "").lower()
    return " ".join(lowered.split())


def text_statistics(text):
    words = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
    sentences = [part for part in re.split(r"[.!?]+", text or "") if part.strip()]
    characters = len(text or "")
    unique_ratio = (len(set(words)) / len(words)) if words else 0.0
    return {
        "characters": characters,
        "words": len(words),
        "sentences": len(sentences),
        "unique_ratio": round(unique_ratio, 4),
    }


def _trigrams(text):
    compact = re.sub(r"\s+", " ", text).strip().lower()
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[index : index + 3] for index in range(len(compact) - 2)}


def near_duplicate_ratio(left, right):
    """Jaccard similarity over character trigrams."""
    left_set = _trigrams(left)
    right_set = _trigrams(right)
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection / union if union else 0.0


def validate_feedback(text, user_label, allow_training=False, user_confidence=None):
    """Validate a submission before it is persisted.

    Raises :class:`FeedbackValidationError` for conditions that make the record
    impossible to store safely. Returns basic statistics on success.
    """
    if user_label not in VALID_LABELS:
        raise FeedbackValidationError(
            "Feedback label must be human, ai, ai_assisted, or unsure"
        )
    if user_confidence is not None and user_confidence not in VALID_CONFIDENCE:
        raise FeedbackValidationError(
            "user_confidence must be certain or not_sure"
        )
    if not isinstance(allow_training, bool):
        raise FeedbackValidationError("allow_training must be a boolean")
    if not isinstance(text, str) or not text.strip():
        raise FeedbackValidationError("The analyzed text is empty")
    if len(text.strip()) < MIN_TEXT_LENGTH:
        raise FeedbackValidationError("The analyzed text is too short")
    if len(text) > MAX_TEXT_LENGTH:
        raise FeedbackValidationError("The analyzed text is too long")
    if CONTROL_CHARACTER_PATTERN.search(text):
        raise FeedbackValidationError(
            "The analyzed text contains invalid control characters"
        )
    if len(set(text.strip())) == 1:
        raise FeedbackValidationError("The analyzed text is suspiciously repetitive")
    if allow_training and user_label not in TRAINABLE_LABELS:
        raise FeedbackValidationError(
            "Training consent is only meaningful for human or ai labels"
        )
    return text_statistics(text)


def detect_suspicious(text):
    """Return soft flags for content that a human should inspect."""
    flags = []
    stats = text_statistics(text)
    if stats["words"] and stats["unique_ratio"] < 0.2:
        flags.append("low_vocabulary_diversity")
    if stats["characters"] and len(set(text)) / stats["characters"] < 0.15:
        flags.append("repetitive_characters")
    if stats["words"] and stats["words"] < 12:
        flags.append("very_short_submission")
    if stats["sentences"] == 0:
        flags.append("no_sentence_boundaries")
    return flags


def evaluate_submission(
    text,
    user_label,
    existing_entries=None,
    same_analysis_feedback=None,
):
    """Validate content plus duplicate/contradiction context.

    ``existing_entries`` is a list of ``(feedback_id, text)`` pairs used for
    normalized and near-duplicate detection. ``same_analysis_feedback`` is the
    record already stored for this analysis, if any.
    """
    outcome = ValidationOutcome()
    outcome.flags.extend(detect_suspicious(text))
    normalized = normalize_text(text)

    if same_analysis_feedback is not None:
        outcome.flags.append("repeated_submission")
        outcome.result = FLAGGED_RESULT

    normalized_matches = [
        (feedback_id, candidate)
        for feedback_id, candidate in (existing_entries or [])
        if candidate and normalize_text(candidate) == normalized
    ]
    if normalized_matches:
        outcome.flags.append("normalized_duplicate")
        outcome.duplicate_of = normalized_matches[0][0]
        outcome.result = "duplicate"
        return outcome

    best_id = None
    best_ratio = 0.0
    for feedback_id, candidate in existing_entries or []:
        if not candidate:
            continue
        ratio = near_duplicate_ratio(text, candidate)
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = feedback_id
    if best_ratio >= NEAR_DUPLICATE_THRESHOLD:
        outcome.flags.append("near_duplicate")
        outcome.duplicate_of = best_id
        outcome.result = "duplicate"
        return outcome

    if user_label in TRAINABLE_LABELS and any(
        flag in outcome.flags for flag in ("repeated_submission",)
    ):
        outcome.result = FLAGGED_RESULT

    if outcome.result == ACCEPTED_RESULT and outcome.flags:
        outcome.result = FLAGGED_RESULT
    return outcome
