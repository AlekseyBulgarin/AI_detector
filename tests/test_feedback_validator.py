import pytest

from src.feedback_validator import FeedbackValidationError, validate_feedback


VALID_TEXT = "Это достаточно длинный текст для проверки качества обратной связи."


def test_validator_accepts_valid_label_and_consent():
    assert validate_feedback(VALID_TEXT, "human", True) is True


@pytest.mark.parametrize(
    "text,label,allow_training,message",
    [
        ("", "human", False, "empty"),
        ("коротко", "human", False, "short"),
        (VALID_TEXT, "unknown", False, "human, ai, or unsure"),
        ("x" * 40, "ai", False, "repetitive"),
        (VALID_TEXT, "ai", "yes", "boolean"),
    ],
)
def test_validator_rejects_invalid_or_suspicious_feedback(text, label, allow_training, message):
    with pytest.raises(FeedbackValidationError, match=message):
        validate_feedback(text, label, allow_training)
