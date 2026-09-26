"""Privacy policy: retention and training consent are independent."""

import sqlite3

import src.database as database
from src.feedback import record_analysis, submit_feedback
from src.privacy import (
    VALID_USER_LABELS,
    can_export_for_training,
    default_store_text,
    should_store_text,
)


def test_retention_flag_is_independent_of_policy_default(monkeypatch):
    monkeypatch.setenv("STORE_TEXT_DEFAULT", "false")

    assert default_store_text() is False
    assert should_store_text() is False
    # An explicit request still wins over the configured default.
    assert should_store_text(True) is True
    assert should_store_text(False) is False


def test_retention_default_is_true_for_the_review_queue(monkeypatch):
    monkeypatch.delenv("STORE_TEXT_DEFAULT", raising=False)

    assert should_store_text() is True
    assert should_store_text(None) is True


def test_export_requires_every_precondition():
    assert can_export_for_training(True, True, "approved", "human") is True
    assert can_export_for_training(True, True, "approved", "ai") is True

    assert can_export_for_training(True, True, "pending", "human") is False
    assert can_export_for_training(False, True, "approved", "human") is False
    assert can_export_for_training(True, False, "approved", "human") is False
    assert can_export_for_training(True, True, "approved", "ai_assisted") is False
    assert can_export_for_training(True, True, "approved", "unsure") is False


def test_feedback_labels_match_the_privacy_policy_set():
    assert VALID_USER_LABELS == {"human", "ai", "ai_assisted", "unsure"}


def test_disabled_retention_never_writes_raw_text(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()

    text = "Это конфиденциальный черновик, который нельзя сохранять на диске."
    analysis_id = record_analysis(text, "human", 12.0, "phase2-v1", store_text=False)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT text_content, text_hash, store_text FROM analysis WHERE id = ?",
            (analysis_id,),
        ).fetchone()

    assert row[0] is None
    assert row[1]
    assert row[2] == 0
    assert text not in database_path.read_text(encoding="utf-8", errors="ignore")

    # Feedback can still be submitted by supplying the text explicitly.
    result = submit_feedback(
        analysis_id, "human", True, "certain", text
    )
    assert result["status"] in {"pending", "duplicate"}
