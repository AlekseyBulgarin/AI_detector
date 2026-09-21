import sqlite3

import pytest

import src.database as database
from src.database import AnalysisNotFoundError
from src.feedback import record_analysis, record_feedback


def test_analysis_and_feedback_are_persisted_with_pending_status(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()

    analysis_id = record_analysis("Текст для проверки сохранения.", "human", 12.5, "test-v1")
    feedback_id = record_feedback(analysis_id, "human")

    with sqlite3.connect(database_path) as connection:
        analysis = connection.execute(
            "SELECT prediction, probability, model_version FROM analysis WHERE id = ?",
            (analysis_id,),
        ).fetchone()
        feedback = connection.execute(
            "SELECT analysis_id, label, status FROM feedback WHERE id = ?",
            (feedback_id,),
        ).fetchone()

    assert analysis == ("human", 12.5, "test-v1")
    assert feedback == (analysis_id, "human", "pending")


def test_feedback_rejects_unknown_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "feedback.db"))
    database.initialize_database()

    with pytest.raises(AnalysisNotFoundError):
        record_feedback("unknown-analysis", "ai")


def test_feedback_rejects_invalid_label(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", str(tmp_path / "feedback.db"))
    database.initialize_database()

    with pytest.raises(ValueError, match="human, ai, or unsure"):
        record_feedback("analysis-id", "maybe")
