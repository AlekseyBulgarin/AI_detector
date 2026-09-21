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


def test_sqlite_uses_wal_mode_and_hash_lookup_index(tmp_path):
    database_path = tmp_path / "feedback.db"
    database.initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        indexes = connection.execute("PRAGMA index_list('analysis')").fetchall()

    assert journal_mode.lower() == "wal"
    assert any("idx_analysis_text_model" in index[1] for index in indexes)


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


def test_legacy_feedback_schema_is_migrated_without_losing_review_data(tmp_path):
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE analysis (
                id TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );
            INSERT INTO analysis VALUES ('a1', 'hash', 'ai', 88.0, 'phase2-v1', '2026-09-21');
            INSERT INTO feedback VALUES (1, 'a1', 'ai', '2026-09-21', 'pending');
            """
        )

    database.initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(feedback)")}
        row = connection.execute(
            "SELECT prediction_label, prediction_probability, user_label, status FROM feedback"
        ).fetchone()

    assert {"text_hash", "prediction_label", "user_label", "allow_training", "reviewed_at"}.issubset(columns)
    assert row == ("ai", 88.0, "ai", "pending")
