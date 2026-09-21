import csv
import sqlite3

import src.database as database
from src.feedback import record_analysis, submit_feedback
from tools.export_feedback_dataset import export_feedback_dataset


VALID_TEXT = "Это достаточно длинный текст для проверки экспорта обратной связи."


def test_export_includes_only_approved_consented_feedback(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()

    approved_analysis = record_analysis(VALID_TEXT, "human", 22.0, "phase2-v1")
    pending_analysis = record_analysis(
        "Это другой достаточно длинный текст для проверки pending статуса.",
        "ai",
        78.0,
        "phase2-v1",
    )
    approved = submit_feedback(approved_analysis, "human", True)
    submit_feedback(pending_analysis, "ai", True)
    database.update_feedback_status(
        approved["feedback_id"], "approved", "2026-09-21T00:00:00+00:00"
    )

    output_root = tmp_path / "feedback"
    metadata_path, rows = export_feedback_dataset(output_root, str(database_path))

    assert len(rows) == 1
    assert metadata_path.exists()
    assert (output_root / "human" / f"feedback_{approved['feedback_id']}.txt").exists()
    assert not list((output_root / "ai").glob("*.txt"))
    with metadata_path.open(encoding="utf-8", newline="") as file:
        metadata = list(csv.DictReader(file))
    assert metadata[0]["label"] == "human"
    assert metadata[0]["model_version"] == "phase2-v1"


def test_duplicate_feedback_is_not_inserted(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()
    analysis_id = record_analysis(VALID_TEXT, "human", 22.0, "phase2-v1")

    first = submit_feedback(analysis_id, "human", True)
    second = submit_feedback(analysis_id, "human", True)

    assert first["status"] == "pending"
    assert second == {
        "success": False,
        "feedback_id": first["feedback_id"],
        "id": first["feedback_id"],
        "status": "duplicate",
    }
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
