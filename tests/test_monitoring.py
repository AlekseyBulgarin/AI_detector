"""Production monitoring statistics are counts only, never raw text."""

import src.database as database
from src.decision import load_thresholds
from src.feedback import record_analysis, submit_feedback
from src.monitoring import compute_production_stats


REQUIRED_KEYS = {
    "generated_at",
    "predictions_total",
    "feedback_total",
    "feedback_pending",
    "feedback_rate",
    "model_disagreement_rate",
    "average_latency_ms",
    "low_confidence_rate",
    "false_positive_candidates",
    "false_negative_candidates",
    "high_disagreement",
    "ai_assisted_feedback",
    "last_prediction_at",
    "thresholds",
}


def test_stats_are_returned_for_an_empty_database(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()

    stats = compute_production_stats()

    assert REQUIRED_KEYS <= set(stats)
    assert stats["predictions_total"] == 0
    assert stats["feedback_rate"] == 0.0
    assert stats["model_disagreement_rate"] == 0.0
    assert stats["thresholds"]["low"] <= stats["thresholds"]["high"]


def test_stats_aggregate_predictions_feedback_and_disagreement(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()

    wrong = record_analysis(
        "Текст, который модель приняла за сгенерированный.",
        "ai",
        88.0,
        "phase2-v1",
        latency_ms=120.0,
    )
    uncertain = record_analysis(
        "Ещё один текст посередине шкалы уверенности модели.",
        "human",
        50.0,
        "phase2-v1",
        latency_ms=80.0,
    )
    record_analysis("Третий текст без единой отметки человека.", "ai", 91.0, "phase2-v1")

    submit_feedback(wrong, "human", False)

    stats = compute_production_stats()

    assert stats["predictions_total"] == 3
    assert stats["feedback_total"] == 1
    assert stats["feedback_rate"] == 0.3333
    assert stats["model_disagreement_rate"] == 1.0
    assert stats["average_latency_ms"] == 100.0
    assert stats["last_prediction_at"]

    thresholds = load_thresholds()
    low = thresholds["low"] * 100
    high = thresholds["high"] * 100
    expected_low_confidence = 1 if low <= 50.0 <= high else 0
    assert stats["low_confidence_rate"] == round(expected_low_confidence / 3, 4)

    # Statistics only: no raw text is exposed.
    assert "text" not in stats
    assert "user_text" not in stats


def test_hard_case_counts_track_review_queue_categories(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()

    false_positive = record_analysis(
        "Очень человеческий текст, который всё же попал в категорию ИИ.",
        "ai",
        95.0,
        "phase2-v1",
    )
    false_negative = record_analysis(
        "Явно сгенерированный текст, отмеченный преподавателем как человеческий.",
        "human",
        5.0,
        "phase2-v1",
    )
    assisted = record_analysis(
        "Текст, где ученик использовал нейросеть для черновика.",
        "human",
        30.0,
        "phase2-v1",
    )

    submit_feedback(false_positive, "human", False)
    submit_feedback(false_negative, "ai", False)
    submit_feedback(assisted, "ai_assisted", False)

    stats = compute_production_stats()

    assert stats["false_positive_candidates"] == 1
    assert stats["false_negative_candidates"] == 1
    assert stats["ai_assisted_feedback"] == 1
