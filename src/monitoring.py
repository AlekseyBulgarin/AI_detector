"""Aggregate production statistics.

Only counts and numbers are returned; raw text never leaves the database.
"""

from datetime import datetime, timezone

from src.database import get_connection
from src.decision import load_thresholds


def compute_production_stats(database_path=None):
    thresholds = load_thresholds()
    low = thresholds["low"] * 100
    high = thresholds["high"] * 100

    with get_connection(database_path) as connection:
        prediction_row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   AVG(latency_ms) AS avg_latency,
                   MAX(created_at) AS last_prediction
            FROM analysis
            """
        ).fetchone()
        feedback_row = connection.execute(
            "SELECT COUNT(*) AS total FROM feedback"
        ).fetchone()
        pending_row = connection.execute(
            "SELECT COUNT(*) AS total FROM feedback WHERE status = 'pending'"
        ).fetchone()
        disagreement_row = connection.execute(
            """
            SELECT COUNT(*) AS total FROM feedback
            WHERE status IN ('approved', 'pending', 'needs_review')
              AND CASE
                    WHEN prediction_label = 'ai' THEN 'ai'
                    WHEN prediction_label = 'human' THEN 'human'
                    ELSE prediction_label
                  END <> user_label
            """
        ).fetchone()
        priority_rows = connection.execute(
            """
            SELECT priority_category, COUNT(*) AS total
            FROM feedback GROUP BY priority_category
            """
        ).fetchall()
        low_confidence_row = connection.execute(
            "SELECT COUNT(*) AS total FROM analysis WHERE probability >= ? AND probability <= ?",
            (low, high),
        ).fetchone()

    predictions_total = int(prediction_row["total"] or 0)
    feedback_total = int(feedback_row["total"] or 0)
    by_category = {row["priority_category"]: int(row["total"]) for row in priority_rows}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions_total": predictions_total,
        "feedback_total": feedback_total,
        "feedback_pending": int(pending_row["total"] or 0),
        "feedback_rate": round(feedback_total / predictions_total, 4)
        if predictions_total
        else 0.0,
        "model_disagreement_rate": round(
            int(disagreement_row["total"] or 0) / feedback_total, 4
        )
        if feedback_total
        else 0.0,
        "average_latency_ms": round(float(prediction_row["avg_latency"] or 0), 3),
        "low_confidence_rate": round(
            int(low_confidence_row["total"] or 0) / predictions_total, 4
        )
        if predictions_total
        else 0.0,
        "false_positive_candidates": by_category.get("false_positive_candidate", 0),
        "false_negative_candidates": by_category.get("false_negative_candidate", 0),
        "high_disagreement": by_category.get("high_disagreement", 0),
        "ai_assisted_feedback": by_category.get("ai_assisted", 0),
        "last_prediction_at": prediction_row["last_prediction"],
        "thresholds": {
            "low": thresholds["low"],
            "high": thresholds["high"],
            "source": thresholds.get("source"),
        },
    }
