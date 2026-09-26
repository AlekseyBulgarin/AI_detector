"""End-to-end API contracts added by the human-in-the-loop release."""

import json
import sqlite3
import uuid

import app as application
import src.database as database
from src.decision import load_thresholds


VALID_TEXT = (
    "Это достаточно длинный учебный текст, используемый для проверки контракта "
    "нового API анализа и обратной связи преподавателя."
)


def _unique_text(prefix):
    return f"{prefix} {uuid.uuid4()} : {VALID_TEXT}"


def test_api_check_exposes_decision_timing_and_retention(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()

    response = application.app.test_client().post(
        "/api/check", json={"text": _unique_text("API-contract")}
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["decision"]["band"] in {
        "likely_human",
        "uncertain",
        "ai_indicators",
    }
    assert payload["decision"]["limitation"]
    assert payload["store_text"] is True
    assert payload["timing"]["feature_ms"] is not None
    assert payload["timing"]["prediction_ms"] is not None
    assert payload["timing"]["model_total_ms"] is not None
    assert payload["timing"]["db_write_ms"] is not None
    assert payload["latency_ms"] >= 0
    assert payload["model_version"]


def test_api_check_honours_store_text_false(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()

    text = _unique_text("no-retention")
    payload = application.app.test_client().post(
        "/api/check", json={"text": text, "store_text": False}
    ).get_json()

    assert payload["store_text"] is False
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT text_content, store_text FROM analysis WHERE id = ?",
            (payload["analysis_id"],),
        ).fetchone()
    assert row == (None, 0)


def test_api_check_validates_store_text_type():
    response = application.app.test_client().post(
        "/api/check", json={"text": VALID_TEXT, "store_text": "yes"}
    )

    assert response.status_code == 400
    assert "store_text" in response.get_json()["error"]


def test_form_check_renders_decision_and_review_controls():
    response = application.app.test_client().post(
        "/check", data={"user_text": _unique_text("form"), "store_text": "1"}
    )

    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'data-band="' in body
    assert 'data-model-version="' in body
    assert "verdict-meta" in body
    assert "threshold-scale" in body
    assert "latency-row" in body
    assert 'data-label="ai_assisted"' in body
    assert 'id="storeText"' in body
    assert 'id="predictionIncorrect"' in body
    assert 'data-analysis-id="' in body


def test_feedback_accepts_ai_assisted_and_confidence():
    client = application.app.test_client()
    analysis = client.post(
        "/api/check", json={"text": _unique_text("assisted")}
    ).get_json()

    response = client.post(
        "/api/feedback",
        json={
            "analysis_id": analysis["analysis_id"],
            "label": "ai_assisted",
            "allow_training": False,
            "user_confidence": "not_sure",
            "prediction_incorrect": True,
        },
    )

    assert response.status_code == 201
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["priority_category"] == "ai_assisted"
    assert "ai_assisted" in payload["categories"]

    with sqlite3.connect(database.DATABASE_PATH) as connection:
        row = connection.execute(
            "SELECT user_confidence, validation_flags FROM feedback WHERE id = ?",
            (payload["feedback_id"],),
        ).fetchone()

    assert row[0] == "not_sure"
    assert "prediction_incorrect" in json.loads(row[1])


def test_feedback_rejects_invalid_confidence():
    client = application.app.test_client()
    analysis = client.post(
        "/api/check", json={"text": _unique_text("confidence")}
    ).get_json()

    response = client.post(
        "/api/feedback",
        json={
            "analysis_id": analysis["analysis_id"],
            "label": "human",
            "user_confidence": "very",
        },
    )

    assert response.status_code == 400
    assert "user_confidence" in response.get_json()["error"]


def test_feedback_can_supply_text_when_retention_is_off(tmp_path, monkeypatch):
    database_path = tmp_path / "feedback.db"
    monkeypatch.setattr(database, "DATABASE_PATH", str(database_path))
    database.initialize_database()

    text = _unique_text("consent")
    analysis = application.app.test_client().post(
        "/api/check", json={"text": text, "store_text": False}
    ).get_json()

    response = application.app.test_client().post(
        "/api/feedback",
        json={
            "analysis_id": analysis["analysis_id"],
            "label": "human",
            "allow_training": True,
            "user_confidence": "certain",
            "text": text,
        },
    )

    assert response.status_code == 201
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT text_content, allow_training FROM feedback WHERE id = ?",
            (response.get_json()["feedback_id"],),
        ).fetchone()

    # The reviewer supplied the text, so it is retained for review and export.
    assert row[0] == text
    assert row[1] == 1


def test_metrics_endpoint_reports_production_statistics():
    response = application.app.test_client().get("/api/metrics")

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["predictions_total"] >= 0
    assert payload["model"]["version"] == application.model_version
    assert payload["thresholds"]["low"] <= payload["thresholds"]["high"]
    assert "text" not in payload


def test_thresholds_report_annotates_calibration_source():
    thresholds = load_thresholds()

    assert thresholds["source"] in {
        "fallback_not_calibrated",
        "validation_roc",
        "train_roc",
        "frozen_test_roc",
        "frozen_test_roc_recalibrated",
        "fallback_insufficient_data",
        "unknown",
    }
