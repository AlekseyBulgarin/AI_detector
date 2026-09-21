from collections import OrderedDict
import logging
import uuid

import app as application


VALID_TEXT = "Это достаточно длинный учебный текст для проверки API маршрутов."


def test_index_renders_dashboard_contract():
    response = application.app.test_client().get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="view-analyzer"' in body
    assert 'id="view-history"' in body
    assert 'id="view-settings"' in body
    assert 'id="user_text"' in body


def test_api_check_rejects_missing_and_short_text():
    client = application.app.test_client()

    missing = client.post("/api/check", json={})
    short = client.post("/api/check", json={"text": "коротко"})

    assert missing.status_code == 400
    assert short.status_code == 400
    assert missing.get_json()["probability"] is None
    assert "слишком короткий" in short.get_json()["error"]


def test_api_check_rejects_non_string_text():
    response = application.app.test_client().post("/api/check", json={"text": 42})

    assert response.status_code == 400
    assert "строкой" in response.get_json()["error"]


def test_api_check_returns_service_error_when_model_prediction_fails(monkeypatch):
    def fail_prediction(_text):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(application, "predict_probability", fail_prediction)
    response = application.app.test_client().post("/api/check", json={"text": VALID_TEXT})

    assert response.status_code == 503
    assert response.get_json() == {"error": "Модель временно недоступна", "probability": None}


def test_form_check_handles_short_text_without_prediction(monkeypatch):
    called = False

    def unexpected_prediction(_text):
        nonlocal called
        called = True
        return 50

    monkeypatch.setattr(application, "predict_probability", unexpected_prediction)
    response = application.app.test_client().post("/check", data={"user_text": "коротко"})

    assert response.status_code == 200
    assert "Текст слишком короткий" in response.get_data(as_text=True)
    assert called is False


def test_feedback_validates_payload_and_unknown_analysis():
    client = application.app.test_client()

    invalid_label = client.post(
        "/api/feedback", json={"analysis_id": "missing", "label": "maybe"}
    )
    unknown_analysis = client.post(
        "/api/feedback", json={"analysis_id": "missing", "label": "human"}
    )

    assert invalid_label.status_code == 400
    assert unknown_analysis.status_code == 404
    assert unknown_analysis.get_json()["error"] == "Analysis not found"


def test_health_reports_model_contract(monkeypatch):
    monkeypatch.setattr(application, "model", object())
    monkeypatch.setattr(application, "model_version", "test-model")

    response = application.app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "test-model",
    }


def test_repeated_text_reuses_cached_analysis(monkeypatch):
    calls = {"predict": 0, "record": 0}
    monkeypatch.setattr(application, "analysis_cache", OrderedDict())
    monkeypatch.setattr(application, "find_analysis_by_hash", lambda *_args: None)

    def fake_prediction(_text):
        calls["predict"] += 1
        return 42.5

    def fake_record(*_args):
        calls["record"] += 1
        return "cached-analysis"

    monkeypatch.setattr(application, "predict_probability", fake_prediction)
    monkeypatch.setattr(application, "record_analysis", fake_record)

    first = application.analyze_text(VALID_TEXT)
    second = application.analyze_text(VALID_TEXT)

    assert first == second
    assert first["id"] == "cached-analysis"
    assert calls == {"predict": 1, "record": 1}


def test_request_timing_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="app"):
        response = application.app.test_client().get("/health")

    assert response.status_code in {200, 503}
    assert any("event=request_completed" in record.message for record in caplog.records)


def test_feedback_api_returns_duplicate_and_preserves_consent():
    client = application.app.test_client()
    text = f"Это уникальный текст для проверки feedback API {uuid.uuid4()}."
    analysis = client.post("/api/check", json={"text": text}).get_json()

    first = client.post(
        "/api/feedback",
        json={"analysis_id": analysis["analysis_id"], "label": "ai", "allow_training": True},
    )
    second = client.post(
        "/api/feedback",
        json={"analysis_id": analysis["analysis_id"], "label": "ai", "allow_training": True},
    )

    assert first.status_code == 201
    assert first.get_json()["success"] is True
    assert second.status_code == 200
    assert second.get_json()["status"] == "duplicate"


def test_admin_feedback_page_and_review_action_work_without_configured_token():
    client = application.app.test_client()
    text = f"This is an admin review sample {uuid.uuid4()} with enough text."
    analysis = client.post("/api/check", json={"text": text}).get_json()
    feedback = client.post(
        "/api/feedback",
        json={"analysis_id": analysis["analysis_id"], "label": "human", "allow_training": True},
    ).get_json()
    review = client.post(
        f"/admin/feedback/{feedback['feedback_id']}",
        json={"status": "approved"},
    )
    page = client.get("/admin/feedback")

    assert page.status_code == 200
    assert review.status_code == 200
    assert review.get_json()["status"] == "approved"
    assert "Feedback review" in page.get_data(as_text=True)
