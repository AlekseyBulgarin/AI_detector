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
