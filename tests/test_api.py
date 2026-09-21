import uuid

from app import app


def test_api_check_returns_analysis_id_and_probability():
    client = app.test_client()
    response = client.post(
        "/api/check",
        json={"text": "Это достаточно длинный текст для проверки работы API модели."},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload["analysis_id"], str)
    assert 0 <= payload["probability"] <= 100


def test_feedback_is_saved_as_pending():
    client = app.test_client()
    analysis = client.post(
        "/api/check",
        json={"text": f"Это достаточно длинный текст для проверки сохранения обратной связи {uuid.uuid4()}."},
    ).get_json()

    response = client.post(
        "/api/feedback",
        json={"analysis_id": analysis["analysis_id"], "label": "unsure"},
    )

    assert response.status_code == 201
    assert response.get_json()["status"] == "pending"
