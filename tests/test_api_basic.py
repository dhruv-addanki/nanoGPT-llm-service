from fastapi.testclient import TestClient

from service.config import ServiceSettings
from service.server import create_app


def build_test_client() -> TestClient:
    settings = ServiceSettings(
        mock_model=True,
        max_new_tokens_default=3,
        max_new_tokens_limit=5,
        max_context_tokens=64,
    )
    app = create_app(settings)
    return TestClient(app)


def test_health_endpoint():
    client = build_test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["loaded"] is True
    assert "device" in payload


def test_generate_endpoint():
    client = build_test_client()
    response = client.post(
        "/generate",
        json={"prompt": "Hello world", "max_new_tokens": 3, "temperature": 0.8},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["completion"] != ""
    assert payload["tokens_out"] > 0
