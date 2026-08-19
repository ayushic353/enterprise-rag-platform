import os

os.environ["REQUIRE_API_KEY"] = "true"
os.environ["API_KEYS"] = "test-key-123"

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health_is_public():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_without_key_is_rejected():
    response = client.post("/chat", json={"question": "hello"})
    assert response.status_code == 401


def test_chat_with_bad_key_is_rejected():
    response = client.post(
        "/chat", json={"question": "hello"}, headers={"X-API-Key": "wrong"}
    )
    assert response.status_code == 401
