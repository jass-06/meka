from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_requires_question():
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 400
