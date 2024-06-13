from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_anonymize_invalid_input():
    response = client.post("/anonymize", json={})
    assert response.status_code == 422  # Unprocessable Entity


def test_deanonymize_invalid_input():
    response = client.post("/deanonymize", json={})
    assert response.status_code == 422  # Unprocessable Entity


def test_pseudoanonymize_flow():
    response = client.post("/pseudoanonymize", json={"text": "Hi, I am Bob"})
    assert response.status_code == 200
    data = response.json()
    assert "anonymizedText" in data
    assert "replacementDict" in data
    assert "deanonymizedText" in data
