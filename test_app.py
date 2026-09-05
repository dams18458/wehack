"""
Unit tests for the Silent Alert Dashboard Flask app.
Verifies GET /, POST /trigger, GET /api/alerts, and POST /reset endpoints.
"""

import json
import pytest
from app import app, ALERTS_HISTORY


@pytest.fixture
def client():
    app.config["TESTING"] = True
    ALERTS_HISTORY.clear()
    with app.test_client() as client:
        yield client
    ALERTS_HISTORY.clear()


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    text = res.get_data(as_text=True)
    assert "Silent Dispatch" in text
    assert "Incident Proxy Monitor" in text
    assert "statusPanel" in text


def test_initial_api_alerts_is_idle(client):
    res = client.get("/api/alerts")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "idle"
    assert data["count"] == 0
    assert data["alerts"] == []
    assert data["latest"] is None


def test_post_trigger_and_retrieve(client):
    payload = {
        "contact": "Sarah Chen (Emergency Contact)",
        "timestamp": "2026-09-05 10:45:00",
        "confidence": 91.5,
        "audio_path": "evidence/duress_20260905.wav",
    }
    res_post = client.post("/trigger", json=payload)
    assert res_post.status_code == 201
    post_data = res_post.get_json()
    assert post_data["status"] == "success"
    assert post_data["alert"]["contact"] == payload["contact"]
    assert post_data["alert"]["confidence"] == 91.5

    # Check API status changes to alert_sent
    res_get = client.get("/api/alerts")
    assert res_get.status_code == 200
    get_data = res_get.get_json()
    assert get_data["status"] == "alert_sent"
    assert get_data["count"] == 1
    assert get_data["latest"]["contact"] == payload["contact"]
    assert get_data["latest"]["confidence"] == 91.5


def test_post_multiple_alerts_order(client):
    # First alert
    client.post("/trigger", json={"contact": "Contact A", "confidence": 75.0})
    # Second alert
    client.post("/trigger", json={"contact": "Contact B", "confidence": 92.0})

    res = client.get("/api/alerts")
    data = res.get_json()
    assert data["count"] == 2
    # Latest alert must be Contact B (newest first)
    assert data["latest"]["contact"] == "Contact B"
    assert data["alerts"][0]["contact"] == "Contact B"
    assert data["alerts"][1]["contact"] == "Contact A"


def test_reset_endpoint(client):
    client.post("/trigger", json={"contact": "Contact Test", "confidence": 80.0})
    assert len(ALERTS_HISTORY) == 1

    res_reset = client.post("/reset")
    assert res_reset.status_code == 200
    assert len(ALERTS_HISTORY) == 0

    res_get = client.get("/api/alerts")
    data = res_get.get_json()
    assert data["status"] == "idle"
    assert data["count"] == 0