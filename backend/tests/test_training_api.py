"""Integration tests for fine-tuning API endpoints."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_get_training_history():
    resp = client.get("/api/training/history")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_get_training_status_nonexistent():
    resp = client.get("/api/training/status/non_existent_job_12345")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()

