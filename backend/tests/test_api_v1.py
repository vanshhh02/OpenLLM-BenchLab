"""Integration tests for FastAPI endpoints: datasets, models, evaluation, experiments."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "openllm-benchlab"


def test_hardware_endpoint():
    response = client.get("/api/hardware")
    assert response.status_code == 200
    data = response.json()
    assert "platform_system" in data
    assert "accelerator_type" in data
    assert data["total_ram_gb"] > 0


def test_list_datasets():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    billsum = next((d for d in data if d["id"] == "billsum"), None)
    assert billsum is not None


def test_dataset_statistics():
    response = client.get("/api/datasets/billsum/statistics")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "billsum"
    assert data["train_count"] > 0
    assert "input_tokens" in data["metadata_json"]


def test_list_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 3
    qwen = next((m for m in data if "Qwen" in m["id"]), None)
    assert qwen is not None


def test_get_model_details():
    response = client.get("/api/models/Qwen/Qwen2.5-0.5B-Instruct")
    assert response.status_code == 200
    data = response.json()
    assert data["parameter_count_b"] == 0.49
    assert data["context_window"] == 32768


def test_list_evaluations():
    response = client.get("/api/evaluation/results")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["rougeL"] > 0.0


def test_compare_evaluations():
    response = client.get("/api/evaluation/compare")
    assert response.status_code == 200
    data = response.json()
    assert "evaluations" in data
    assert "metrics_comparison" in data


def test_list_experiments():
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    categories = [e["category"] for e in data]
    assert "baseline" in categories or "lora" in categories

