"""Integration tests for Model Quantization REST API endpoints."""

from fastapi.testclient import TestClient
import pytest

from backend.app.main import app

client = TestClient(app)


def test_list_quantization_results():
    response = client.get("/api/quantization/results?base_model_id=Qwen/Qwen2.5-0.5B-Instruct")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_precision_comparison_endpoint():
    response = client.get("/api/quantization/comparison?base_model_id=Qwen/Qwen2.5-0.5B-Instruct")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_run_gguf_quantization_api():
    payload = {
        "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "quantization_method": "gguf_q4_k_m",
        "target_precision": "q4_k_m",
        "evaluate_after_quant": False,
        "eval_samples": 2,
    }
    response = client.post("/api/quantization/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "gguf_q4_k_m"
    assert data["target_precision"] == "q4_k_m"
    assert data["memory_reduction_pct"] > 60.0
    assert data["status"] == "completed"


def test_get_quantization_nonexistent():
    response = client.get("/api/quantization/non_existent_id")
    assert response.status_code == 404

