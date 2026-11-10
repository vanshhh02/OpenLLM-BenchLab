"""Integration tests for Inference and Benchmarking REST API endpoints."""

from fastapi.testclient import TestClient
import pytest

from backend.app.main import app

client = TestClient(app)


def test_kv_cache_stats_endpoint():
    response = client.get("/api/inference/kv-cache-stats?batch_size=2&context_length=1024&precision=fp16")
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert data["context_length"] == 1024
    assert data["total_memory_mb"] > 0
    assert len(data["scaling_curve"]) > 0


def test_list_benchmarks_endpoint():
    response = client.get("/api/benchmark/results?model_id=Qwen/Qwen2.5-0.5B-Instruct")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_generate_text_api():
    payload = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": "Summarize: Section 1. This Act supports science education.",
        "max_output_tokens": 10,
        "temperature": 0.0,
    }
    response = client.post("/api/inference/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert data["prompt_tokens"] > 0
    assert data["completion_tokens"] > 0
    assert data["time_to_first_token_ms"] > 0
    assert data["total_latency_ms"] > 0
    assert data["tokens_per_second"] > 0


def test_batch_generate_api():
    payload = {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompts": [
            "Bill 1: Solar panel incentives.",
            "Bill 2: Clean river preservation.",
        ],
        "max_output_tokens": 8,
        "temperature": 0.0,
    }
    response = client.post("/api/inference/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["total_batch_latency_ms"] > 0
    assert data["throughput_tps"] > 0


def test_get_benchmark_nonexistent():
    response = client.get("/api/benchmark/non_existent_id")
    assert response.status_code == 404

