"""Unit tests for multi-concurrency LoadTester and latency percentile analysis."""

import tempfile
from pathlib import Path
import pytest

from ml.benchmarking.load_tester import LoadTester


@pytest.fixture
def temp_benchmark_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)


def test_load_tester_init(temp_benchmark_dir):
    tester = LoadTester(artifact_dir=temp_benchmark_dir)
    assert tester.artifact_dir == temp_benchmark_dir
    assert tester.hardware is not None


def test_load_test_sweep_fast(temp_benchmark_dir):
    tester = LoadTester(artifact_dir=temp_benchmark_dir)
    # Run fast test with 1 concurrency level and 2 requests
    summary = tester.run_load_test_sweep(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        concurrency_levels=[1],
        requests_per_level=2,
        prompt="Summarize this brief bill clause.",
        max_output_tokens=10,
    )
    assert summary["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert len(summary["levels"]) == 1
    level = summary["levels"][0]
    assert level["concurrency_level"] == 1
    assert level["successful_requests"] == 2
    assert level["failed_requests"] == 0
    assert level["p50_latency_ms"] > 0
    assert level["p95_latency_ms"] >= level["p50_latency_ms"]
    assert level["p99_latency_ms"] >= level["p95_latency_ms"]
    assert level["requests_per_second"] > 0
    assert level["tokens_per_second"] > 0
    assert level["kv_cache_usage_mb"] > 0

