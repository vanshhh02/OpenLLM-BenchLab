"""Unit tests for InferenceEngine, ContinuousBatchingServer, and KVCacheCalculator."""

import asyncio
import pytest

from ml.inference.engine import ContinuousBatchingServer, InferenceEngine, KVCacheCalculator


def test_kv_cache_calculator_formula():
    """Verify exact formula: 2 * layers * B * L * kv_heads * head_dim * bytes_per_elem."""
    # Test for Qwen2.5-0.5B: layers=24, kv_heads=2, head_dim=64, precision=fp16 (2 bytes)
    # For B=1, L=1024:
    # 2 * 24 * 1 * 1024 * 2 * 64 * 2 = 12,582,912 bytes = 12.0 MB
    res = KVCacheCalculator.calculate_memory(
        batch_size=1,
        context_length=1024,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        precision="fp16",
    )
    assert res["total_bytes"] == 12582912
    assert res["total_memory_mb"] == 12.0
    assert res["bytes_per_element"] == 2.0


def test_kv_cache_precision_scaling():
    """Verify INT8 cuts KV memory in half and INT4 cuts it by 75% compared to FP16."""
    fp16 = KVCacheCalculator.calculate_memory(batch_size=4, context_length=2048, precision="fp16")
    int8 = KVCacheCalculator.calculate_memory(batch_size=4, context_length=2048, precision="int8")
    int4 = KVCacheCalculator.calculate_memory(batch_size=4, context_length=2048, precision="int4")

    assert int8["total_bytes"] == fp16["total_bytes"] / 2
    assert int4["total_bytes"] == fp16["total_bytes"] / 4


def test_kv_cache_scaling_curve():
    res = KVCacheCalculator.calculate_memory(batch_size=2, context_length=2048, precision="fp16")
    curve = res["scaling_curve"]
    assert len(curve) > 0
    assert curve[0]["context_length"] == 512
    assert curve[-1]["context_length"] == 32768
    assert curve[-1]["memory_mb"] > curve[0]["memory_mb"]


def test_inference_engine_generation():
    engine = InferenceEngine(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    res = engine.generate(
        prompt="Summarize: Section 1. This is the Clean Air Act.",
        max_output_tokens=15,
        temperature=0.0,
    )
    assert "generated_text" in res
    assert res["prompt_tokens"] > 0
    assert res["completion_tokens"] > 0
    assert res["total_latency_ms"] > 0
    assert res["time_to_first_token_ms"] > 0
    assert res["tokens_per_second"] > 0


def test_inference_engine_batch():
    engine = InferenceEngine(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    prompts = [
        "Bill 1: Education funding act.",
        "Bill 2: Clean water infrastructure act.",
    ]
    res = engine.generate_batch(prompts=prompts, max_output_tokens=10, temperature=0.0)
    assert len(res["results"]) == 2
    assert res["total_batch_latency_ms"] > 0
    assert res["throughput_tps"] > 0


@pytest.mark.asyncio
async def test_continuous_batching_server():
    server = ContinuousBatchingServer(model_id="Qwen/Qwen2.5-0.5B-Instruct", max_batch_size=4, batch_timeout_ms=10.0)
    await server.start()
    try:
        task1 = server.submit_request("Summarize bill A", max_output_tokens=5)
        task2 = server.submit_request("Summarize bill B", max_output_tokens=5)
        res1, res2 = await asyncio.gather(task1, task2)
        assert "generated_text" in res1
        assert "generated_text" in res2
    finally:
        await server.stop()

