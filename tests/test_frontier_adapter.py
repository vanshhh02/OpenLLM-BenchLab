"""Unit tests for commercial frontier API adapter and mock fallback."""

import pytest
from ml.models.frontier_adapter import MockFrontierProvider, get_frontier_provider


def test_mock_frontier_provider():
    provider = MockFrontierProvider(model_id="gpt-4o-mini-baseline")
    prompt = "Summarize the following bill:\nSECTION 1. SHORT TITLE.\nThis is a clean energy grid bill."
    resp = provider.generate(prompt=prompt, max_tokens=100)

    assert resp.model_id == "gpt-4o-mini-baseline"
    assert resp.is_simulated is True
    assert len(resp.text) > 20
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert resp.cost_usd > 0
    assert resp.latency_ms > 0
    assert resp.ttft_ms > 0


def test_cost_calculation():
    provider = MockFrontierProvider()
    # 10,000 prompt tokens @ $0.15/1M, 1,000 completion tokens @ $0.60/1M
    cost = provider.calculate_cost(prompt_tokens=10000, completion_tokens=1000)
    expected = (10000 * 0.15 / 1e6) + (1000 * 0.60 / 1e6)
    assert abs(cost - expected) < 1e-5


def test_get_frontier_provider_fallback():
    provider = get_frontier_provider("unconfigured_provider")
    assert isinstance(provider, MockFrontierProvider)

