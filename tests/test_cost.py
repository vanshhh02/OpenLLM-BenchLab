"""Unit tests for Phase 6 Cost Analysis & Frontier ROI Modeling Engine."""

import pytest
from ml.cost_analysis.calculator import (
    CostAnalysisEngine,
    CostCalculator,
    FRONTIER_API_PRESETS,
    HARDWARE_PRESETS,
)


def test_calculate_api_cost():
    # 100,000 requests, 1,500 input tokens, 250 output tokens
    # input rate: $0.150/1M, output rate: $0.600/1M (GPT-4o-mini rates)
    res = CostCalculator.calculate_api_cost(
        requests=100000,
        input_tokens=1500,
        output_tokens=250,
        input_cost_per_m=0.150,
        output_cost_per_m=0.600,
    )

    # Expected per request: (1500 * 0.150 / 1e6) + (250 * 0.600 / 1e6)
    # = 0.000225 + 0.000150 = 0.000375
    assert res["cost_per_request_usd"] == 0.000375
    assert res["cost_per_1k_usd"] == 0.375
    assert res["total_monthly_usd"] == 37.5
    assert res["total_annual_usd"] == 450.0


def test_calculate_electricity():
    # 100k requests, 250 output tokens, 50 tokens/sec
    # -> 5 sec/req -> 500,000 total active seconds (~138.89 active hours)
    res = CostCalculator.calculate_electricity(
        requests=100000,
        avg_output_tokens=250,
        tokens_per_second=50.0,
        power_watts=200.0,
        idle_power_watts=50.0,
        electricity_kwh_cost_usd=0.15,
    )

    assert res["active_hours"] > 0
    assert res["idle_hours"] > 0
    assert res["active_hours"] + res["idle_hours"] == 730.0
    assert res["total_kwh"] > 0
    assert res["monthly_electricity_cost_usd"] > 0
    assert res["variable_cost_per_request_usd"] > 0


def test_cost_analysis_high_volume():
    # At 1 million requests, self-hosted cloud RTX 4090 ($85/mo) should achieve major cost savings
    # compared to GPT-4o standard rate ($2.50 in, $10.00 out)
    res = CostCalculator.analyze(
        monthly_requests=1000000,
        avg_input_tokens=1500,
        avg_output_tokens=250,
        api_name="OpenAI GPT-4o",
        api_input_cost_per_m=2.50,
        api_output_cost_per_m=10.00,
        server_monthly_rental_usd=85.0,
        hardware_purchase_cost_usd=0.0,
        server_power_watts=350.0,
        electricity_kwh_cost_usd=0.15,
        operational_overhead_monthly_usd=25.0,
        tokens_per_second_local=80.0,
    )

    assert res.monthly_api_cost_usd > res.monthly_local_cost_usd
    assert res.monthly_savings_usd > 0
    assert res.savings_percentage > 50.0
    assert res.break_even_status == "achievable"
    assert res.break_even_requests is not None
    assert res.break_even_requests < 1000000
    assert len(res.sensitivity_curve) > 5


def test_cost_analysis_low_volume_api_advantage():
    # At 1,000 requests, commercial API ($0.38/month) is vastly cheaper than paying $85/month GPU rental
    res = CostCalculator.analyze(
        monthly_requests=1000,
        avg_input_tokens=1500,
        avg_output_tokens=250,
        api_name="OpenAI GPT-4o-mini",
        api_input_cost_per_m=0.150,
        api_output_cost_per_m=0.600,
        server_monthly_rental_usd=85.0,
        hardware_purchase_cost_usd=0.0,
        operational_overhead_monthly_usd=20.0,
    )

    assert res.monthly_api_cost_usd < res.monthly_local_cost_usd
    assert res.monthly_savings_usd < 0  # Negative savings means self-hosting loses money at low volume
    assert res.sensitivity_curve[0]["recommended_architecture"] == "frontier_api"


def test_presets():
    engine = CostAnalysisEngine()
    frontier = engine.get_frontier_presets()
    assert "gpt-4o" in frontier
    assert "gpt-4o-mini" in frontier
    assert "claude-3-5-sonnet" in frontier
    assert "gemini-1.5-flash" in frontier

    hw = engine.get_hardware_presets()
    assert "cloud_rtx4090" in hw
    assert "local_mac_studio_m2_ultra" in hw
    assert "cloud_a100_80gb" in hw


def test_pareto_frontier_computation():
    evals = [
        # Dominated model (lower rouge, higher cost, higher latency than Model B)
        {"id": "m1", "name": "Inferior Model", "rougeL": 30.0, "cost_per_1k_requests": 1.50, "avg_latency_ms": 250.0},
        # Model B: Good quality, low cost
        {"id": "m2", "name": "Qwen 0.5B LoRA", "rougeL": 38.5, "cost_per_1k_requests": 0.05, "avg_latency_ms": 90.0},
        # Model C: Frontier API - Highest quality, highest cost
        {"id": "m3", "name": "GPT-4o", "rougeL": 44.0, "cost_per_1k_requests": 6.25, "avg_latency_ms": 350.0},
        # Model D: Ultra fast, moderate quality
        {"id": "m4", "name": "Qwen 0.5B Quantized INT8", "rougeL": 37.8, "cost_per_1k_requests": 0.03, "avg_latency_ms": 45.0},
    ]

    pareto = CostAnalysisEngine.compute_pareto_frontier(evals)
    pareto_ids = {m["id"] for m in pareto}

    assert "m1" not in pareto_ids  # Should be filtered out
    assert "m2" in pareto_ids
    assert "m3" in pareto_ids
    assert "m4" in pareto_ids

