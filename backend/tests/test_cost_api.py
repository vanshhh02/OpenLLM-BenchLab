"""Integration tests for Cost Analysis API endpoints."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_calculate_cost_endpoint():
    payload = {
        "monthly_requests": 200000,
        "avg_input_tokens": 1200,
        "avg_output_tokens": 200,
        "api_name": "Test Commercial API",
        "api_input_cost_per_m": 0.200,
        "api_output_cost_per_m": 0.800,
        "server_monthly_rental_usd": 120.0,
        "hardware_purchase_cost_usd": 0.0,
        "amortization_months": 36,
        "server_power_watts": 300.0,
        "electricity_kwh_cost_usd": 0.15,
        "operational_overhead_monthly_usd": 30.0,
        "tokens_per_second_local": 70.0,
    }
    response = client.post("/api/cost/calculate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["monthly_requests"] == 200000
    assert data["api_cost_per_request_usd"] > 0
    assert data["monthly_api_cost_usd"] > 0
    assert data["monthly_local_cost_usd"] > 0
    assert "break_even_requests" in data
    assert "break_even_status" in data
    assert "sensitivity_curve" in data
    assert len(data["sensitivity_curve"]) > 5


def test_frontier_presets_endpoint():
    response = client.get("/api/cost/frontier-presets")
    assert response.status_code == 200
    data = response.json()
    assert "gpt-4o" in data
    assert "gpt-4o-mini" in data
    assert "claude-3-5-sonnet" in data
    assert "gemini-1.5-flash" in data


def test_hardware_presets_endpoint():
    response = client.get("/api/cost/hardware-presets")
    assert response.status_code == 200
    data = response.json()
    assert "cloud_rtx4090" in data
    assert "local_mac_studio_m2_ultra" in data


def test_pareto_analysis_endpoint():
    response = client.get("/api/cost/pareto-analysis")
    assert response.status_code == 200
    data = response.json()
    assert "total_experiments_evaluated" in data
    assert "pareto_optimal_count" in data
    assert "pareto_frontier" in data
    assert isinstance(data["pareto_frontier"], list)

