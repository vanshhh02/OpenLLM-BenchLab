"""Cost Analysis & Frontier ROI Modeling Engine for OpenLLM BenchLab.

Provides quantitative cost modeling, total cost of ownership (TCO) calculations,
break-even analysis between open-weight self-hosted LLMs and commercial APIs,
and sensitivity curve evaluations across inference volumes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ml.core.config import CostConfig

logger = logging.getLogger("benchlab.cost")

# Frontier Commercial API Pricing Presets (per 1M tokens in USD)
FRONTIER_API_PRESETS: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {
        "name": "OpenAI GPT-4o",
        "provider": "openai",
        "input_cost_per_m": 2.50,
        "output_cost_per_m": 10.00,
        "context_window": 128000,
        "notes": "Flagship multimodal frontier model",
    },
    "gpt-4o-mini": {
        "name": "OpenAI GPT-4o-mini",
        "provider": "openai",
        "input_cost_per_m": 0.150,
        "output_cost_per_m": 0.600,
        "context_window": 128000,
        "notes": "Cost-efficient lightweight frontier model",
    },
    "claude-3-5-sonnet": {
        "name": "Anthropic Claude 3.5 Sonnet",
        "provider": "anthropic",
        "input_cost_per_m": 3.00,
        "output_cost_per_m": 15.00,
        "context_window": 200000,
        "notes": "High intelligence, coding and reasoning benchmark leader",
    },
    "claude-3-5-haiku": {
        "name": "Anthropic Claude 3.5 Haiku",
        "provider": "anthropic",
        "input_cost_per_m": 0.80,
        "output_cost_per_m": 4.00,
        "context_window": 200000,
        "notes": "Fastest and lowest-latency Claude model",
    },
    "gemini-1.5-pro": {
        "name": "Google Gemini 1.5 Pro",
        "provider": "gemini",
        "input_cost_per_m": 1.25,
        "output_cost_per_m": 5.00,
        "context_window": 2000000,
        "notes": "Ultra-long 2M context window multimodal model",
    },
    "gemini-1.5-flash": {
        "name": "Google Gemini 1.5 Flash",
        "provider": "gemini",
        "input_cost_per_m": 0.075,
        "output_cost_per_m": 0.300,
        "context_window": 1000000,
        "notes": "High speed, cost-effective multimodal workhorse",
    },
}

# Dedicated & Self-Hosted Hardware Presets
HARDWARE_PRESETS: Dict[str, Dict[str, Any]] = {
    "local_mac_studio_m2_ultra": {
        "name": "Apple Mac Studio (M2 Ultra, 192GB Unified Memory)",
        "hardware_purchase_cost_usd": 4299.0,
        "server_monthly_rental_usd": 0.0,
        "power_watts": 180.0,
        "idle_power_watts": 35.0,
        "amortization_months": 36,
        "estimated_tps": 75.0,
        "type": "on_premise",
        "notes": "Can host 70B quantized models with zero recurring cloud bill",
    },
    "local_macbook_pro_m3_max": {
        "name": "Apple MacBook Pro (M3 Max, 64GB Unified Memory)",
        "hardware_purchase_cost_usd": 3499.0,
        "server_monthly_rental_usd": 0.0,
        "power_watts": 85.0,
        "idle_power_watts": 15.0,
        "amortization_months": 36,
        "estimated_tps": 55.0,
        "type": "on_premise",
        "notes": "Mobile workstation running 7B-14B models locally",
    },
    "cloud_rtx4090": {
        "name": "Cloud Dedicated RTX 4090 (24GB VRAM)",
        "hardware_purchase_cost_usd": 0.0,
        "server_monthly_rental_usd": 85.0,
        "power_watts": 350.0,
        "idle_power_watts": 50.0,
        "amortization_months": 1,
        "estimated_tps": 110.0,
        "type": "cloud_dedicated",
        "notes": "RunPod / Vast.ai dedicated community cloud GPU instance",
    },
    "cloud_a100_80gb": {
        "name": "Cloud Dedicated A100-SXM4 (80GB VRAM)",
        "hardware_purchase_cost_usd": 0.0,
        "server_monthly_rental_usd": 680.0,
        "power_watts": 400.0,
        "idle_power_watts": 80.0,
        "amortization_months": 1,
        "estimated_tps": 165.0,
        "type": "cloud_enterprise",
        "notes": "High-throughput production serving with vLLM / continuous batching",
    },
    "cloud_l4_gpu": {
        "name": "Cloud Google Cloud L4 (24GB VRAM)",
        "hardware_purchase_cost_usd": 0.0,
        "server_monthly_rental_usd": 210.0,
        "power_watts": 120.0,
        "idle_power_watts": 30.0,
        "amortization_months": 1,
        "estimated_tps": 80.0,
        "type": "cloud_enterprise",
        "notes": "Energy-efficient modern Ada Lovelace cloud inference GPU",
    },
    "cloud_t4_gpu": {
        "name": "Cloud AWS / GCP T4 (16GB VRAM)",
        "hardware_purchase_cost_usd": 0.0,
        "server_monthly_rental_usd": 95.0,
        "power_watts": 70.0,
        "idle_power_watts": 25.0,
        "amortization_months": 1,
        "estimated_tps": 40.0,
        "type": "cloud_budget",
        "notes": "Budget GPU for smaller quantized 0.5B-3B models",
    },
}


@dataclass
class CostAnalysisResult:
    """Comprehensive output container for cost comparison modeling."""
    monthly_requests: int
    avg_input_tokens: int
    avg_output_tokens: int

    # API figures
    api_name: str
    api_cost_per_request_usd: float
    api_cost_per_1k_usd: float
    monthly_api_cost_usd: float
    annual_api_cost_usd: float

    # Local / Dedicated infrastructure figures
    local_fixed_cost_monthly_usd: float
    local_electricity_cost_monthly_usd: float
    local_variable_cost_monthly_usd: float
    monthly_local_cost_usd: float
    annual_local_cost_usd: float
    local_cost_per_request_usd: float
    local_cost_per_1k_usd: float

    # Comparative economics
    break_even_requests: Optional[float]
    break_even_status: str
    monthly_savings_usd: float
    savings_percentage: float
    annual_savings_usd: float

    # Detailed component breakdown
    electricity_kwh_monthly: float
    hardware_amortization_monthly_usd: float
    server_monthly_rental_usd: float
    operational_overhead_monthly_usd: float

    # Sensitivity across volumes
    sensitivity_curve: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CostCalculator:
    """Core mathematical engine for LLM operational economic modeling."""

    @staticmethod
    def calculate_api_cost(
        requests: int,
        input_tokens: int,
        output_tokens: int,
        input_cost_per_m: float,
        output_cost_per_m: float,
    ) -> Dict[str, float]:
        """Compute costs for commercial API usage."""
        cost_per_req = (input_tokens * input_cost_per_m / 1e6) + (
            output_tokens * output_cost_per_m / 1e6
        )
        total_cost = requests * cost_per_req
        return {
            "cost_per_request_usd": round(cost_per_req, 6),
            "cost_per_1k_usd": round(cost_per_req * 1000.0, 4),
            "total_monthly_usd": round(total_cost, 2),
            "total_annual_usd": round(total_cost * 12.0, 2),
        }

    @staticmethod
    def calculate_electricity(
        requests: int,
        avg_output_tokens: int,
        tokens_per_second: float,
        power_watts: float,
        idle_power_watts: float,
        electricity_kwh_cost_usd: float,
    ) -> Dict[str, float]:
        """Calculate power draw and electricity expense for self-hosted infrastructure.
        
        Takes into account active compute hours under load plus idle standby hours.
        """
        if tokens_per_second <= 0:
            tokens_per_second = 50.0

        # Compute seconds per request
        seconds_per_req = avg_output_tokens / tokens_per_second
        active_hours = (requests * seconds_per_req) / 3600.0
        
        # Standard month: 730 hours
        total_month_hours = 730.0
        active_hours = min(active_hours, total_month_hours)
        idle_hours = max(0.0, total_month_hours - active_hours)

        active_kwh = (active_hours * power_watts) / 1000.0
        idle_kwh = (idle_hours * idle_power_watts) / 1000.0
        total_kwh = active_kwh + idle_kwh

        monthly_electricity_cost = total_kwh * electricity_kwh_cost_usd
        variable_kwh_per_req = (seconds_per_req * power_watts) / (3600.0 * 1000.0)
        variable_cost_per_req = variable_kwh_per_req * electricity_kwh_cost_usd

        return {
            "active_hours": round(active_hours, 2),
            "idle_hours": round(idle_hours, 2),
            "total_kwh": round(total_kwh, 2),
            "monthly_electricity_cost_usd": round(monthly_electricity_cost, 2),
            "variable_cost_per_request_usd": round(variable_cost_per_req, 6),
        }

    @classmethod
    def analyze(
        cls,
        monthly_requests: int = 100000,
        avg_input_tokens: int = 1500,
        avg_output_tokens: int = 250,
        api_name: str = "Frontier API",
        api_input_cost_per_m: float = 0.150,
        api_output_cost_per_m: float = 0.600,
        server_monthly_rental_usd: float = 85.0,
        hardware_purchase_cost_usd: float = 0.0,
        amortization_months: int = 36,
        server_power_watts: float = 250.0,
        idle_power_watts: Optional[float] = None,
        electricity_kwh_cost_usd: float = 0.15,
        operational_overhead_monthly_usd: float = 20.0,
        tokens_per_second_local: float = 65.0,
    ) -> CostAnalysisResult:
        """Run full comparative cost analysis between API and self-hosted infrastructure."""
        if idle_power_watts is None:
            idle_power_watts = server_power_watts * 0.25

        # 1. API Pricing
        api_calc = cls.calculate_api_cost(
            requests=monthly_requests,
            input_tokens=avg_input_tokens,
            output_tokens=avg_output_tokens,
            input_cost_per_m=api_input_cost_per_m,
            output_cost_per_m=api_output_cost_per_m,
        )

        # 2. Electricity
        elec_calc = cls.calculate_electricity(
            requests=monthly_requests,
            avg_output_tokens=avg_output_tokens,
            tokens_per_second=tokens_per_second_local,
            power_watts=server_power_watts,
            idle_power_watts=idle_power_watts,
            electricity_kwh_cost_usd=electricity_kwh_cost_usd,
        )

        # 3. Fixed and Amortized Costs
        amortization_months = max(1, amortization_months)
        hw_amortization_monthly = hardware_purchase_cost_usd / amortization_months
        fixed_infra_monthly = (
            server_monthly_rental_usd
            + hw_amortization_monthly
            + operational_overhead_monthly_usd
        )

        total_local_monthly = fixed_infra_monthly + elec_calc["monthly_electricity_cost_usd"]
        total_local_annual = total_local_monthly * 12.0
        local_cost_per_req = total_local_monthly / max(1, monthly_requests)
        local_cost_per_1k = local_cost_per_req * 1000.0

        # 4. Break-Even Analysis
        api_cost_per_req = api_calc["cost_per_request_usd"]
        var_cost_per_req = elec_calc["variable_cost_per_request_usd"]
        marginal_margin_per_req = api_cost_per_req - var_cost_per_req

        baseline_idle_elec = (
            (730.0 * idle_power_watts / 1000.0) * electricity_kwh_cost_usd
            if server_monthly_rental_usd == 0 and hardware_purchase_cost_usd > 0
            else 0.0
        )
        effective_fixed_monthly = fixed_infra_monthly + baseline_idle_elec

        if marginal_margin_per_req > 0:
            break_even_requests = round(effective_fixed_monthly / marginal_margin_per_req, 1)
            break_even_status = "achievable"
        else:
            break_even_requests = None
            break_even_status = "api_cheaper_per_token"

        # 5. Comparative Savings
        monthly_savings = api_calc["total_monthly_usd"] - total_local_monthly
        annual_savings = monthly_savings * 12.0
        savings_pct = (
            round((monthly_savings / api_calc["total_monthly_usd"]) * 100.0, 2)
            if api_calc["total_monthly_usd"] > 0
            else 0.0
        )

        # 6. Sensitivity Curve across diverse volume tiers
        volume_steps = [
            1_000,
            5_000,
            10_000,
            25_000,
            50_000,
            100_000,
            250_000,
            500_000,
            1_000_000,
            2_500_000,
            5_000_000,
        ]
        sensitivity_curve = []
        for vol in volume_steps:
            sub_api = cls.calculate_api_cost(
                requests=vol,
                input_tokens=avg_input_tokens,
                output_tokens=avg_output_tokens,
                input_cost_per_m=api_input_cost_per_m,
                output_cost_per_m=api_output_cost_per_m,
            )
            sub_elec = cls.calculate_electricity(
                requests=vol,
                avg_output_tokens=avg_output_tokens,
                tokens_per_second=tokens_per_second_local,
                power_watts=server_power_watts,
                idle_power_watts=idle_power_watts,
                electricity_kwh_cost_usd=electricity_kwh_cost_usd,
            )
            sub_local_total = fixed_infra_monthly + sub_elec["monthly_electricity_cost_usd"]
            sub_savings = sub_api["total_monthly_usd"] - sub_local_total
            sub_savings_pct = (
                round((sub_savings / sub_api["total_monthly_usd"]) * 100.0, 2)
                if sub_api["total_monthly_usd"] > 0
                else 0.0
            )

            rec = "frontier_api" if sub_api["total_monthly_usd"] < sub_local_total else "self_hosted"

            sensitivity_curve.append({
                "monthly_requests": vol,
                "api_monthly_cost_usd": sub_api["total_monthly_usd"],
                "local_monthly_cost_usd": round(sub_local_total, 2),
                "savings_usd": round(sub_savings, 2),
                "savings_percentage": sub_savings_pct,
                "recommended_architecture": rec,
            })

        return CostAnalysisResult(
            monthly_requests=monthly_requests,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            api_name=api_name,
            api_cost_per_request_usd=api_calc["cost_per_request_usd"],
            api_cost_per_1k_usd=api_calc["cost_per_1k_usd"],
            monthly_api_cost_usd=api_calc["total_monthly_usd"],
            annual_api_cost_usd=api_calc["total_annual_usd"],
            local_fixed_cost_monthly_usd=round(fixed_infra_monthly, 2),
            local_electricity_cost_monthly_usd=elec_calc["monthly_electricity_cost_usd"],
            local_variable_cost_monthly_usd=round(
                elec_calc["variable_cost_per_request_usd"] * monthly_requests, 2
            ),
            monthly_local_cost_usd=round(total_local_monthly, 2),
            annual_local_cost_usd=round(total_local_annual, 2),
            local_cost_per_request_usd=round(local_cost_per_req, 6),
            local_cost_per_1k_usd=round(local_cost_per_1k, 4),
            break_even_requests=break_even_requests,
            break_even_status=break_even_status,
            monthly_savings_usd=round(monthly_savings, 2),
            savings_percentage=savings_pct,
            annual_savings_usd=round(annual_savings, 2),
            electricity_kwh_monthly=elec_calc["total_kwh"],
            hardware_amortization_monthly_usd=round(hw_amortization_monthly, 2),
            server_monthly_rental_usd=server_monthly_rental_usd,
            operational_overhead_monthly_usd=operational_overhead_monthly_usd,
            sensitivity_curve=sensitivity_curve,
        )


class CostAnalysisEngine:
    """High-level facade to integrate cost analysis with database evaluations & presets."""

    def __init__(self, default_config: Optional[CostConfig] = None):
        self.config = default_config or CostConfig()

    @staticmethod
    def get_frontier_presets() -> Dict[str, Dict[str, Any]]:
        return FRONTIER_API_PRESETS

    @staticmethod
    def get_hardware_presets() -> Dict[str, Dict[str, Any]]:
        return HARDWARE_PRESETS

    def run_calculation(self, **kwargs) -> CostAnalysisResult:
        merged_params = self.config.model_dump()
        merged_params.update({k: v for k, v in kwargs.items() if v is not None})

        return CostCalculator.analyze(
            monthly_requests=merged_params.get("monthly_requests", 100000),
            avg_input_tokens=merged_params.get("avg_input_tokens", 1500),
            avg_output_tokens=merged_params.get("avg_output_tokens", 250),
            api_name=merged_params.get("api_name", "Frontier API"),
            api_input_cost_per_m=merged_params.get("api_input_cost_per_m", 0.150),
            api_output_cost_per_m=merged_params.get("api_output_cost_per_m", 0.600),
            server_monthly_rental_usd=merged_params.get("server_monthly_rental_usd", 85.0),
            hardware_purchase_cost_usd=merged_params.get("hardware_purchase_cost_usd", 0.0),
            amortization_months=merged_params.get("amortization_months", 36),
            server_power_watts=merged_params.get("server_power_watts", 250.0),
            idle_power_watts=merged_params.get("idle_power_watts"),
            electricity_kwh_cost_usd=merged_params.get("electricity_kwh_cost_usd", 0.15),
            operational_overhead_monthly_usd=merged_params.get(
                "operational_overhead_monthly_usd", 20.0
            ),
            tokens_per_second_local=merged_params.get("tokens_per_second_local", 65.0),
        )

    @staticmethod
    def compute_pareto_frontier(
        evaluations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify Pareto-optimal models optimizing Quality (ROUGE-L) vs Cost vs Latency.
        
        A model A is Pareto-optimal if no other model B has:
        ROUGE-L >= A, Cost <= A, Latency <= A with at least one strictly better.
        """
        if not evaluations:
            return []

        pareto_front = []
        for candidate in evaluations:
            c_rouge = candidate.get("rougeL", 0.0) or 0.0
            c_cost = candidate.get("cost_per_1k_requests", 0.0) or 0.0
            c_lat = candidate.get("avg_latency_ms", 0.0) or 0.0

            dominated = False
            for other in evaluations:
                if other is candidate:
                    continue
                o_rouge = other.get("rougeL", 0.0) or 0.0
                o_cost = other.get("cost_per_1k_requests", 0.0) or 0.0
                o_lat = other.get("avg_latency_ms", 0.0) or 0.0

                # Other is at least as good in all 3 and strictly better in at least one
                better_or_equal = (
                    o_rouge >= c_rouge and o_cost <= c_cost and o_lat <= c_lat
                )
                strictly_better = (
                    o_rouge > c_rouge or o_cost < c_cost or o_lat < c_lat
                )
                if better_or_equal and strictly_better:
                    dominated = True
                    break

            if not dominated:
                pareto_front.append({
                    **candidate,
                    "is_pareto_optimal": True,
                })

        return pareto_front
