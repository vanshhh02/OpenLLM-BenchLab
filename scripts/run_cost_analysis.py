"""CLI Script to execute Phase 6 Cost Analysis & Frontier ROI Modeling."""

import json
import logging
from ml.cost_analysis.calculator import CostAnalysisEngine, CostCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    print("=================================================================")
    print(" OpenLLM BenchLab: Cost Analysis & Frontier ROI Modeling ")
    print("=================================================================")

    # Compare 500,000 monthly legal summarization requests:
    # Option A: OpenAI GPT-4o-mini ($0.150 / $0.600 per M)
    # Option B: Dedicated Cloud RTX 4090 ($85/mo + electricity) running Qwen2.5-0.5B LoRA
    monthly_volume = 500000
    avg_input = 1500
    avg_output = 250

    print(f"\nScenario: {monthly_volume:,} requests/month")
    print(f"Document Tokens: {avg_input} prompt / {avg_output} summary\n")

    result = CostCalculator.analyze(
        monthly_requests=monthly_volume,
        avg_input_tokens=avg_input,
        avg_output_tokens=avg_output,
        api_name="OpenAI GPT-4o-mini",
        api_input_cost_per_m=0.150,
        api_output_cost_per_m=0.600,
        server_monthly_rental_usd=85.0,
        hardware_purchase_cost_usd=0.0,
        server_power_watts=300.0,
        electricity_kwh_cost_usd=0.15,
        operational_overhead_monthly_usd=25.0,
        tokens_per_second_local=75.0,
    )

    print("------------------- COMMERCIAL API COSTS -------------------")
    print(f"API Provider:               {result.api_name}")
    print(f"Cost per Request:           ${result.api_cost_per_request_usd:.6f}")
    print(f"Cost per 1,000 Requests:    ${result.api_cost_per_1k_usd:.4f}")
    print(f"Monthly API Bill:           ${result.monthly_api_cost_usd:,.2f}")
    print(f"Annual Projected API Cost:  ${result.annual_api_cost_usd:,.2f}")

    print("\n----------------- SELF-HOSTED CLOUD INFRA -----------------")
    print(f"Server Monthly Rental:      ${result.server_monthly_rental_usd:,.2f}")
    print(f"Operational Overhead:       ${result.operational_overhead_monthly_usd:,.2f}")
    print(f"Monthly Power Consumed:     {result.electricity_kwh_monthly:.1f} kWh (${result.local_electricity_cost_monthly_usd:.2f})")
    print(f"Total Monthly Hosting Cost: ${result.monthly_local_cost_usd:,.2f}")
    print(f"Effective Cost / Request:   ${result.local_cost_per_request_usd:.6f}")
    print(f"Effective Cost / 1k Reqs:   ${result.local_cost_per_1k_usd:.4f}")

    print("\n------------------- ROI & BREAK-EVEN -----------------------")
    print(f"Monthly Net Savings:        ${result.monthly_savings_usd:,.2f} ({result.savings_percentage:.1f}%)")
    print(f"Annual Projected Savings:   ${result.annual_savings_usd:,.2f}")
    print(f"Break-Even Status:          {result.break_even_status.upper()}")
    if result.break_even_requests:
        print(f"Break-Even Volume:          {int(result.break_even_requests):,} requests/month")

    print("\n----------------- SENSITIVITY TABLE ------------------------")
    print(f"{'Volume':<12} | {'API Cost ($)':<12} | {'Hosting ($)':<12} | {'Savings ($)':<12} | {'Winning Model'}")
    print("-" * 68)
    for pt in result.sensitivity_curve:
        print(
            f"{pt['monthly_requests']:<12,d} | "
            f"${pt['api_monthly_cost_usd']:<11.2f} | "
            f"${pt['local_monthly_cost_usd']:<11.2f} | "
            f"${pt['savings_usd']:<11.2f} | "
            f"{pt['recommended_architecture']}"
        )


if __name__ == "__main__":
    main()
