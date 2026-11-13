"""FastAPI Endpoints for Cost Analysis and Frontier ROI Modeling."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.security import RequestContext, get_request_context
from backend.app.database.models import EvaluationRecord, ExperimentRecord
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    CostCalculationRequest,
    CostCalculationResponse,
)
from ml.cost_analysis.calculator import (
    CostAnalysisEngine,
    CostCalculator,
    FRONTIER_API_PRESETS,
    HARDWARE_PRESETS,
)

logger = logging.getLogger("benchlab.api.cost")

router = APIRouter()


@router.post("/calculate", response_model=CostCalculationResponse)
def calculate_cost_and_roi(
    req: CostCalculationRequest,
    ctx: RequestContext = Depends(get_request_context),
):
    """Compute complete comparative cost analysis, break-even requests, and sensitivity curve."""
    result = CostCalculator.analyze(
        monthly_requests=req.monthly_requests,
        avg_input_tokens=req.avg_input_tokens,
        avg_output_tokens=req.avg_output_tokens,
        api_name=req.api_name,
        api_input_cost_per_m=req.api_input_cost_per_m,
        api_output_cost_per_m=req.api_output_cost_per_m,
        server_monthly_rental_usd=req.server_monthly_rental_usd,
        hardware_purchase_cost_usd=req.hardware_purchase_cost_usd,
        amortization_months=req.amortization_months,
        server_power_watts=req.server_power_watts,
        electricity_kwh_cost_usd=req.electricity_kwh_cost_usd,
        operational_overhead_monthly_usd=req.operational_overhead_monthly_usd,
        tokens_per_second_local=req.tokens_per_second_local,
    )

    return CostCalculationResponse(
        monthly_requests=result.monthly_requests,
        avg_input_tokens=result.avg_input_tokens,
        avg_output_tokens=result.avg_output_tokens,
        api_cost_per_request_usd=result.api_cost_per_request_usd,
        api_cost_per_1k_usd=result.api_cost_per_1k_usd,
        monthly_api_cost_usd=result.monthly_api_cost_usd,
        local_fixed_cost_monthly_usd=result.local_fixed_cost_monthly_usd,
        local_electricity_cost_monthly_usd=result.local_electricity_cost_monthly_usd,
        local_variable_cost_monthly_usd=result.local_variable_cost_monthly_usd,
        monthly_local_cost_usd=result.monthly_local_cost_usd,
        local_cost_per_request_usd=result.local_cost_per_request_usd,
        local_cost_per_1k_usd=result.local_cost_per_1k_usd,
        break_even_requests=result.break_even_requests,
        break_even_status=result.break_even_status,
        monthly_savings_usd=result.monthly_savings_usd,
        savings_percentage=result.savings_percentage,
        sensitivity_curve=result.sensitivity_curve,
    )


@router.get("/frontier-presets")
def list_frontier_presets() -> Dict[str, Any]:
    """Retrieve preset rates and parameters for industry frontier LLM APIs."""
    return FRONTIER_API_PRESETS


@router.get("/hardware-presets")
def list_hardware_presets() -> Dict[str, Any]:
    """Retrieve preset dedicated hardware costs and compute configurations."""
    return HARDWARE_PRESETS


@router.get("/pareto-analysis")
def get_pareto_analysis(
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Analyze all registered model evaluations to construct Quality vs Cost vs Latency Pareto frontier."""
    experiments = db.query(ExperimentRecord).filter_by(workspace_id=ctx.workspace_id).all()
    records = []
    for exp in experiments:
        records.append({
            "id": exp.id,
            "name": exp.name,
            "model_id": exp.model_id,
            "category": exp.category,
            "rougeL": exp.rougeL or 0.0,
            "avg_latency_ms": exp.avg_latency_ms or 0.0,
            "tokens_per_second": exp.tokens_per_second or 0.0,
            "cost_per_1k_requests": exp.cost_per_1k_requests or 0.0,
        })

    pareto_models = CostAnalysisEngine.compute_pareto_frontier(records)

    return {
        "total_experiments_evaluated": len(records),
        "pareto_optimal_count": len(pareto_models),
        "pareto_frontier": pareto_models,
        "all_records": records,
    }

