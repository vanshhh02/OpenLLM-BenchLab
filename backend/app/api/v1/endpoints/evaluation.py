"""Evaluation API endpoints for baseline and comparative benchmarking."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.models import EvaluationRecord, ExperimentRecord
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    EvaluationCompareResponse,
    EvaluationResultItem,
    EvaluationRunRequest,
)
from ml.evaluation.evaluator import BaselineEvaluator

logger = logging.getLogger("benchlab.api.evaluation")
router = APIRouter()


@router.get("/results", response_model=List[EvaluationResultItem])
def get_evaluation_results(db: Session = Depends(get_db)):
    """Retrieve all evaluation benchmarks recorded in the database."""
    records = db.query(EvaluationRecord).order_by(EvaluationRecord.created_at.desc()).all()
    return [
        EvaluationResultItem(
            id=r.id,
            experiment_id=r.experiment_id,
            model_id=r.model_id,
            dataset_name=r.dataset_name,
            split=r.split,
            num_samples=r.num_samples,
            rouge1=r.rouge1,
            rouge2=r.rouge2,
            rougeL=r.rougeL,
            bert_score=r.bert_score,
            factual_consistency=r.factual_consistency,
            avg_input_length=r.avg_input_length,
            avg_output_length=r.avg_output_length,
            avg_latency_ms=r.avg_latency_ms,
            avg_ttft_ms=r.avg_ttft_ms,
            tokens_per_second=r.tokens_per_second,
            peak_memory_mb=r.peak_memory_mb,
            cost_per_request_usd=r.cost_per_request_usd,
            hardware_info=r.hardware_info or {},
            generations_sample=r.generations_sample or [],
            created_at=r.created_at,
        )
        for r in records
    ]


@router.post("/run", response_model=Dict[str, Any])
def run_evaluation_endpoint(
    req: EvaluationRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Trigger an evaluation run on BillSum test split with optional commercial API comparison."""
    # Execute evaluation synchronously or in background
    evaluator = BaselineEvaluator()
    try:
        result = evaluator.run_evaluation(
            model_id=req.model_id,
            dataset_name=req.dataset_name,
            split=req.split,
            num_samples=req.num_samples,
            max_input_length=req.max_input_length,
            max_output_tokens=req.max_output_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            compare_with_api=req.compare_with_api,
        )
        return {
            "status": "completed",
            "evaluation_id": result["evaluation_id"],
            "model_id": result["model_id"],
            "quality": result["quality"],
            "operational": result["operational"],
            "frontier_comparison": result.get("frontier_comparison"),
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare", response_model=EvaluationCompareResponse)
def compare_evaluations(
    ids: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Compare evaluation results across multiple models and benchmarks."""
    query = db.query(EvaluationRecord)
    if ids:
        id_list = [i.strip() for i in ids.split(",") if i.strip()]
        query = query.filter(EvaluationRecord.id.in_(id_list))
    records = query.order_by(EvaluationRecord.created_at.desc()).limit(10).all()

    if not records:
        return EvaluationCompareResponse(
            evaluations=[],
            metrics_comparison={},
            relative_speedup={},
            memory_savings_pct={},
        )

    eval_items = [
        EvaluationResultItem(
            id=r.id,
            experiment_id=r.experiment_id,
            model_id=r.model_id,
            dataset_name=r.dataset_name,
            split=r.split,
            num_samples=r.num_samples,
            rouge1=r.rouge1,
            rouge2=r.rouge2,
            rougeL=r.rougeL,
            bert_score=r.bert_score,
            factual_consistency=r.factual_consistency,
            avg_input_length=r.avg_input_length,
            avg_output_length=r.avg_output_length,
            avg_latency_ms=r.avg_latency_ms,
            avg_ttft_ms=r.avg_ttft_ms,
            tokens_per_second=r.tokens_per_second,
            peak_memory_mb=r.peak_memory_mb,
            cost_per_request_usd=r.cost_per_request_usd,
            hardware_info=r.hardware_info or {},
            generations_sample=r.generations_sample or [],
            created_at=r.created_at,
        )
        for r in records
    ]

    metrics_comp: Dict[str, Dict[str, float]] = {}
    speedup: Dict[str, float] = {}
    memory_savings: Dict[str, float] = {}

    baseline_rec = eval_items[0]
    for item in eval_items:
        metrics_comp[item.model_id] = {
            "rouge1": item.rouge1,
            "rouge2": item.rouge2,
            "rougeL": item.rougeL,
            "latency_ms": item.avg_latency_ms,
            "tokens_per_second": item.tokens_per_second,
            "peak_memory_mb": item.peak_memory_mb,
        }
        if baseline_rec.avg_latency_ms > 0:
            speedup[item.model_id] = round(baseline_rec.avg_latency_ms / max(1.0, item.avg_latency_ms), 2)
        if baseline_rec.peak_memory_mb > 0:
            savings = (baseline_rec.peak_memory_mb - item.peak_memory_mb) / baseline_rec.peak_memory_mb * 100
            memory_savings[item.model_id] = round(max(0.0, savings), 1)

    return EvaluationCompareResponse(
        evaluations=eval_items,
        metrics_comparison=metrics_comp,
        relative_speedup=speedup,
        memory_savings_pct=memory_savings,
    )

