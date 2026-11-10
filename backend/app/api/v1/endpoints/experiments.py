"""Experiment tracking and comparative analysis API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.models import ExperimentRecord
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    ExperimentCompareRequest,
    ExperimentItemResponse,
)

logger = logging.getLogger("benchlab.api.experiments")
router = APIRouter()


@router.get("", response_model=List[ExperimentItemResponse])
def list_experiments(db: Session = Depends(get_db)):
    """List all registered benchmark and fine-tuning experiments."""
    records = db.query(ExperimentRecord).order_by(ExperimentRecord.created_at.desc()).all()
    return [
        ExperimentItemResponse(
            id=r.id,
            name=r.name,
            model_id=r.model_id,
            category=r.category,
            rouge1=r.rouge1,
            rouge2=r.rouge2,
            rougeL=r.rougeL,
            bert_score=r.bert_score,
            avg_latency_ms=r.avg_latency_ms,
            tokens_per_second=r.tokens_per_second,
            peak_memory_mb=r.peak_memory_mb,
            cost_per_1k_requests=r.cost_per_1k_requests,
            status=r.status,
            parameters=r.parameters or {},
            metrics=r.metrics or {},
            mlflow_run_id=r.mlflow_run_id,
            created_at=r.created_at,
        )
        for r in records
    ]


@router.get("/{experiment_id}", response_model=ExperimentItemResponse)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Retrieve full details and metrics for an individual experiment."""
    r = db.query(ExperimentRecord).filter_by(id=experiment_id).first()
    if not r:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    return ExperimentItemResponse(
        id=r.id,
        name=r.name,
        model_id=r.model_id,
        category=r.category,
        rouge1=r.rouge1,
        rouge2=r.rouge2,
        rougeL=r.rougeL,
        bert_score=r.bert_score,
        avg_latency_ms=r.avg_latency_ms,
        tokens_per_second=r.tokens_per_second,
        peak_memory_mb=r.peak_memory_mb,
        cost_per_1k_requests=r.cost_per_1k_requests,
        status=r.status,
        parameters=r.parameters or {},
        metrics=r.metrics or {},
        mlflow_run_id=r.mlflow_run_id,
        created_at=r.created_at,
    )


@router.post("/compare", response_model=Dict[str, Any])
def compare_experiments(
    req: ExperimentCompareRequest,
    db: Session = Depends(get_db),
):
    """Perform side-by-side multi-metric comparative analysis across selected experiments."""
    records = db.query(ExperimentRecord).filter(ExperimentRecord.id.in_(req.experiment_ids)).all()
    if not records:
        raise HTTPException(status_code=404, detail="No matching experiments found for comparison.")

    experiments = []
    for r in records:
        experiments.append({
            "id": r.id,
            "name": r.name,
            "model_id": r.model_id,
            "category": r.category,
            "rouge1": r.rouge1,
            "rouge2": r.rouge2,
            "rougeL": r.rougeL,
            "bert_score": r.bert_score,
            "avg_latency_ms": r.avg_latency_ms,
            "tokens_per_second": r.tokens_per_second,
            "peak_memory_mb": r.peak_memory_mb,
            "cost_per_1k_requests": r.cost_per_1k_requests,
            "parameters": r.parameters or {},
        })

    # Determine metric leaders
    best_rougel = max((e for e in experiments if e["rougeL"] is not None), key=lambda x: x["rougeL"], default=None)
    best_latency = min((e for e in experiments if e["avg_latency_ms"] is not None), key=lambda x: x["avg_latency_ms"], default=None)
    best_memory = min((e for e in experiments if e["peak_memory_mb"] is not None), key=lambda x: x["peak_memory_mb"], default=None)
    best_throughput = max((e for e in experiments if e["tokens_per_second"] is not None), key=lambda x: x["tokens_per_second"], default=None)

    return {
        "experiments": experiments,
        "leaders": {
            "best_quality": best_rougel["id"] if best_rougel else None,
            "lowest_latency": best_latency["id"] if best_latency else None,
            "lowest_memory": best_memory["id"] if best_memory else None,
            "highest_throughput": best_throughput["id"] if best_throughput else None,
        },
    }

