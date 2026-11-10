"""Model Quantization REST API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.database.models import QuantizationRecord
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    QuantizationResultItem,
    QuantizationRunRequest,
)
from ml.quantization.quantizer import QuantizationEngine

logger = logging.getLogger("benchlab.api.quantization")
router = APIRouter()


@router.post("/run", response_model=QuantizationResultItem)
def run_quantization(
    req: QuantizationRunRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Execute model quantization using the specified method and precision format."""
    try:
        engine = QuantizationEngine()
        result = engine.run_quantization(
            base_model_id=req.base_model_id,
            method=req.quantization_method,
            target_precision=req.target_precision,
            eval_samples=req.eval_samples,
            evaluate_after=req.evaluate_after_quant,
        )

        return QuantizationResultItem(
            id=result["id"],
            base_model_id=result["base_model_id"],
            quantized_model_id=result["quantized_model_id"],
            method=result["method"],
            target_precision=result["target_precision"],
            original_memory_mb=result.get("original_memory_mb", 0.0),
            quantized_memory_mb=result.get("quantized_memory_mb", 0.0),
            memory_reduction_pct=result.get("memory_reduction_pct", 0.0),
            baseline_rougel=result.get("baseline_rougel"),
            quantized_rougel=result.get("quantized_rougel"),
            rougel_delta=result.get("rougel_delta"),
            baseline_latency_ms=result.get("baseline_latency_ms"),
            quantized_latency_ms=result.get("quantized_latency_ms"),
            speedup_ratio=result.get("speedup_ratio"),
            status=result.get("status", "completed"),
            artifact_path=result.get("artifact_path"),
            notes=result.get("notes"),
        )
    except Exception as e:
        logger.error(f"Quantization execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results", response_model=List[QuantizationResultItem])
def list_quantization_results(
    base_model_id: str = Query("Qwen/Qwen2.5-0.5B-Instruct", description="Filter by base model"),
    db: Session = Depends(get_db),
) -> Any:
    """List all completed or registered quantization benchmark records."""
    records = (
        db.query(QuantizationRecord)
        .filter(QuantizationRecord.base_model_id == base_model_id)
        .order_by(QuantizationRecord.created_at.desc())
        .all()
    )

    items = []
    for r in records:
        items.append(
            QuantizationResultItem(
                id=r.id,
                base_model_id=r.base_model_id,
                quantized_model_id=r.quantized_model_id,
                method=r.method,
                target_precision=r.target_precision,
                original_memory_mb=r.original_memory_mb or 0.0,
                quantized_memory_mb=r.quantized_memory_mb or 0.0,
                memory_reduction_pct=r.memory_reduction_pct or 0.0,
                baseline_rougel=r.baseline_rougel,
                quantized_rougel=r.quantized_rougel,
                rougel_delta=r.rougel_delta,
                baseline_latency_ms=r.baseline_latency_ms,
                quantized_latency_ms=r.quantized_latency_ms,
                speedup_ratio=r.speedup_ratio,
                status=r.status or "completed",
                artifact_path=r.artifact_path,
                notes=r.notes,
                created_at=r.created_at,
            )
        )
    return items


@router.get("/comparison", response_model=List[Dict[str, Any]])
def get_precision_comparison(
    base_model_id: str = Query("Qwen/Qwen2.5-0.5B-Instruct", description="Base model identifier"),
    db: Session = Depends(get_db),
) -> Any:
    """Get cross-precision trade-off comparison matrix (FP16 vs INT8 vs INT4 vs GGUF)."""
    engine = QuantizationEngine()
    matrix = engine.build_precision_comparison_matrix(base_model_id=base_model_id)
    return matrix


@router.get("/{quant_id}", response_model=QuantizationResultItem)
def get_quantization_detail(
    quant_id: str,
    db: Session = Depends(get_db),
) -> Any:
    """Retrieve detailed metadata and metrics for a specific quantization run."""
    record = db.query(QuantizationRecord).filter(QuantizationRecord.id == quant_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Quantization run '{quant_id}' not found")

    return QuantizationResultItem(
        id=record.id,
        base_model_id=record.base_model_id,
        quantized_model_id=record.quantized_model_id,
        method=record.method,
        target_precision=record.target_precision,
        original_memory_mb=record.original_memory_mb or 0.0,
        quantized_memory_mb=record.quantized_memory_mb or 0.0,
        memory_reduction_pct=record.memory_reduction_pct or 0.0,
        baseline_rougel=record.baseline_rougel,
        quantized_rougel=record.quantized_rougel,
        rougel_delta=record.rougel_delta,
        baseline_latency_ms=record.baseline_latency_ms,
        quantized_latency_ms=record.quantized_latency_ms,
        speedup_ratio=record.speedup_ratio,
        status=record.status or "completed",
        artifact_path=record.artifact_path,
        notes=record.notes,
        created_at=record.created_at,
    )

