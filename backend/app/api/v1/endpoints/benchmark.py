"""Benchmarking and Concurrent Load Testing REST API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.core.security import RequestContext, get_request_context, require_min_role
from backend.app.database.models import BenchmarkRecord
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    BenchmarkResultItem,
    LoadTestRequest,
)
from ml.benchmarking.load_tester import LoadTester

logger = logging.getLogger("benchlab.api.benchmark")
router = APIRouter()


@router.post("/load-test", response_model=Dict[str, Any])
def run_load_test(
    req: LoadTestRequest,
    ctx: RequestContext = Depends(get_request_context),
) -> Any:
    """Execute concurrent load test sweep across specified concurrency levels."""
    require_min_role(ctx, "member")
    try:
        tester = LoadTester()
        result = tester.run_load_test_sweep(
            model_id=req.model_id,
            concurrency_levels=req.concurrency_levels,
            requests_per_level=req.requests_per_level,
            prompt=req.prompt,
            max_output_tokens=req.max_output_tokens,
        )
        return result
    except Exception as e:
        logger.error(f"Load testing execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results", response_model=List[BenchmarkResultItem])
def list_benchmark_results(
    model_id: str = Query("Qwen/Qwen2.5-0.5B-Instruct", description="Filter by model identifier"),
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
) -> Any:
    """Query past concurrent load test results and latency percentile curves in the active workspace."""
    records = (
        db.query(BenchmarkRecord)
        .filter(
            BenchmarkRecord.workspace_id == ctx.workspace_id,
            BenchmarkRecord.model_id == model_id,
        )
        .order_by(BenchmarkRecord.concurrency_level.asc(), BenchmarkRecord.created_at.desc())
        .all()
    )

    items = []
    for r in records:
        items.append(
            BenchmarkResultItem(
                id=r.id,
                model_id=r.model_id,
                benchmark_type=r.benchmark_type or "load_test",
                concurrency_level=r.concurrency_level or 1,
                num_requests=r.num_requests or 0,
                successful_requests=r.successful_requests or 0,
                failed_requests=r.failed_requests or 0,
                p50_latency_ms=r.p50_latency_ms or 0.0,
                p95_latency_ms=r.p95_latency_ms or 0.0,
                p99_latency_ms=r.p99_latency_ms or 0.0,
                avg_latency_ms=r.avg_latency_ms or 0.0,
                avg_ttft_ms=r.avg_ttft_ms or 0.0,
                requests_per_second=r.requests_per_second or 0.0,
                tokens_per_second=r.tokens_per_second or 0.0,
                peak_memory_mb=r.peak_memory_mb or 0.0,
                engine=r.engine or "pytorch",
                kv_cache_usage_mb=r.kv_cache_usage_mb,
                metrics_json=r.metrics_json or {},
                created_at=r.created_at,
            )
        )
    return items


@router.get("/{benchmark_id}", response_model=BenchmarkResultItem)
def get_benchmark_detail(
    benchmark_id: str,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
) -> Any:
    """Retrieve detailed operational metrics for a specific benchmark level run."""
    r = (
        db.query(BenchmarkRecord)
        .filter(
            BenchmarkRecord.id == benchmark_id,
            BenchmarkRecord.workspace_id == ctx.workspace_id,
        )
        .first()
    )
    if not r:
        raise HTTPException(status_code=404, detail=f"Benchmark '{benchmark_id}' not found")

    return BenchmarkResultItem(
        id=r.id,
        model_id=r.model_id,
        benchmark_type=r.benchmark_type or "load_test",
        concurrency_level=r.concurrency_level or 1,
        num_requests=r.num_requests or 0,
        successful_requests=r.successful_requests or 0,
        failed_requests=r.failed_requests or 0,
        p50_latency_ms=r.p50_latency_ms or 0.0,
        p95_latency_ms=r.p95_latency_ms or 0.0,
        p99_latency_ms=r.p99_latency_ms or 0.0,
        avg_latency_ms=r.avg_latency_ms or 0.0,
        avg_ttft_ms=r.avg_ttft_ms or 0.0,
        requests_per_second=r.requests_per_second or 0.0,
        tokens_per_second=r.tokens_per_second or 0.0,
        peak_memory_mb=r.peak_memory_mb or 0.0,
        engine=r.engine or "pytorch",
        kv_cache_usage_mb=r.kv_cache_usage_mb,
        metrics_json=r.metrics_json or {},
        created_at=r.created_at,
    )

