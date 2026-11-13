"""Async job queue & background worker management endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.core.security import RequestContext, get_request_context, require_min_role
from backend.app.database.models import AsyncJobRecord
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import AsyncJobResponse, AsyncJobSubmitRequest
from backend.app.workers.queue import job_dispatcher

router = APIRouter()


def _to_job_response(job: AsyncJobRecord) -> AsyncJobResponse:
    return AsyncJobResponse(
        id=job.id,
        workspace_id=job.workspace_id,
        created_by_user_id=job.created_by_user_id,
        job_type=job.job_type,
        status=job.status,
        backend=job.backend,
        celery_task_id=job.celery_task_id,
        payload_json=job.payload_json or {},
        result_json=job.result_json,
        progress_pct=job.progress_pct or 0.0,
        current_step=job.current_step,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get("/worker/health", response_model=Dict[str, Any])
def get_worker_health(
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
):
    """Return active async worker backend (local_threadpool or celery) and queue counts."""
    return job_dispatcher.get_worker_stats(db, ctx.workspace_id)


@router.post("/submit", response_model=AsyncJobResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_async_job(
    req: AsyncJobSubmitRequest,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
):
    """Enqueue an asynchronous ML job (training, evaluation, quantization, load_test)."""
    require_min_role(ctx, "member")
    try:
        record = job_dispatcher.submit_job(
            db=db,
            job_type=req.job_type,
            payload=req.payload,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
        )
        return _to_job_response(record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=List[AsyncJobResponse])
def list_async_jobs(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    job_type: Optional[str] = Query(default=None),
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
):
    """List background jobs scoped to the active workspace."""
    query = db.query(AsyncJobRecord).filter_by(workspace_id=ctx.workspace_id)
    if status_filter:
        query = query.filter_by(status=status_filter)
    if job_type:
        query = query.filter_by(job_type=job_type)
    jobs = query.order_by(AsyncJobRecord.created_at.desc()).limit(100).all()
    return [_to_job_response(j) for j in jobs]


@router.get("/{job_id}", response_model=AsyncJobResponse)
def get_async_job(
    job_id: str,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
):
    """Fetch status, progress, and output of an async job in the active workspace."""
    job = db.query(AsyncJobRecord).filter_by(id=job_id, workspace_id=ctx.workspace_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _to_job_response(job)


@router.post("/{job_id}/cancel", response_model=AsyncJobResponse)
def cancel_async_job(
    job_id: str,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
):
    """Cancel a queued or running async job in the active workspace."""
    require_min_role(ctx, "member")
    job = db.query(AsyncJobRecord).filter_by(id=job_id, workspace_id=ctx.workspace_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    updated = job_dispatcher.cancel_job(db, job)
    return _to_job_response(updated)

