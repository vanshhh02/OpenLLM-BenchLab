"""Training and fine-tuning REST API endpoints."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.models import TrainingJobRecord
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    TrainingStartRequest,
    TrainingStatusResponse,
)
from ml.core.config import LoRAConfig as LoRASchema, TrainingConfig
from ml.training.trainer import FineTuningEngine

logger = logging.getLogger("benchlab.api.training")
router = APIRouter()


def _run_training_background(config: TrainingConfig) -> None:
    """Execute training in an asynchronous background worker thread."""
    try:
        engine = FineTuningEngine()
        engine.train(config)
    except Exception as e:
        logger.error(f"Background training failed for job {config.job_id}: {e}")
        # Mark failed in DB
        from backend.app.database.session import SessionLocal
        session = SessionLocal()
        try:
            job = session.query(TrainingJobRecord).filter_by(id=config.job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()


@router.post("/start", response_model=Dict[str, Any])
def start_training(
    req: TrainingStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Start an asynchronous LoRA or QLoRA fine-tuning run."""
    import time
    job_id = f"{req.method}_qwen_{int(time.time())}"

    lora_cfg = LoRASchema(
        r=req.rank,
        lora_alpha=req.alpha,
        lora_dropout=req.dropout,
        use_qlora=(req.method == "qlora"),
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    train_cfg = TrainingConfig(
        job_id=job_id,
        base_model_id=req.base_model_id,
        dataset_name=req.dataset_name,
        output_dir=f"artifacts/checkpoints/{job_id}",
        lora=lora_cfg,
        learning_rate=req.learning_rate,
        per_device_train_batch_size=req.batch_size,
        per_device_eval_batch_size=req.batch_size,
        gradient_accumulation_steps=req.gradient_accumulation_steps,
        num_train_epochs=req.epochs,
        max_steps=req.max_steps,
        max_seq_length=req.max_seq_length,
    )

    # Insert pending record in database
    job_record = TrainingJobRecord(
        id=job_id,
        base_model_id=req.base_model_id,
        method=req.method,
        status="pending",
        config_json=train_cfg.dict(),
    )
    db.add(job_record)
    db.commit()

    # Launch in background thread
    worker = threading.Thread(target=_run_training_background, args=(train_cfg,), daemon=True)
    worker.start()

    logger.info(f"Dispatched training job '{job_id}' in background worker.")

    return {
        "status": "started",
        "job_id": job_id,
        "method": req.method,
        "base_model_id": req.base_model_id,
        "max_steps": req.max_steps,
        "message": f"{req.method.upper()} fine-tuning job launched successfully.",
    }


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
def get_training_status(job_id: str, db: Session = Depends(get_db)):
    """Poll live progress, loss curves, and parameter statistics for a fine-tuning run."""
    job = db.query(TrainingJobRecord).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")

    return TrainingStatusResponse(
        id=job.id,
        base_model_id=job.base_model_id,
        method=job.method,
        status=job.status,
        config_json=job.config_json or {},
        train_loss=job.train_loss,
        val_loss=job.val_loss,
        trainable_parameters=job.trainable_parameters,
        total_parameters=job.total_parameters,
        trainable_percentage=job.trainable_percentage,
        duration_seconds=job.duration_seconds,
        checkpoint_path=job.checkpoint_path,
        checkpoint_size_mb=job.checkpoint_size_mb,
        peak_memory_mb=job.peak_memory_mb,
        metrics_history=job.metrics_history or [],
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("/history", response_model=List[TrainingStatusResponse])
def get_training_history(db: Session = Depends(get_db)):
    """Retrieve full history of all fine-tuning runs."""
    jobs = db.query(TrainingJobRecord).order_by(TrainingJobRecord.created_at.desc()).all()
    return [
        TrainingStatusResponse(
            id=job.id,
            base_model_id=job.base_model_id,
            method=job.method,
            status=job.status,
            config_json=job.config_json or {},
            train_loss=job.train_loss,
            val_loss=job.val_loss,
            trainable_parameters=job.trainable_parameters,
            total_parameters=job.total_parameters,
            trainable_percentage=job.trainable_percentage,
            duration_seconds=job.duration_seconds,
            checkpoint_path=job.checkpoint_path,
            checkpoint_size_mb=job.checkpoint_size_mb,
            peak_memory_mb=job.peak_memory_mb,
            metrics_history=job.metrics_history or [],
            error_message=job.error_message,
            created_at=job.created_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]

