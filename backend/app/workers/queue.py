"""Unified async job queue supporting local ThreadPoolExecutor and Celery/Redis."""

from __future__ import annotations

import datetime
import logging
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.context import workspace_scope
from backend.app.database.models import AsyncJobRecord
from backend.app.database.session import engine
from backend.app.workers.celery_app import CELERY_AVAILABLE, celery_app

logger = logging.getLogger("benchlab.workers.queue")

JobHandler = Callable[[Dict[str, Any], Callable[[float, str], None]], Dict[str, Any]]


def _run_training_job(
    payload: Dict[str, Any],
    progress_cb: Callable[[float, str], None],
) -> Dict[str, Any]:
    from ml.core.config import LoRAConfig as LoRASchema, TrainingConfig
    from ml.training.trainer import FineTuningEngine

    progress_cb(10.0, "Initializing LoRA/QLoRA training configuration")
    method = payload.get("method", "lora")
    job_id = payload.get("training_job_id") or f"{method}_{uuid.uuid4().hex[:10]}"

    lora_cfg = LoRASchema(
        r=int(payload.get("rank", 16)),
        lora_alpha=int(payload.get("alpha", 32)),
        lora_dropout=float(payload.get("dropout", 0.05)),
        use_qlora=(method == "qlora"),
        target_modules=payload.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
    )
    train_cfg = TrainingConfig(
        job_id=job_id,
        base_model_id=payload.get("base_model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
        dataset_name=payload.get("dataset_name", "billsum"),
        output_dir=f"artifacts/checkpoints/{job_id}",
        lora=lora_cfg,
        learning_rate=float(payload.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(payload.get("batch_size", 2)),
        per_device_eval_batch_size=int(payload.get("batch_size", 2)),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 4)),
        num_train_epochs=int(payload.get("epochs", 1)),
        max_steps=int(payload.get("max_steps", 20)),
        max_seq_length=int(payload.get("max_seq_length", 512)),
    )
    progress_cb(30.0, f"Running {method.upper()} fine-tuning")
    result = FineTuningEngine().train(train_cfg)
    progress_cb(100.0, "Fine-tuning completed")
    return result if isinstance(result, dict) else {"training_job_id": job_id, "status": "completed"}


def _run_evaluation_job(
    payload: Dict[str, Any],
    progress_cb: Callable[[float, str], None],
) -> Dict[str, Any]:
    from ml.evaluation.evaluator import BaselineEvaluator

    progress_cb(15.0, "Loading dataset & model for evaluation")
    evaluator = BaselineEvaluator()
    result = evaluator.run_evaluation(
        model_id=payload.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
        dataset_name=payload.get("dataset_name", "billsum"),
        split=payload.get("split", "ca_test"),
        num_samples=int(payload.get("num_samples", 5)),
        max_input_length=int(payload.get("max_input_length", 1024)),
        max_output_tokens=int(payload.get("max_output_tokens", 128)),
        temperature=float(payload.get("temperature", 0.2)),
        top_p=float(payload.get("top_p", 0.95)),
        compare_with_api=payload.get("compare_with_api"),
    )
    progress_cb(100.0, "Evaluation completed")
    return result


def _run_quantization_job(
    payload: Dict[str, Any],
    progress_cb: Callable[[float, str], None],
) -> Dict[str, Any]:
    from ml.quantization.quantizer import QuantizationEngine

    progress_cb(15.0, "Starting model quantization")
    q_engine = QuantizationEngine()
    result = q_engine.run_quantization(
        base_model_id=payload.get("base_model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
        method=payload.get("quantization_method", "dynamic_int8"),
        target_precision=payload.get("target_precision", "int8"),
        eval_samples=int(payload.get("eval_samples", 5)),
        evaluate_after=bool(payload.get("evaluate_after_quant", True)),
    )
    progress_cb(100.0, "Quantization completed")
    return result


def _run_load_test_job(
    payload: Dict[str, Any],
    progress_cb: Callable[[float, str], None],
) -> Dict[str, Any]:
    from ml.benchmarking.load_tester import LoadTester

    progress_cb(15.0, "Running concurrent inference load test")
    tester = LoadTester()
    result = tester.run_load_test_sweep(
        model_id=payload.get("model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
        concurrency_levels=payload.get("concurrency_levels", [1, 4]),
        requests_per_level=int(payload.get("requests_per_level", 4)),
        prompt=payload.get("prompt", "Summarize the key provisions of this legislative bill."),
        max_output_tokens=int(payload.get("max_output_tokens", 64)),
    )
    progress_cb(100.0, "Load test completed")
    return result


class JobDispatcher:
    """Manages async ML/benchmark jobs via local ThreadPoolExecutor or Celery/Redis."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        workers = max_workers or int(os.getenv("BENCHLAB_WORKER_THREADS", "4"))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="benchlab-worker")
        self._lock = threading.Lock()
        self._futures: Dict[str, Future] = {}
        self._cancelled_ids: set[str] = set()
        self._handlers: Dict[str, JobHandler] = {
            "training": _run_training_job,
            "evaluation": _run_evaluation_job,
            "quantization": _run_quantization_job,
            "load_test": _run_load_test_job,
        }

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """Register or override a job handler (useful for custom pipelines or testing)."""
        self._handlers[job_type] = handler

    def active_backend(self) -> str:
        configured = os.getenv("JOB_BACKEND", "auto").strip().lower()
        if configured == "celery" and CELERY_AVAILABLE:
            return "celery"
        if configured == "auto" and CELERY_AVAILABLE:
            return "celery"
        return "local_threadpool"

    def submit_job(
        self,
        db: Session,
        job_type: str,
        payload: Dict[str, Any],
        workspace_id: str,
        user_id: Optional[str] = None,
    ) -> AsyncJobRecord:
        """Persist an AsyncJobRecord and enqueue execution on the active worker backend."""
        if job_type not in self._handlers:
            raise ValueError(
                f"Unsupported job_type '{job_type}'. Expected one of: {sorted(self._handlers.keys())}"
            )

        backend = self.active_backend()
        job_id = f"job_{job_type}_{uuid.uuid4().hex[:12]}"
        record = AsyncJobRecord(
            id=job_id,
            workspace_id=workspace_id,
            created_by_user_id=user_id,
            job_type=job_type,
            status="queued",
            backend=backend,
            payload_json=payload,
            progress_pct=0.0,
            current_step="Queued for execution",
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        bind_engine = db.get_bind()

        if backend == "celery" and celery_app is not None:
            task = celery_app.send_task("benchlab.execute_job", args=[job_id])
            record.celery_task_id = str(task.id)
            db.commit()
            db.refresh(record)
        else:
            future = self._executor.submit(self.execute_job_sync, job_id, bind_engine)
            with self._lock:
                self._futures[job_id] = future

        return record

    def execute_job_sync(
        self,
        job_id: str,
        bind_engine: Any = None,
    ) -> Dict[str, Any]:
        """Execute a persisted job synchronously inside the worker thread/process."""
        SessionMaker = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=bind_engine or engine,
        )
        db = SessionMaker()
        try:
            job = db.query(AsyncJobRecord).filter_by(id=job_id).first()
            if not job:
                return {"status": "missing", "job_id": job_id}

            with self._lock:
                is_cancelled = job_id in self._cancelled_ids or job.status == "cancelled"
            if is_cancelled:
                job.status = "cancelled"
                job.completed_at = datetime.datetime.utcnow()
                job.current_step = "Cancelled before execution"
                db.commit()
                return {"status": "cancelled", "job_id": job_id}

            job.status = "running"
            job.started_at = datetime.datetime.utcnow()
            job.progress_pct = 5.0
            job.current_step = "Starting worker execution"
            db.commit()

            handler = self._handlers[job.job_type]
            payload = dict(job.payload_json or {})

            def update_progress(pct: float, step: str) -> None:
                with self._lock:
                    if job_id in self._cancelled_ids:
                        raise InterruptedError("Job cancelled by user.")
                j = db.query(AsyncJobRecord).filter_by(id=job_id).first()
                if j:
                    j.progress_pct = round(min(max(pct, 0.0), 100.0), 1)
                    j.current_step = step
                    db.commit()

            with workspace_scope(job.workspace_id, job.created_by_user_id):
                result = handler(payload, update_progress)

            with self._lock:
                cancelled_after = job_id in self._cancelled_ids

            job = db.query(AsyncJobRecord).filter_by(id=job_id).first()
            if job:
                if cancelled_after:
                    job.status = "cancelled"
                    job.current_step = "Cancelled"
                else:
                    job.status = "completed"
                    job.progress_pct = 100.0
                    job.current_step = "Completed"
                    job.result_json = result
                job.completed_at = datetime.datetime.utcnow()
                db.commit()
            return result
        except InterruptedError as exc:
            db.rollback()
            job = db.query(AsyncJobRecord).filter_by(id=job_id).first()
            if job:
                job.status = "cancelled"
                job.error_message = str(exc)
                job.current_step = "Cancelled"
                job.completed_at = datetime.datetime.utcnow()
                db.commit()
            return {"status": "cancelled", "job_id": job_id}
        except Exception as exc:
            logger.error(f"Async job '{job_id}' failed: {exc}", exc_info=True)
            db.rollback()
            job = db.query(AsyncJobRecord).filter_by(id=job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(exc)
                job.current_step = "Failed"
                job.completed_at = datetime.datetime.utcnow()
                db.commit()
            return {"status": "failed", "job_id": job_id, "error": str(exc)}
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
                self._cancelled_ids.discard(job_id)
            db.close()

    def cancel_job(self, db: Session, job: AsyncJobRecord) -> AsyncJobRecord:
        """Request cancellation of a queued or running job."""
        if job.status in {"completed", "failed", "cancelled"}:
            return job

        with self._lock:
            self._cancelled_ids.add(job.id)
            fut = self._futures.get(job.id)
            if fut and fut.cancel():
                job.status = "cancelled"
                job.current_step = "Cancelled while queued"
                job.completed_at = datetime.datetime.utcnow()
                db.commit()
                db.refresh(job)
                return job

        if job.backend == "celery" and job.celery_task_id and celery_app is not None:
            celery_app.control.revoke(job.celery_task_id, terminate=False)

        job.status = "cancelled"
        job.current_step = "Cancellation requested"
        job.completed_at = datetime.datetime.utcnow()
        db.commit()
        db.refresh(job)
        return job

    def get_worker_stats(self, db: Session, workspace_id: str) -> Dict[str, Any]:
        """Return queue & worker telemetry for a workspace."""
        jobs = db.query(AsyncJobRecord).filter_by(workspace_id=workspace_id).all()
        counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0}
        for j in jobs:
            counts[j.status] = counts.get(j.status, 0) + 1
        with self._lock:
            active_futures = len(self._futures)
        return {
            "backend": self.active_backend(),
            "celery_configured": CELERY_AVAILABLE,
            "active_local_futures": active_futures,
            "workspace_job_counts": counts,
            "total_jobs": len(jobs),
        }


job_dispatcher = JobDispatcher()

