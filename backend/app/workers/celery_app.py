"""Optional Celery + Redis worker configuration for distributed task execution."""

from __future__ import annotations

import os
from typing import Optional

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL", "")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND") or CELERY_BROKER_URL

try:
    from celery import Celery  # type: ignore

    CELERY_AVAILABLE = bool(CELERY_BROKER_URL)
except ImportError:
    Celery = None  # type: ignore
    CELERY_AVAILABLE = False

celery_app: Optional[object] = None

if Celery is not None and CELERY_BROKER_URL:
    celery_app = Celery(
        "openllm_benchlab",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND or "rpc://",
    )
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
    )

    @celery_app.task(name="benchlab.execute_job", bind=True)
    def execute_celery_job(self, job_id: str) -> dict:
        """Celery worker entrypoint that runs a persisted AsyncJobRecord."""
        from backend.app.workers.queue import job_dispatcher

        return job_dispatcher.execute_job_sync(job_id)

