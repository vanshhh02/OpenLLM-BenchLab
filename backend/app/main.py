"""OpenLLM BenchLab FastAPI Application Entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.endpoints import (
    benchmark,
    datasets,
    evaluation,
    experiments,
    inference,
    models,
    quantization,
    training,
)
from backend.app.core.hardware import detect_hardware
from backend.app.database.session import init_db
from backend.app.schemas.api_schemas import HardwareInfoResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchlab.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info("Initializing OpenLLM BenchLab database tables...")
    init_db()
    hw = detect_hardware()
    logger.info(
        f"Hardware environment detected: {hw.accelerator_name} "
        f"({hw.accelerator_type.upper()}), Total RAM: {hw.total_ram_gb} GB"
    )
    yield
    logger.info("OpenLLM BenchLab server shutting down...")


app = FastAPI(
    title="OpenLLM BenchLab API",
    description=(
        "Production-grade platform for open-weight LLM optimization, fine-tuning, "
        "quantization, inference benchmarking, and cost analysis against frontier models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Configuration for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(training.router, prefix="/api/training", tags=["Fine-Tuning"])
app.include_router(quantization.router, prefix="/api/quantization", tags=["Quantization"])
app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])
app.include_router(benchmark.router, prefix="/api/benchmark", tags=["Benchmarking"])


@app.get("/api/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Service health and readiness check."""
    return {
        "status": "healthy",
        "service": "openllm-benchlab",
        "version": "1.0.0",
    }


@app.get("/api/hardware", response_model=HardwareInfoResponse, tags=["System"])
def get_hardware_info():
    """Retrieve detailed hardware profile and accelerator capabilities."""
    profile = detect_hardware()
    return HardwareInfoResponse(**profile.to_dict())

