"""Model registry API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.hardware import detect_hardware
from backend.app.database.models import ModelRecord
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    ModelItemResponse,
    ModelLoadRequest,
)
from ml.models.base_loader import ModelManager

logger = logging.getLogger("benchlab.api.models")
router = APIRouter()

PRECONFIGURED_MODELS = [
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen 2.5 0.5B Instruct",
        "provider": "huggingface",
        "architecture": "Qwen2ForCausalLM",
        "parameter_count_b": 0.49,
        "context_window": 32768,
        "precision": "fp16",
        "is_commercial_api": False,
        "is_fine_tuned": False,
        "is_quantized": False,
        "license": "Apache-2.0",
        "hardware_compatibility": {
            "supported_devices": ["mps", "cuda", "cpu"],
            "min_ram_gb": 2.0,
            "min_vram_gb": 1.2,
            "recommended": True,
        },
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen 2.5 1.5B Instruct",
        "provider": "huggingface",
        "architecture": "Qwen2ForCausalLM",
        "parameter_count_b": 1.54,
        "context_window": 32768,
        "precision": "fp16",
        "is_commercial_api": False,
        "is_fine_tuned": False,
        "is_quantized": False,
        "license": "Apache-2.0",
        "hardware_compatibility": {
            "supported_devices": ["mps", "cuda", "cpu"],
            "min_ram_gb": 4.5,
            "min_vram_gb": 3.2,
        },
    },
    {
        "id": "google/gemma-2-2b-it",
        "name": "Gemma 2 2B Instruct",
        "provider": "huggingface",
        "architecture": "Gemma2ForCausalLM",
        "parameter_count_b": 2.61,
        "context_window": 8192,
        "precision": "bfloat16",
        "is_commercial_api": False,
        "is_fine_tuned": False,
        "is_quantized": False,
        "license": "Gemma Terms of Use",
        "hardware_compatibility": {
            "supported_devices": ["mps", "cuda", "cpu"],
            "min_ram_gb": 6.0,
            "min_vram_gb": 5.5,
        },
    },
    {
        "id": "openai/gpt-4o-mini",
        "name": "OpenAI GPT-4o Mini (Frontier API)",
        "provider": "openai",
        "architecture": "Commercial Frontier API",
        "parameter_count_b": 0.0,
        "context_window": 128000,
        "precision": "fp16-equivalent",
        "is_commercial_api": True,
        "is_fine_tuned": False,
        "is_quantized": False,
        "license": "Proprietary Commercial",
        "hardware_compatibility": {
            "supported_devices": ["cloud-api"],
            "min_ram_gb": 0.1,
            "min_vram_gb": 0.0,
        },
    },
    {
        "id": "google/gemini-1.5-flash",
        "name": "Google Gemini 1.5 Flash (Frontier API)",
        "provider": "google",
        "architecture": "Commercial Frontier API",
        "parameter_count_b": 0.0,
        "context_window": 1048576,
        "precision": "fp16-equivalent",
        "is_commercial_api": True,
        "is_fine_tuned": False,
        "is_quantized": False,
        "license": "Proprietary Commercial",
        "hardware_compatibility": {
            "supported_devices": ["cloud-api"],
            "min_ram_gb": 0.1,
            "min_vram_gb": 0.0,
        },
    },
]


@router.get("", response_model=List[ModelItemResponse])
def list_models(db: Session = Depends(get_db)):
    """List all registered open-weight models and commercial frontier models."""
    db_models = db.query(ModelRecord).all()
    if not db_models:
        # Populate initial default models
        for item in PRECONFIGURED_MODELS:
            m = ModelRecord(
                id=item["id"],
                name=item["name"],
                provider=item["provider"],
                architecture=item["architecture"],
                parameter_count_b=item["parameter_count_b"],
                context_window=item["context_window"],
                precision=item["precision"],
                is_commercial_api=item["is_commercial_api"],
                is_fine_tuned=item["is_fine_tuned"],
                is_quantized=item["is_quantized"],
                hardware_compatibility=item["hardware_compatibility"],
                license=item["license"],
            )
            db.add(m)
        db.commit()
        db_models = db.query(ModelRecord).all()

    return [
        ModelItemResponse(
            id=m.id,
            name=m.name,
            provider=m.provider,
            architecture=m.architecture,
            parameter_count_b=m.parameter_count_b,
            context_window=m.context_window,
            precision=m.precision,
            is_commercial_api=m.is_commercial_api,
            is_fine_tuned=m.is_fine_tuned,
            is_quantized=m.is_quantized,
            base_model_id=m.base_model_id,
            adapter_path=m.adapter_path,
            quantization_method=m.quantization_method,
            hardware_compatibility=m.hardware_compatibility or {},
            license=m.license,
            created_at=m.created_at,
        )
        for m in db_models
    ]


@router.get("/{model_id:path}", response_model=ModelItemResponse)
def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get detailed metadata for a specific model."""
    m = db.query(ModelRecord).filter_by(id=model_id).first()
    if not m:
        # Check PRECONFIGURED_MODELS
        for item in PRECONFIGURED_MODELS:
            if item["id"] == model_id:
                return ModelItemResponse(**item)
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    return ModelItemResponse(
        id=m.id,
        name=m.name,
        provider=m.provider,
        architecture=m.architecture,
        parameter_count_b=m.parameter_count_b,
        context_window=m.context_window,
        precision=m.precision,
        is_commercial_api=m.is_commercial_api,
        is_fine_tuned=m.is_fine_tuned,
        is_quantized=m.is_quantized,
        base_model_id=m.base_model_id,
        adapter_path=m.adapter_path,
        quantization_method=m.quantization_method,
        hardware_compatibility=m.hardware_compatibility or {},
        license=m.license,
        created_at=m.created_at,
    )


@router.post("/load", response_model=Dict[str, Any])
def load_model_endpoint(req: ModelLoadRequest):
    """Load model weights into GPU/Metal VRAM or system RAM."""
    hw = detect_hardware()
    try:
        model, tokenizer, meta = ModelManager.load_model_and_tokenizer(
            model_id=req.model_id,
            precision=req.precision or "fp16",
            device=req.device or "auto",
        )
        return {
            "status": "loaded",
            "model_id": req.model_id,
            "device": meta.get("device"),
            "load_time_sec": meta.get("load_time_sec"),
            "parameter_count_b": meta.get("parameter_count_b"),
            "hardware_device": hw.accelerator_name,
        }
    except Exception as e:
        logger.error(f"Error loading model {req.model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

