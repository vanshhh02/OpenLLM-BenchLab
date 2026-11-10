"""Inference Serving and KV Cache Analysis REST API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.app.schemas.api_schemas import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    InferenceGenerateRequest,
    InferenceGenerateResponse,
    KVCacheStatsRequest,
    KVCacheStatsResponse,
)
from ml.inference.engine import InferenceEngine, KVCacheCalculator

logger = logging.getLogger("benchlab.api.inference")
router = APIRouter()


@router.post("/generate", response_model=InferenceGenerateResponse)
def generate_text(req: InferenceGenerateRequest) -> Any:
    """Execute single-prompt generation with Time-To-First-Token and throughput metrics."""
    try:
        engine = InferenceEngine(model_id=req.model_id)
        res = engine.generate(
            prompt=req.prompt,
            max_output_tokens=req.max_output_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return InferenceGenerateResponse(**res)
    except Exception as e:
        logger.error(f"Inference generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchInferenceResponse)
def generate_batch_text(req: BatchInferenceRequest) -> Any:
    """Execute dynamic batch inference across multiple prompts with dynamic padding."""
    try:
        engine = InferenceEngine(model_id=req.model_id)
        res = engine.generate_batch(
            prompts=req.prompts,
            max_output_tokens=req.max_output_tokens,
            temperature=req.temperature,
        )
        return BatchInferenceResponse(**res)
    except Exception as e:
        logger.error(f"Batch generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream")
def stream_generation(
    prompt: str = Query(..., description="Input prompt for generation"),
    model_id: str = Query("Qwen/Qwen2.5-0.5B-Instruct", description="Target model"),
    max_tokens: int = Query(128, description="Maximum tokens to generate"),
    temperature: float = Query(0.1, description="Sampling temperature"),
) -> StreamingResponse:
    """Stream generated tokens in real time via Server-Sent Events (SSE)."""
    try:
        engine = InferenceEngine(model_id=model_id)

        def event_stream():
            for token in engine.generate_stream(prompt=prompt, max_output_tokens=max_tokens, temperature=temperature):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kv-cache-stats", response_model=KVCacheStatsResponse)
def get_kv_cache_stats(
    batch_size: int = Query(1, ge=1, le=256, description="Concurrency / batch size"),
    context_length: int = Query(2048, ge=1, le=131072, description="Sequence context length"),
    num_layers: int = Query(24, ge=1, description="Transformer layer count"),
    num_kv_heads: int = Query(2, ge=1, description="Key-Value attention heads (GQA)"),
    head_dim: int = Query(64, ge=1, description="Head dimension"),
    precision: str = Query("fp16", description="Cache precision (fp16, int8, int4)"),
) -> Any:
    """Calculate exact KV cache memory consumption and scaling curve across context lengths."""
    stats = KVCacheCalculator.calculate_memory(
        batch_size=batch_size,
        context_length=context_length,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        precision=precision,
    )
    return KVCacheStatsResponse(**stats)

