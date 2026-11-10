"""Model loading and caching utility for open-weight Transformer LLMs."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix for PyTorch 2.15 MPSEvent nil assertion on macOS Metal Performance Shaders
try:
    import transformers.generation.utils as _gen_utils
    if hasattr(_gen_utils, "DeferredStopCheck"):
        _gen_utils.DeferredStopCheck.is_supported = staticmethod(lambda *args, **kwargs: False)
except Exception:
    pass

from ml.core.hardware import detect_hardware, MemoryTracker

logger = logging.getLogger("benchlab.models")


class ModelManager:
    """Manages loading, device placement, precision configuration, and caching for open-weight LLMs."""

    _cached_model: Optional[Any] = None
    _cached_tokenizer: Optional[Any] = None
    _cached_model_id: Optional[str] = None
    _cached_precision: Optional[str] = None

    @classmethod
    def get_optimal_device(cls, requested_device: str = "auto") -> str:
        """Select optimal execution device based on hardware profile."""
        if requested_device != "auto":
            return requested_device

        hw = detect_hardware()
        if hw.accelerator_type == "cuda":
            return "cuda"
        elif hw.accelerator_type == "mps":
            return "mps"
        return "cpu"

    @classmethod
    def get_torch_dtype(cls, precision: str, device: str) -> torch.dtype:
        """Map precision string to compatible torch dtype."""
        precision = precision.lower()
        if precision in ("bfloat16", "bf16"):
            # If on MPS or CUDA that supports bf16
            if device == "cuda" and not torch.cuda.is_bf16_supported():
                return torch.float16
            return torch.bfloat16
        elif precision in ("float16", "fp16"):
            return torch.float16
        elif precision in ("float32", "fp32"):
            return torch.float32
        return torch.float16

    @classmethod
    def load_model_and_tokenizer(
        cls,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        precision: str = "fp16",
        device: str = "auto",
        force_reload: bool = False,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load tokenizer and causal language model with resource tracking."""
        target_device = cls.get_optimal_device(device)
        dtype = cls.get_torch_dtype(precision, target_device)

        if (
            not force_reload
            and cls._cached_model is not None
            and cls._cached_model_id == model_id
            and cls._cached_precision == precision
        ):
            logger.info(f"Reusing cached model '{model_id}' on {target_device}")
            return cls._cached_model, cls._cached_tokenizer, {"cached": True, "device": target_device}

        logger.info(f"Loading model '{model_id}' (precision: {precision}, device: {target_device})...")

        tracker = MemoryTracker()
        tracker.start()
        start_time = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Device mapping strategy
        device_map = None
        if target_device == "cuda":
            device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if target_device in ("mps", "cpu") and device_map is None:
            model = model.to(target_device)

        model.eval()
        load_time_sec = round(time.perf_counter() - start_time, 2)
        tracker.stop()
        mem_summary = tracker.get_summary()

        cls._cached_model = model
        cls._cached_tokenizer = tokenizer
        cls._cached_model_id = model_id
        cls._cached_precision = precision

        param_count = sum(p.numel() for p in model.parameters())
        meta = {
            "model_id": model_id,
            "device": target_device,
            "precision": str(dtype),
            "parameter_count": param_count,
            "parameter_count_b": round(param_count / 1e9, 3),
            "load_time_sec": load_time_sec,
            "memory": mem_summary,
            "cached": False,
        }

        logger.info(
            f"Successfully loaded '{model_id}' in {load_time_sec}s "
            f"({meta['parameter_count_b']}B params, {target_device})."
        )
        return model, tokenizer, meta

    @classmethod
    def unload_cached_model(cls) -> None:
        """Free memory by clearing cached model references."""
        cls._cached_model = None
        cls._cached_tokenizer = None
        cls._cached_model_id = None
        cls._cached_precision = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

