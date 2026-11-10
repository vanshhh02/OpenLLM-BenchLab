"""Unit tests for model quantization engine and precision trade-off analysis."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ml.quantization.quantizer import QuantizationEngine


@pytest.fixture
def temp_artifact_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_quantization_engine_init(temp_artifact_dir):
    engine = QuantizationEngine(artifact_dir=temp_artifact_dir)
    assert engine.artifact_dir == temp_artifact_dir
    assert engine.hardware is not None


def test_gguf_export(temp_artifact_dir):
    engine = QuantizationEngine(artifact_dir=temp_artifact_dir)
    res = engine.run_quantization(
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        method="gguf_q4_k_m",
        target_precision="q4_k_m",
    )
    assert res["status"] == "completed"
    assert res["target_precision"] == "q4_k_m"
    assert res["memory_reduction_pct"] > 60.0
    assert os.path.exists(res["artifact_path"])
    assert "GGUF-v3" in open(res["artifact_path"]).read()


def test_bitsandbytes_fallback(temp_artifact_dir):
    engine = QuantizationEngine(artifact_dir=temp_artifact_dir)
    res = engine.run_quantization(
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        method="bitsandbytes_nf4",
        target_precision="int4",
    )
    assert res["status"] in ("completed", "hardware_fallback")
    assert res["target_precision"] == "int4"
    assert res["quantized_memory_mb"] < res["original_memory_mb"]
    assert res["memory_reduction_pct"] > 70.0


def test_fp16_baseline_entry(temp_artifact_dir):
    engine = QuantizationEngine(artifact_dir=temp_artifact_dir)
    res = engine.run_quantization(
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        method="fp16",
        target_precision="fp16",
    )
    assert res["status"] == "completed"
    assert res["memory_reduction_pct"] == 0.0
    assert res["speedup_ratio"] == 1.0


def test_dynamic_quantization_linear_module():
    """Verify PyTorch dynamic quantization on Linear modules."""
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"

    linear = nn.Linear(32, 64)
    q_linear = torch.ao.quantization.quantize_dynamic(linear, {nn.Linear}, dtype=torch.qint8)
    assert isinstance(q_linear, nn.Linear)

    x = torch.randn(2, 32)
    out = q_linear(x)
    assert out.shape == (2, 64)


def test_precision_comparison_matrix(temp_artifact_dir):
    engine = QuantizationEngine(artifact_dir=temp_artifact_dir)
    # Populate test entries
    engine.run_quantization(base_model_id="Qwen/Qwen2.5-0.5B-Instruct", method="fp16")
    engine.run_quantization(base_model_id="Qwen/Qwen2.5-0.5B-Instruct", method="gguf_q4_k_m")
    matrix = engine.build_precision_comparison_matrix("Qwen/Qwen2.5-0.5B-Instruct")
    assert len(matrix) >= 2
    precisions = [m["precision"] for m in matrix]
    assert "FP16" in precisions
    assert "Q4_K_M" in precisions

