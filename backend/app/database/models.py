"""SQLAlchemy ORM models for OpenLLM BenchLab metadata and experiment storage.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ModelRecord(Base):
    """Registered open-weight or commercial models."""
    __tablename__ = "models"

    id = Column(String(128), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    provider = Column(String(64), default="huggingface")
    architecture = Column(String(128), default="Decoder-Only Transformer")
    parameter_count_b = Column(Float, nullable=False, default=0.0)
    context_window = Column(Integer, default=4096)
    precision = Column(String(32), default="fp16")
    is_commercial_api = Column(Boolean, default=False)
    is_fine_tuned = Column(Boolean, default=False)
    is_quantized = Column(Boolean, default=False)
    base_model_id = Column(String(128), nullable=True)
    adapter_path = Column(String(512), nullable=True)
    quantization_method = Column(String(64), nullable=True)
    hardware_compatibility = Column(JSON, default=dict)
    license = Column(String(128), default="Apache-2.0")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class DatasetRecord(Base):
    """Metadata for processed datasets and splits."""
    __tablename__ = "datasets"

    id = Column(String(128), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    train_count = Column(Integer, default=0)
    val_count = Column(Integer, default=0)
    test_count = Column(Integer, default=0)
    avg_input_tokens = Column(Float, default=0.0)
    avg_summary_tokens = Column(Float, default=0.0)
    max_input_tokens = Column(Integer, default=0)
    max_summary_tokens = Column(Integer, default=0)
    metadata_json = Column(JSON, default=dict)
    sample_records = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class EvaluationRecord(Base):
    """Baseline and comparative evaluation results."""
    __tablename__ = "evaluations"

    id = Column(String(128), primary_key=True, index=True)
    experiment_id = Column(String(128), index=True, nullable=True)
    model_id = Column(String(128), index=True, nullable=False)
    dataset_name = Column(String(128), default="billsum")
    split = Column(String(64), default="ca_test")
    num_samples = Column(Integer, default=0)

    # Quality metrics
    rouge1 = Column(Float, default=0.0)
    rouge2 = Column(Float, default=0.0)
    rougeL = Column(Float, default=0.0)
    bert_score = Column(Float, nullable=True)
    factual_consistency = Column(Float, nullable=True)
    avg_input_length = Column(Float, default=0.0)
    avg_output_length = Column(Float, default=0.0)

    # Operational metrics
    avg_latency_ms = Column(Float, default=0.0)
    avg_ttft_ms = Column(Float, default=0.0)
    tokens_per_second = Column(Float, default=0.0)
    peak_memory_mb = Column(Float, default=0.0)
    cost_per_request_usd = Column(Float, default=0.0)

    hardware_info = Column(JSON, default=dict)
    generations_sample = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class TrainingJobRecord(Base):
    """Fine-tuning job runs (LoRA / QLoRA)."""
    __tablename__ = "training_jobs"

    id = Column(String(128), primary_key=True, index=True)
    base_model_id = Column(String(128), nullable=False)
    method = Column(String(32), default="lora")  # lora or qlora
    status = Column(String(32), default="pending")  # pending, running, completed, failed
    config_json = Column(JSON, default=dict)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    trainable_parameters = Column(Integer, nullable=True)
    total_parameters = Column(Integer, nullable=True)
    trainable_percentage = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    checkpoint_path = Column(String(512), nullable=True)
    checkpoint_size_mb = Column(Float, nullable=True)
    peak_memory_mb = Column(Float, nullable=True)
    metrics_history = Column(JSON, default=list)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class QuantizationRecord(Base):
    """Model quantization benchmarks and comparisons."""
    __tablename__ = "quantizations"

    id = Column(String(128), primary_key=True, index=True)
    base_model_id = Column(String(128), nullable=False)
    quantized_model_id = Column(String(128), nullable=False)
    method = Column(String(64), nullable=False)
    target_precision = Column(String(32), nullable=False)
    original_memory_mb = Column(Float, default=0.0)
    quantized_memory_mb = Column(Float, default=0.0)
    memory_reduction_pct = Column(Float, default=0.0)
    baseline_rougel = Column(Float, nullable=True)
    quantized_rougel = Column(Float, nullable=True)
    rougel_delta = Column(Float, nullable=True)
    baseline_latency_ms = Column(Float, nullable=True)
    quantized_latency_ms = Column(Float, nullable=True)
    speedup_ratio = Column(Float, nullable=True)
    status = Column(String(32), default="completed")
    artifact_path = Column(String(512), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class BenchmarkRecord(Base):
    """Inference and concurrent load testing benchmark results."""
    __tablename__ = "benchmarks"

    id = Column(String(128), primary_key=True, index=True)
    model_id = Column(String(128), nullable=False)
    benchmark_type = Column(String(32), default="load_test")  # single, concurrent, continuous_batching
    concurrency_level = Column(Integer, default=1)
    num_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    p50_latency_ms = Column(Float, default=0.0)
    p95_latency_ms = Column(Float, default=0.0)
    p99_latency_ms = Column(Float, default=0.0)
    avg_latency_ms = Column(Float, default=0.0)
    avg_ttft_ms = Column(Float, default=0.0)
    requests_per_second = Column(Float, default=0.0)
    tokens_per_second = Column(Float, default=0.0)
    peak_memory_mb = Column(Float, default=0.0)
    engine = Column(String(64), default="pytorch")
    kv_cache_usage_mb = Column(Float, nullable=True)
    metrics_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ExperimentRecord(Base):
    """Top-level unified experiment registry for cross-comparison."""
    __tablename__ = "experiments"

    id = Column(String(128), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    model_id = Column(String(128), nullable=False)
    category = Column(String(64), default="baseline")  # baseline, lora, qlora, quantized, commercial_api
    rouge1 = Column(Float, nullable=True)
    rouge2 = Column(Float, nullable=True)
    rougeL = Column(Float, nullable=True)
    bert_score = Column(Float, nullable=True)
    avg_latency_ms = Column(Float, nullable=True)
    tokens_per_second = Column(Float, nullable=True)
    peak_memory_mb = Column(Float, nullable=True)
    cost_per_1k_requests = Column(Float, nullable=True)
    status = Column(String(32), default="completed")
    parameters = Column(JSON, default=dict)
    metrics = Column(JSON, default=dict)
    mlflow_run_id = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

