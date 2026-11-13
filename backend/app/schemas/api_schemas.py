"""Pydantic schemas for OpenLLM BenchLab REST API."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# -------------------------------------------------------------------------
# Hardware & System Info Schemas
# -------------------------------------------------------------------------
class HardwareInfoResponse(BaseModel):
    platform_system: str
    platform_release: str
    platform_machine: str
    cpu_brand: str
    cpu_count_physical: int
    cpu_count_logical: int
    total_ram_gb: float
    available_ram_gb: float
    accelerator_type: str
    accelerator_name: str
    accelerator_count: int
    accelerator_memory_gb: Optional[float] = None
    bf16_supported: bool
    fp16_supported: bool
    int8_supported: bool
    int4_bitsandbytes_supported: bool
    vllm_supported: bool
    llamacpp_supported: bool
    notes: List[str]


# -------------------------------------------------------------------------
# Model Schemas
# -------------------------------------------------------------------------
class ModelItemResponse(BaseModel):
    id: str
    name: str
    provider: str
    architecture: str
    parameter_count_b: float
    context_window: int
    precision: str
    is_commercial_api: bool
    is_fine_tuned: bool
    is_quantized: bool
    base_model_id: Optional[str] = None
    adapter_path: Optional[str] = None
    quantization_method: Optional[str] = None
    hardware_compatibility: Dict[str, Any] = Field(default_factory=dict)
    license: str
    created_at: Optional[datetime.datetime] = None


class ModelLoadRequest(BaseModel):
    model_id: str
    precision: Optional[str] = "fp16"
    device: Optional[str] = "auto"


# -------------------------------------------------------------------------
# Dataset Schemas
# -------------------------------------------------------------------------
class DatasetLoadRequest(BaseModel):
    dataset_name: str = "billsum"
    max_train_samples: Optional[int] = 500
    max_test_samples: Optional[int] = 100
    force_reload: bool = False


class DatasetStatisticsResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    train_count: int
    val_count: int
    test_count: int
    avg_input_tokens: float
    avg_summary_tokens: float
    max_input_tokens: int
    max_summary_tokens: int
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    sample_records: List[Dict[str, Any]] = Field(default_factory=list)


# -------------------------------------------------------------------------
# Evaluation Schemas
# -------------------------------------------------------------------------
class EvaluationRunRequest(BaseModel):
    model_id: str
    dataset_name: str = "billsum"
    split: str = "ca_test"
    num_samples: int = Field(default=20, ge=1, le=500)
    max_input_length: int = 1024
    max_output_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    batch_size: int = 1
    compare_with_api: Optional[str] = None  # e.g. "openai/gpt-4o-mini"


class EvaluationResultItem(BaseModel):
    id: str
    experiment_id: Optional[str] = None
    model_id: str
    dataset_name: str
    split: str
    num_samples: int
    rouge1: float
    rouge2: float
    rougeL: float
    bert_score: Optional[float] = None
    factual_consistency: Optional[float] = None
    avg_input_length: float
    avg_output_length: float
    avg_latency_ms: float
    avg_ttft_ms: float
    tokens_per_second: float
    peak_memory_mb: float
    cost_per_request_usd: float
    hardware_info: Dict[str, Any] = Field(default_factory=dict)
    generations_sample: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[datetime.datetime] = None


class EvaluationCompareResponse(BaseModel):
    evaluations: List[EvaluationResultItem]
    metrics_comparison: Dict[str, Dict[str, float]]
    relative_speedup: Dict[str, float]
    memory_savings_pct: Dict[str, float]


# -------------------------------------------------------------------------
# Training / Fine-Tuning Schemas
# -------------------------------------------------------------------------
class TrainingStartRequest(BaseModel):
    base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    method: Literal["lora", "qlora"] = "lora"
    rank: int = Field(default=16, ge=1, le=128)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    learning_rate: float = Field(default=2e-4, gt=0)
    batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    epochs: int = Field(default=3, ge=1)
    max_steps: int = Field(default=50)  # default to fast step limit for local interactive run
    max_seq_length: int = Field(default=512, ge=128, le=4096)
    dataset_name: str = "billsum"
    dataset_samples: int = Field(default=200, ge=10)


class TrainingStatusResponse(BaseModel):
    id: str
    base_model_id: str
    method: str
    status: str
    config_json: Dict[str, Any]
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    trainable_parameters: Optional[int] = None
    total_parameters: Optional[int] = None
    trainable_percentage: Optional[float] = None
    duration_seconds: Optional[float] = None
    checkpoint_path: Optional[str] = None
    checkpoint_size_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    metrics_history: List[Dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None
    created_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None


# -------------------------------------------------------------------------
# Quantization Schemas
# -------------------------------------------------------------------------
class QuantizationRunRequest(BaseModel):
    base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    quantization_method: Literal["dynamic_int8", "bitsandbytes_nf4", "bitsandbytes_int8", "gguf_q4_k_m", "fp16"] = "dynamic_int8"
    target_precision: str = "int8"
    evaluate_after_quant: bool = True
    eval_samples: int = 15


class QuantizationResultItem(BaseModel):
    id: str
    base_model_id: str
    quantized_model_id: str
    method: str
    target_precision: str
    original_memory_mb: float
    quantized_memory_mb: float
    memory_reduction_pct: float
    baseline_rougel: Optional[float] = None
    quantized_rougel: Optional[float] = None
    rougel_delta: Optional[float] = None
    baseline_latency_ms: Optional[float] = None
    quantized_latency_ms: Optional[float] = None
    speedup_ratio: Optional[float] = None
    status: str
    artifact_path: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime.datetime] = None


# -------------------------------------------------------------------------
# Inference & Load Testing Schemas
# -------------------------------------------------------------------------
class InferenceGenerateRequest(BaseModel):
    model_id: str
    prompt: str
    max_output_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    stream: bool = False


class InferenceGenerateResponse(BaseModel):
    model_id: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_per_second: float
    peak_memory_mb: float


class BatchInferenceRequest(BaseModel):
    model_id: str
    prompts: List[str]
    max_output_tokens: int = 256
    temperature: float = 0.1


class BatchInferenceResponse(BaseModel):
    model_id: str
    results: List[InferenceGenerateResponse]
    total_batch_latency_ms: float
    throughput_tps: float


class KVCacheStatsRequest(BaseModel):
    batch_size: int = 1
    context_length: int = 2048
    num_layers: int = 24
    num_kv_heads: int = 2
    head_dim: int = 64
    precision: str = "fp16"


class KVCacheStatsResponse(BaseModel):
    batch_size: int
    context_length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    precision: str
    bytes_per_element: float
    total_elements: int
    total_bytes: int
    total_memory_mb: float
    memory_per_token_bytes: float
    scaling_curve: List[Dict[str, Any]]


class LoadTestRequest(BaseModel):
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    concurrency_levels: List[int] = Field(default_factory=lambda: [1, 5, 10])
    requests_per_level: int = Field(default=10, ge=1)
    prompt: Optional[str] = None
    max_output_tokens: int = 128


class BenchmarkResultItem(BaseModel):
    id: str
    model_id: str
    benchmark_type: str
    concurrency_level: int
    num_requests: int
    successful_requests: int
    failed_requests: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float
    avg_ttft_ms: float
    requests_per_second: float
    tokens_per_second: float
    peak_memory_mb: float
    engine: str
    kv_cache_usage_mb: Optional[float] = None
    metrics_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime.datetime] = None


# -------------------------------------------------------------------------
# Cost Analysis Schemas
# -------------------------------------------------------------------------
class CostCalculationRequest(BaseModel):
    monthly_requests: int = Field(default=100000, ge=1)
    avg_input_tokens: int = Field(default=1500, ge=1)
    avg_output_tokens: int = Field(default=250, ge=1)

    # API Pricing
    api_name: str = "Frontier Commercial API (e.g. GPT-4o-mini)"
    api_input_cost_per_m: float = Field(default=0.150, ge=0.0)
    api_output_cost_per_m: float = Field(default=0.600, ge=0.0)

    # Self-Hosted Local / Cloud Infrastructure
    server_monthly_rental_usd: float = Field(default=85.0, ge=0.0)
    hardware_purchase_cost_usd: float = Field(default=0.0, ge=0.0)
    amortization_months: int = Field(default=36, ge=1)
    server_power_watts: float = Field(default=250.0, ge=0.0)
    electricity_kwh_cost_usd: float = Field(default=0.15, ge=0.0)
    operational_overhead_monthly_usd: float = Field(default=20.0, ge=0.0)
    tokens_per_second_local: float = Field(default=65.0, gt=0.0)


class CostCalculationResponse(BaseModel):
    monthly_requests: int
    avg_input_tokens: int
    avg_output_tokens: int

    # API costs
    api_cost_per_request_usd: float
    api_cost_per_1k_usd: float
    monthly_api_cost_usd: float

    # Local / Dedicated costs
    local_fixed_cost_monthly_usd: float
    local_electricity_cost_monthly_usd: float
    local_variable_cost_monthly_usd: float
    monthly_local_cost_usd: float
    local_cost_per_request_usd: float
    local_cost_per_1k_usd: float

    # Economics & Break-Even
    break_even_requests: Optional[float] = None
    break_even_status: str
    monthly_savings_usd: float
    savings_percentage: float

    # Sensitivity Analysis across volumes
    sensitivity_curve: List[Dict[str, Any]]


# -------------------------------------------------------------------------
# Unified Experiment Schemas
# -------------------------------------------------------------------------
class ExperimentItemResponse(BaseModel):
    id: str
    name: str
    model_id: str
    category: str
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    bert_score: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    cost_per_1k_requests: Optional[float] = None
    status: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    mlflow_run_id: Optional[str] = None
    created_at: Optional[datetime.datetime] = None


class ExperimentCompareRequest(BaseModel):
    experiment_ids: List[str]


# -------------------------------------------------------------------------
# Phase 8: Auth, Multi-User Workspaces, API Keys & Async Job Queue Schemas
# -------------------------------------------------------------------------
class UserRegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: str = Field(default="", max_length=255)
    workspace_name: Optional[str] = Field(default=None, max_length=255)


class UserLoginRequest(BaseModel):
    email: str
    password: str
    workspace_id: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserSummary(BaseModel):
    id: str
    email: str
    full_name: str
    is_active: bool
    is_superuser: bool = False
    created_at: Optional[datetime.datetime] = None


class WorkspaceSummary(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str] = None
    owner_id: Optional[str] = None
    role: str = "member"
    member_count: int = 1
    created_at: Optional[datetime.datetime] = None


class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserSummary
    workspace: WorkspaceSummary


class AuthMeResponse(BaseModel):
    authenticated: bool
    auth_method: str
    user: Optional[UserSummary] = None
    workspace_id: str
    workspace_role: str
    scopes: List[str] = Field(default_factory=list)


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    scopes: List[str] = Field(default_factory=lambda: ["read", "write"])
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    workspace_id: str
    scopes: List[str] = Field(default_factory=list)
    is_active: bool
    last_used_at: Optional[datetime.datetime] = None
    expires_at: Optional[datetime.datetime] = None
    created_at: Optional[datetime.datetime] = None
    raw_key: Optional[str] = None  # Returned only once upon creation


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    description: Optional[str] = None


class WorkspaceUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=255)
    description: Optional[str] = None
    settings_json: Optional[Dict[str, Any]] = None


class WorkspaceMemberAddRequest(BaseModel):
    email: str
    role: str = Field(default="member", pattern="^(owner|admin|member|viewer)$")


class WorkspaceMemberUpdateRequest(BaseModel):
    role: str = Field(..., pattern="^(owner|admin|member|viewer)$")


class WorkspaceMemberResponse(BaseModel):
    id: str
    workspace_id: str
    user_id: str
    email: str
    full_name: str
    role: str
    created_at: Optional[datetime.datetime] = None


class AsyncJobSubmitRequest(BaseModel):
    job_type: str = Field(..., pattern="^(training|evaluation|quantization|load_test)$")
    payload: Dict[str, Any] = Field(default_factory=dict)


class AsyncJobResponse(BaseModel):
    id: str
    workspace_id: str
    created_by_user_id: Optional[str] = None
    job_type: str
    status: str
    backend: str
    celery_task_id: Optional[str] = None
    payload_json: Dict[str, Any] = Field(default_factory=dict)
    result_json: Optional[Dict[str, Any]] = None
    progress_pct: float = 0.0
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime.datetime] = None
    started_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None


