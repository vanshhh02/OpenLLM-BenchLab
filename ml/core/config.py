"""Configuration management module for OpenLLM BenchLab.

Loads, validates, and serializes YAML and JSON configuration files
for models, training runs, evaluation suites, and deployment profiles.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for an open-weight or commercial frontier LLM."""
    model_id: str = Field(..., description="Unique model identifier or HF repo ID")
    model_name: str = Field(..., description="Human-readable model name")
    provider: Literal["huggingface", "openai", "anthropic", "google"] = "huggingface"
    architecture: str = "Decoder-Only Transformer"
    parameter_count_b: float = Field(..., description="Parameter count in billions")
    context_window: int = Field(default=4096, description="Maximum supported context length")
    default_precision: str = "fp16"  # fp16, bf16, fp32, int8, int4
    is_commercial_api: bool = False
    api_model_identifier: Optional[str] = None
    min_ram_gb: float = 4.0
    min_vram_gb: float = 2.0
    supported_devices: List[str] = Field(default_factory=lambda: ["cuda", "mps", "cpu"])
    license: str = "Apache-2.0"
    task_types: List[str] = Field(default_factory=lambda: ["summarization", "instruction-tuning"])
    chat_template: Optional[str] = None


class LoRAConfig(BaseModel):
    """Configuration for LoRA / QLoRA parameter-efficient fine-tuning."""
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    r: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA scaling parameter alpha")
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="Dropout probability for LoRA layers")
    bias: Literal["none", "all", "lora_only"] = "none"
    use_qlora: bool = False
    quant_type: Literal["fp4", "nf4"] = "nf4"
    use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


class TrainingConfig(BaseModel):
    """Configuration for fine-tuning jobs."""
    job_id: str
    base_model_id: str
    dataset_name: str = "billsum"
    dataset_split_train: str = "train"
    dataset_split_val: str = "ca_test"
    output_dir: str = "artifacts/checkpoints"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    learning_rate: float = Field(default=2e-4, gt=0)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    per_device_eval_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_train_epochs: int = Field(default=3, ge=1)
    max_steps: int = Field(default=-1)
    max_seq_length: int = Field(default=1024, ge=128)
    warmup_ratio: float = Field(default=0.03, ge=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    logging_steps: int = Field(default=10, ge=1)
    eval_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=50, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    gradient_checkpointing: bool = True
    seed: int = 42


class QuantizationConfig(BaseModel):
    """Configuration for model quantization."""
    model_id: str
    quantization_method: Literal["dynamic_int8", "bitsandbytes_int8", "bitsandbytes_nf4", "gguf_q4_k_m", "gguf_q8_0", "fp16"]
    target_precision: str
    device: str = "auto"
    save_dir: str = "artifacts/quantized"


class BenchmarkConfig(BaseModel):
    """Configuration for inference and load benchmarking."""
    model_id: str
    dataset_name: str = "billsum"
    num_samples: int = Field(default=50, ge=1)
    max_input_length: int = Field(default=1024, ge=128)
    max_output_tokens: int = Field(default=256, ge=16)
    temperature: float = Field(default=0.1, ge=0.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    concurrency_levels: List[int] = Field(default_factory=lambda: [1, 5, 10, 25, 50])
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    track_kv_cache: bool = True


class CostConfig(BaseModel):
    """Configuration and pricing parameters for Cost Analysis Engine."""
    monthly_requests: int = Field(default=100000, ge=1)
    avg_input_tokens: int = Field(default=1500, ge=1)
    avg_output_tokens: int = Field(default=250, ge=1)

    # API Pricing ($ per 1M tokens)
    api_input_cost_per_m: float = Field(default=0.150, description="Cost per 1M input tokens")
    api_output_cost_per_m: float = Field(default=0.600, description="Cost per 1M output tokens")

    # Local / Dedicated Infrastructure
    server_monthly_rental_usd: float = Field(default=85.0, description="Monthly cloud GPU or host cost")
    hardware_purchase_cost_usd: float = Field(default=0.0, description="Capital hardware purchase cost")
    amortization_months: int = Field(default=36, ge=1)
    electricity_kwh_cost_usd: float = Field(default=0.15, ge=0.0)
    server_power_watts: float = Field(default=250.0, ge=0.0)
    operational_overhead_monthly_usd: float = Field(default=20.0, ge=0.0)


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_yaml_config(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to a YAML file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

