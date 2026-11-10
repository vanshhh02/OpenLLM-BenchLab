"""Unit tests for LoRA/QLoRA configuration, dataset creation, and adapter management."""

import os
import pytest
import torch
from transformers import AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"

from ml.core.config import LoRAConfig as LoRASchema, TrainingConfig
from ml.training.adapter_manager import AdapterManager
from ml.training.trainer import BillSumTorchDataset, FineTuningEngine


def test_billsum_torch_dataset():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = [
        {
            "text": "SECTION 1. This is a congressional bill text for testing fine-tuning.",
            "summary": "Summary of test bill.",
        }
    ]
    ds = BillSumTorchDataset(records=records, tokenizer=tokenizer, max_length=64)
    assert len(ds) == 1
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape[0] == 64
    assert sample["labels"].shape[0] == 64
    # Verify padding tokens are masked to -100 in labels
    assert -100 in sample["labels"]


def test_peft_model_creation():
    engine = FineTuningEngine()
    lora_cfg = LoRASchema(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    peft_model, tokenizer, meta = engine.create_peft_model(
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        lora_cfg=lora_cfg,
    )
    assert meta["trainable_params"] > 0
    assert meta["trainable_params"] < meta["total_params"]
    assert meta["trainable_percentage"] < 2.0  # LoRA should be ~0.2% - 1.5%
    assert tokenizer is not None


def test_adapter_manager_list():
    mgr = AdapterManager()
    checkpoints = mgr.list_checkpoints()
    assert isinstance(checkpoints, list)
