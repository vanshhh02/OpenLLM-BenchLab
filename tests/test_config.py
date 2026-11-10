"""Unit tests for configuration loaders and validation models."""

from pathlib import Path
import pytest
from ml.core.config import ModelConfig, TrainingConfig, load_yaml_config


def test_load_model_configs():
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "models"
    config_files = list(config_dir.glob("*.yaml"))
    assert len(config_files) >= 3, f"Expected at least 3 model configs, found {len(config_files)}"

    for cfg_file in config_files:
        raw_data = load_yaml_config(cfg_file)
        model_cfg = ModelConfig(**raw_data)
        assert model_cfg.model_id
        assert model_cfg.model_name
        assert model_cfg.context_window > 0


def test_load_training_configs():
    training_dir = Path(__file__).resolve().parents[1] / "configs" / "training"
    lora_file = training_dir / "lora_default.yaml"
    assert lora_file.exists()

    raw_data = load_yaml_config(lora_file)
    train_cfg = TrainingConfig(**raw_data)
    assert train_cfg.job_id
    assert train_cfg.lora.r > 0
    assert train_cfg.learning_rate > 0

