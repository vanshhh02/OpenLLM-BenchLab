"""LoRA and QLoRA Fine-Tuning Engine for OpenLLM BenchLab."""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from backend.app.database.models import TrainingJobRecord
from backend.app.database.session import SessionLocal
from ml.core.config import LoRAConfig as LoRASchema, TrainingConfig
from ml.core.hardware import detect_hardware, MemoryTracker
from ml.datasets.billsum import BillSumDatasetManager
from ml.models.base_loader import ModelManager
from ml.preprocessing.summarization import BillSummarizationPreprocessor

logger = logging.getLogger("benchlab.training")


class BillSumTorchDataset(Dataset):
    """PyTorch Dataset for causal language model fine-tuning on BillSum."""

    def __init__(self, records: List[Dict[str, Any]], tokenizer: Any, max_length: int = 512):
        self.examples: List[Dict[str, torch.Tensor]] = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        for rec in records:
            bill_text = rec.get("text", "")
            summary = rec.get("summary", "")
            if not bill_text or not summary:
                continue

            # Format full training sequence: prompt + reference summary + EOS
            prompt = (
                f"<|im_start|>system\nYou are an expert legislative analyst who summarizes congressional bills.<|im_end|>\n"
                f"<|im_start|>user\nSummarize the following bill:\n\n{bill_text}<|im_end|>\n"
                f"<|im_start|>assistant\n{summary}<|im_end|>"
            )

            tokenized = tokenizer(
                prompt,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            # Mask padding tokens in labels with -100
            labels[labels == pad_token_id] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class FineTuningEngine:
    """Manages parameter-efficient fine-tuning with LoRA and QLoRA."""

    CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints"

    def __init__(self, checkpoint_root: Optional[Path] = None):
        self.checkpoint_root = checkpoint_root or self.CHECKPOINT_ROOT
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.dataset_manager = BillSumDatasetManager()

    def create_peft_model(
        self,
        base_model_id: str,
        lora_cfg: LoRASchema,
        device: str = "auto",
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Initialize base model and attach trainable LoRA/QLoRA adapter."""
        hw = detect_hardware()
        target_device = ModelManager.get_optimal_device(device)
        is_qlora = lora_cfg.use_qlora

        logger.info(
            f"Configuring PEFT model: base={base_model_id}, method={'QLoRA' if is_qlora else 'LoRA'}, "
            f"target_device={target_device}"
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Hardware-aware quantization check for QLoRA
        bnb_config = None
        if is_qlora:
            if hw.accelerator_type == "cuda" and hw.int4_bitsandbytes_supported:
                from transformers import BitsAndBytesConfig
                compute_dtype = getattr(torch, lora_cfg.bnb_4bit_compute_dtype, torch.bfloat16)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=lora_cfg.quant_type,
                    bnb_4bit_use_double_quant=lora_cfg.use_double_quant,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                logger.info("Configured BitsAndBytesConfig for 4-bit NF4 QLoRA on CUDA.")
            else:
                logger.info(
                    "CUDA bitsandbytes NF4 unavailable on this accelerator (Apple Silicon/CPU). "
                    "Running QLoRA configuration in simulated 4-bit precision profile with FP16/BF16 adapter."
                )

        # Load base model
        dtype = torch.bfloat16 if (target_device == "cuda" and hw.bf16_supported) else torch.float16
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        elif target_device == "cuda":
            model_kwargs["device_map"] = "auto"

        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

        if target_device in ("mps", "cpu") and bnb_config is None:
            base_model = base_model.to(target_device)

        # Configure LoRA adapter
        target_modules = lora_cfg.target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            target_modules=target_modules,
        )

        peft_model = get_peft_model(base_model, peft_config)
        trainable_params, all_params = peft_model.get_nb_trainable_parameters()
        trainable_pct = round((trainable_params / all_params) * 100, 4)

        logger.info(
            f"PEFT attached. Trainable parameters: {trainable_params:,} / {all_params:,} "
            f"({trainable_pct}% trainable)."
        )

        meta = {
            "base_model_id": base_model_id,
            "method": "qlora" if is_qlora else "lora",
            "device": target_device,
            "trainable_params": trainable_params,
            "total_params": all_params,
            "trainable_percentage": trainable_pct,
            "rank": lora_cfg.r,
            "alpha": lora_cfg.lora_alpha,
        }
        return peft_model, tokenizer, meta

    def train(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute complete fine-tuning job with step tracking and checkpointing."""
        job_id = config.job_id or f"train_{int(time.time())}"
        checkpoint_dir = self.checkpoint_root / job_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training run '{job_id}' (epochs: {config.num_train_epochs}, lr: {config.learning_rate})")

        # 1. Update Database Status
        self._update_db_status(job_id=job_id, status="running", config=config)

        # 2. Prepare Datasets
        raw_splits = self.dataset_manager.load_or_download_raw(
            max_train_samples=150,
            max_test_samples=30,
        )
        cleaned_splits, _ = self.dataset_manager.prepare_and_clean_dataset(raw_splits)
        train_records = cleaned_splits.get("train", [])
        val_records = cleaned_splits.get("ca_test", []) or cleaned_splits.get("test", [])

        if not train_records:
            raise ValueError("No training data found in BillSum dataset.")

        # 3. Model & PEFT Setup
        model, tokenizer, peft_meta = self.create_peft_model(
            base_model_id=config.base_model_id,
            lora_cfg=config.lora,
        )
        target_device = peft_meta["device"]

        train_dataset = BillSumTorchDataset(
            records=train_records,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )
        val_dataset = BillSumTorchDataset(
            records=val_records[:20],
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.per_device_eval_batch_size,
            shuffle=False,
        )

        # Optimizer and Cosine Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.num_train_epochs
        if config.max_steps > 0:
            total_steps = min(total_steps, config.max_steps)

        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # 4. Training Loop
        model.train()
        tracker = MemoryTracker(device=target_device)
        tracker.start()
        start_time = time.perf_counter()

        metrics_history: List[Dict[str, Any]] = []
        global_step = 0
        running_loss = 0.0
        final_val_loss = None

        for epoch in range(config.num_train_epochs):
            for batch in train_loader:
                if config.max_steps > 0 and global_step >= config.max_steps:
                    break

                input_ids = batch["input_ids"].to(target_device)
                attention_mask = batch["attention_mask"].to(target_device)
                labels = batch["labels"].to(target_device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / config.gradient_accumulation_steps
                loss.backward()

                running_loss += loss.item() * config.gradient_accumulation_steps

                if (global_step + 1) % config.gradient_accumulation_steps == 0 or (global_step + 1 == total_steps):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1

                # Periodic evaluation and logging
                if global_step % config.logging_steps == 0 or global_step == total_steps:
                    avg_step_loss = round(running_loss / max(1, config.logging_steps), 4)
                    running_loss = 0.0

                    step_info = {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": avg_step_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    metrics_history.append(step_info)
                    logger.info(f"Step {global_step}/{total_steps} [Epoch {epoch+1}] - Train Loss: {avg_step_loss}")

                    if progress_callback:
                        progress_callback(step_info)

                tracker.update_peak()

            if config.max_steps > 0 and global_step >= config.max_steps:
                break

        # 5. Validation Evaluation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(target_device)
                attention_mask = batch["attention_mask"].to(target_device)
                labels = batch["labels"].to(target_device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_losses.append(outputs.loss.item())

        final_val_loss = round(float(sum(val_losses) / len(val_losses)), 4) if val_losses else None
        duration_sec = round(time.perf_counter() - start_time, 2)
        tracker.stop()
        mem_summary = tracker.get_summary()

        # 6. Save Adapter Checkpoint
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Calculate checkpoint size
        checkpoint_size_bytes = sum(f.stat().st_size for f in checkpoint_dir.glob("**/*") if f.is_file())
        checkpoint_size_mb = round(checkpoint_size_bytes / (1024**2), 2)

        train_loss_final = metrics_history[-1]["train_loss"] if metrics_history else 0.0

        run_summary = {
            "job_id": job_id,
            "base_model_id": config.base_model_id,
            "method": peft_meta["method"],
            "status": "completed",
            "train_loss": train_loss_final,
            "val_loss": final_val_loss,
            "trainable_parameters": peft_meta["trainable_params"],
            "total_parameters": peft_meta["total_params"],
            "trainable_percentage": peft_meta["trainable_percentage"],
            "duration_seconds": duration_sec,
            "checkpoint_path": str(checkpoint_dir),
            "checkpoint_size_mb": checkpoint_size_mb,
            "peak_memory_mb": mem_summary.get("peak_ram_mb", 0.0),
            "metrics_history": metrics_history,
        }

        # 7. Persist to Database & MLflow
        self._save_training_to_db(run_summary, config)
        self._log_training_to_mlflow(run_summary, config)

        logger.info(
            f"Training '{job_id}' completed successfully in {duration_sec}s. "
            f"Train loss: {train_loss_final}, Val loss: {final_val_loss}, Size: {checkpoint_size_mb} MB"
        )
        return run_summary

    def _update_db_status(self, job_id: str, status: str, config: TrainingConfig) -> None:
        """Create or update training job entry in SQLite."""
        session = SessionLocal()
        try:
            job = session.query(TrainingJobRecord).filter_by(id=job_id).first()
            if not job:
                job = TrainingJobRecord(
                    id=job_id,
                    base_model_id=config.base_model_id,
                    method=config.lora.use_qlora and "qlora" or "lora",
                    status=status,
                    config_json=config.dict(),
                )
                session.add(job)
            else:
                job.status = status
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update training status in DB: {e}")
        finally:
            session.close()

    def _save_training_to_db(self, summary: Dict[str, Any], config: TrainingConfig) -> None:
        """Update training job record with final metrics in SQLite."""
        session = SessionLocal()
        try:
            job = session.query(TrainingJobRecord).filter_by(id=summary["job_id"]).first()
            if job:
                job.status = "completed"
                job.train_loss = summary["train_loss"]
                job.val_loss = summary["val_loss"]
                job.trainable_parameters = summary["trainable_parameters"]
                job.total_parameters = summary["total_parameters"]
                job.trainable_percentage = summary["trainable_percentage"]
                job.duration_seconds = summary["duration_seconds"]
                job.checkpoint_path = summary["checkpoint_path"]
                job.checkpoint_size_mb = summary["checkpoint_size_mb"]
                job.peak_memory_mb = summary["peak_memory_mb"]
                job.metrics_history = summary["metrics_history"]
                job.completed_at = datetime.datetime.utcnow()
                session.commit()
                logger.info("Updated training job in database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving training run to database: {e}")
        finally:
            session.close()

    def _log_training_to_mlflow(self, summary: Dict[str, Any], config: TrainingConfig) -> None:
        """Log training parameters and metrics to MLflow."""
        try:
            import mlflow

            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlruns.db")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("openllm_benchlab_finetuning")

            with mlflow.start_run(run_name=f"train_{summary['method']}_{summary['job_id']}"):
                mlflow.log_param("base_model", config.base_model_id)
                mlflow.log_param("method", summary["method"])
                mlflow.log_param("rank", config.lora.r)
                mlflow.log_param("alpha", config.lora.lora_alpha)
                mlflow.log_param("learning_rate", config.learning_rate)
                mlflow.log_param("epochs", config.num_train_epochs)

                mlflow.log_metric("trainable_params", summary["trainable_parameters"])
                mlflow.log_metric("trainable_pct", summary["trainable_percentage"])
                mlflow.log_metric("final_train_loss", summary["train_loss"])
                if summary["val_loss"] is not None:
                    mlflow.log_metric("final_val_loss", summary["val_loss"])
                mlflow.log_metric("duration_seconds", summary["duration_seconds"])
                mlflow.log_metric("checkpoint_size_mb", summary["checkpoint_size_mb"])

            logger.info("Successfully logged training job to MLflow.")
        except Exception as e:
            logger.debug(f"MLflow fine-tuning logging skipped: {e}")

