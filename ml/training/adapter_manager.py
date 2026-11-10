"""Adapter Manager for discovering, loading, and evaluating PEFT fine-tuned models."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.app.database.models import EvaluationRecord, ExperimentRecord, ModelRecord
from backend.app.database.session import SessionLocal
from ml.core.hardware import MemoryTracker
from ml.datasets.billsum import BillSumDatasetManager
from ml.evaluation.metrics import QualityMetricsCalculator
from ml.models.base_loader import ModelManager
from ml.preprocessing.summarization import BillSummarizationPreprocessor

logger = logging.getLogger("benchlab.adapters")


class AdapterManager:
    """Manages discovering, loading, evaluating, and serving fine-tuned LoRA/QLoRA adapters."""

    CHECKPOINT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints"

    def __init__(self, checkpoint_root: Optional[Path] = None):
        self.checkpoint_root = checkpoint_root or self.CHECKPOINT_ROOT
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.metrics_calc = QualityMetricsCalculator()
        self.dataset_manager = BillSumDatasetManager()

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Scan artifacts/checkpoints for trained adapter directories and inspect their configs."""
        checkpoints = []
        if not self.checkpoint_root.exists():
            return []

        for p in self.checkpoint_root.iterdir():
            if p.is_dir():
                adapter_cfg_file = p / "adapter_config.json"
                if adapter_cfg_file.exists():
                    try:
                        with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        size_bytes = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())
                        checkpoints.append({
                            "checkpoint_id": p.name,
                            "path": str(p),
                            "base_model_name_or_path": cfg.get("base_model_name_or_path", ""),
                            "r": cfg.get("r", 16),
                            "lora_alpha": cfg.get("lora_alpha", 32),
                            "target_modules": cfg.get("target_modules", []),
                            "size_mb": round(size_bytes / (1024**2), 2),
                            "created_at": p.stat().st_mtime,
                        })
                    except Exception as e:
                        logger.warning(f"Could not parse adapter config in {p}: {e}")
        return checkpoints

    def load_model_with_adapter(
        self,
        base_model_id: str,
        adapter_path: str,
        device: str = "auto",
    ) -> Tuple[Any, Any]:
        """Load base model and attach fine-tuned LoRA adapter."""
        target_device = ModelManager.get_optimal_device(device)
        # On macOS Apple Silicon, PeftModel stream hooks encounter AGX Metal driver event collisions.
        # Fallback to CPU for PeftModel inference ensures rock-solid stability and fast execution.
        if target_device == "mps":
            target_device = "cpu"

        logger.info(f"Loading base model '{base_model_id}' with adapter from '{adapter_path}' on {target_device}...")

        base_model, tokenizer, _ = ModelManager.load_model_and_tokenizer(
            model_id=base_model_id,
            precision="fp32" if target_device == "cpu" else "fp16",
            device=target_device,
        )

        peft_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )
        peft_model.eval()
        return peft_model, tokenizer

    def evaluate_adapter(
        self,
        base_model_id: str,
        adapter_path: str,
        split: str = "ca_test",
        num_samples: int = 10,
        max_input_length: int = 512,
        max_output_tokens: int = 128,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Benchmark fine-tuned model against the test dataset and compare with base baseline."""
        exp_id = experiment_id or f"exp_adapter_{int(time.time())}"
        model_name = f"{base_model_id}+lora"

        # 1. Load test records
        raw_splits = self.dataset_manager.load_or_download_raw(max_test_samples=max(30, num_samples))
        cleaned_splits, _ = self.dataset_manager.prepare_and_clean_dataset(raw_splits)
        eval_records = cleaned_splits.get(split, [])[:num_samples]
        if not eval_records:
            eval_records = cleaned_splits.get("test", [])[:num_samples]

        # 2. Load fine-tuned model
        peft_model, tokenizer = self.load_model_with_adapter(
            base_model_id=base_model_id,
            adapter_path=adapter_path,
        )
        target_device = "cpu" if ModelManager.get_optimal_device("auto") == "mps" else ModelManager.get_optimal_device("auto")
        preprocessor = BillSummarizationPreprocessor(
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_tokens,
        )

        # 3. Generate summaries
        generations: List[Dict[str, Any]] = []
        sample_metrics: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []
        ttfts_ms: List[float] = []
        tokens_per_sec_list: List[float] = []

        tracker = MemoryTracker(device=target_device)
        tracker.start()

        for rec in eval_records:
            bill_text = rec["text"]
            reference = rec["summary"]

            prompt = preprocessor.format_prompt(bill_text)
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            if target_device in ("mps", "cuda", "cpu"):
                input_ids = input_ids.to(target_device)

            input_tok_count = input_ids.shape[1]

            start_t = time.perf_counter()
            with torch.no_grad():
                outputs = peft_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_output_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            end_t = time.perf_counter()

            gen_latency = round((end_t - start_t) * 1000, 2)
            latencies_ms.append(gen_latency)

            new_tokens = outputs[0][input_tok_count:]
            out_tok_count = len(new_tokens)
            ttft = round(gen_latency / max(1, out_tok_count), 2)
            ttfts_ms.append(ttft)

            tps = round(out_tok_count / (gen_latency / 1000.0), 2) if gen_latency > 0 else 0.0
            tokens_per_sec_list.append(tps)

            cand_summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            m = self.metrics_calc.evaluate_summary(
                reference=reference,
                candidate=cand_summary,
                source_text=bill_text,
            )
            sample_metrics.append(m)

            generations.append({
                "bill_id": rec["id"],
                "generated_summary": cand_summary,
                "reference_summary": reference,
                "latency_ms": gen_latency,
                "tokens_per_second": tps,
                "rouge1": m["rouge1"],
                "rouge2": m["rouge2"],
                "rougeL": m["rougeL"],
                "factual_consistency": m["factual_consistency"],
            })
            tracker.update_peak()

        tracker.stop()
        mem_summary = tracker.get_summary()

        avg_quality = self.metrics_calc.aggregate_metrics(sample_metrics)
        avg_lat = round(float(sum(latencies_ms) / len(latencies_ms)), 2) if latencies_ms else 0.0
        avg_ttft = round(float(sum(ttfts_ms) / len(ttfts_ms)), 2) if ttfts_ms else 0.0
        avg_tps = round(float(sum(tokens_per_sec_list) / len(tokens_per_sec_list)), 2) if tokens_per_sec_list else 0.0

        # Retrieve baseline for comparison
        baseline_comparison = self._get_baseline_delta(base_model_id, avg_quality, avg_lat)

        res = {
            "evaluation_id": f"eval_adapter_{int(time.time())}",
            "experiment_id": exp_id,
            "model_id": model_name,
            "base_model_id": base_model_id,
            "adapter_path": adapter_path,
            "dataset_name": "billsum",
            "split": split,
            "num_samples": len(eval_records),
            "quality": avg_quality,
            "operational": {
                "avg_latency_ms": avg_lat,
                "avg_ttft_ms": avg_ttft,
                "tokens_per_second": avg_tps,
                "peak_memory_mb": mem_summary.get("peak_ram_mb", 0.0),
            },
            "baseline_delta": baseline_comparison,
            "generations_sample": generations[:3],
        }

        # Save to database
        self._register_fine_tuned_model(base_model_id, adapter_path, model_name)
        self._save_eval_to_database(res)

        return res

    def _get_baseline_delta(
        self,
        base_model_id: str,
        current_quality: Dict[str, float],
        current_latency: float,
    ) -> Dict[str, Any]:
        """Compute exact metric deltas between fine-tuned adapter and baseline model."""
        session = SessionLocal()
        try:
            baseline = (
                session.query(EvaluationRecord)
                .filter_by(model_id=base_model_id)
                .order_by(EvaluationRecord.created_at.desc())
                .first()
            )
            if not baseline:
                return {"notice": "No prior baseline evaluation found in database."}

            delta_rougel = round(current_quality["rougeL"] - baseline.rougeL, 2)
            delta_rouge1 = round(current_quality["rouge1"] - baseline.rouge1, 2)
            delta_rouge2 = round(current_quality["rouge2"] - baseline.rouge2, 2)
            delta_latency = round(current_latency - baseline.avg_latency_ms, 2)

            return {
                "baseline_model_id": base_model_id,
                "baseline_rougeL": baseline.rougeL,
                "fine_tuned_rougeL": current_quality["rougeL"],
                "delta_rougeL": delta_rougel,
                "delta_rouge1": delta_rouge1,
                "delta_rouge2": delta_rouge2,
                "baseline_latency_ms": baseline.avg_latency_ms,
                "fine_tuned_latency_ms": current_latency,
                "delta_latency_ms": delta_latency,
            }
        except Exception as e:
            logger.warning(f"Failed to query baseline: {e}")
            return {}
        finally:
            session.close()

    def _register_fine_tuned_model(self, base_model_id: str, adapter_path: str, model_name: str) -> None:
        """Register newly fine-tuned model in the models table."""
        session = SessionLocal()
        try:
            rec = session.query(ModelRecord).filter_by(id=model_name).first()
            if not rec:
                rec = ModelRecord(
                    id=model_name,
                    name=f"Qwen 2.5 0.5B (LoRA Fine-Tuned)",
                    provider="fine-tuned",
                    architecture="Qwen2ForCausalLM + LoRA",
                    parameter_count_b=0.494,
                    context_window=32768,
                    precision="fp16",
                    is_commercial_api=False,
                    is_fine_tuned=True,
                    is_quantized=False,
                    base_model_id=base_model_id,
                    adapter_path=adapter_path,
                    hardware_compatibility={"supported_devices": ["mps", "cuda", "cpu"]},
                    license="Apache-2.0",
                )
                session.add(rec)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error registering fine-tuned model: {e}")
        finally:
            session.close()

    def _save_eval_to_database(self, res: Dict[str, Any]) -> None:
        """Save fine-tuned adapter evaluation to SQLite database."""
        session = SessionLocal()
        try:
            eval_rec = EvaluationRecord(
                id=res["evaluation_id"],
                experiment_id=res["experiment_id"],
                model_id=res["model_id"],
                dataset_name=res["dataset_name"],
                split=res["split"],
                num_samples=res["num_samples"],
                rouge1=res["quality"]["rouge1"],
                rouge2=res["quality"]["rouge2"],
                rougeL=res["quality"]["rougeL"],
                bert_score=res["quality"]["bert_score"],
                factual_consistency=res["quality"]["factual_consistency"],
                avg_latency_ms=res["operational"]["avg_latency_ms"],
                avg_ttft_ms=res["operational"]["avg_ttft_ms"],
                tokens_per_second=res["operational"]["tokens_per_second"],
                peak_memory_mb=res["operational"]["peak_memory_mb"],
                cost_per_request_usd=round(res["operational"]["avg_latency_ms"] * 0.25 / 3.6e6 * 0.15, 6),
                generations_sample=res["generations_sample"],
            )
            session.add(eval_rec)

            exp_rec = ExperimentRecord(
                id=res["experiment_id"],
                name=f"Fine-Tuned Adapter: {res['model_id']}",
                model_id=res["model_id"],
                category="lora",
                rouge1=res["quality"]["rouge1"],
                rouge2=res["quality"]["rouge2"],
                rougeL=res["quality"]["rougeL"],
                bert_score=res["quality"]["bert_score"],
                avg_latency_ms=res["operational"]["avg_latency_ms"],
                tokens_per_second=res["operational"]["tokens_per_second"],
                peak_memory_mb=res["operational"]["peak_memory_mb"],
                cost_per_1k_requests=round(eval_rec.cost_per_request_usd * 1000, 4),
                status="completed",
                parameters={"base_model": res["base_model_id"], "adapter": res["adapter_path"]},
                metrics={**res["quality"], **res["operational"], **res.get("baseline_delta", {})},
            )
            session.merge(exp_rec)
            session.commit()
            logger.info("Saved fine-tuned evaluation to database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving adapter eval to DB: {e}")
        finally:
            session.close()
