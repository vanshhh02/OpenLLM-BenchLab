"""Baseline and Comparative Evaluation Engine for OpenLLM BenchLab."""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from backend.app.database.models import EvaluationRecord, ExperimentRecord
from backend.app.database.session import SessionLocal
from ml.core.hardware import detect_hardware, MemoryTracker
from ml.datasets.billsum import BillSumDatasetManager
from ml.evaluation.metrics import QualityMetricsCalculator
from ml.models.base_loader import ModelManager
from ml.models.frontier_adapter import get_frontier_provider
from ml.preprocessing.summarization import BillSummarizationPreprocessor

logger = logging.getLogger("benchlab.evaluation")


class BaselineEvaluator:
    """Orchestrates model evaluation across quality and operational dimensions."""

    BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "benchmarks"

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or self.BENCHMARK_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_calc = QualityMetricsCalculator()
        self.dataset_manager = BillSumDatasetManager()

    def run_evaluation(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        dataset_name: str = "billsum",
        split: str = "ca_test",
        num_samples: int = 20,
        max_input_length: int = 1024,
        max_output_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        compare_with_api: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute comprehensive evaluation for an open-weight model on BillSum."""
        exp_id = experiment_id or f"exp_baseline_{int(time.time())}"
        logger.info(
            f"Starting baseline evaluation: model={model_id}, samples={num_samples}, split={split}"
        )

        # 1. Load dataset split
        raw_splits = self.dataset_manager.load_or_download_raw(
            max_train_samples=100,
            max_test_samples=max(50, num_samples),
        )
        cleaned_splits, stats = self.dataset_manager.prepare_and_clean_dataset(raw_splits)
        self.dataset_manager.sync_to_database(stats, cleaned_splits.get(split, []))

        eval_records = cleaned_splits.get(split, [])[:num_samples]
        if not eval_records:
            eval_records = cleaned_splits.get("test", [])[:num_samples]
        if not eval_records:
            raise ValueError(f"No records available in split '{split}' for evaluation.")

        # 2. Load model and tokenizer
        model, tokenizer, load_meta = ModelManager.load_model_and_tokenizer(
            model_id=model_id,
            precision="fp16",
        )
        target_device = load_meta.get("device", "cpu")
        preprocessor = BillSummarizationPreprocessor(
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_output_length=max_output_tokens,
        )

        # 3. Benchmark Generations
        generations: List[Dict[str, Any]] = []
        sample_metrics: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []
        ttfts_ms: List[float] = []
        tokens_per_sec_list: List[float] = []

        total_input_tokens = 0
        total_output_tokens = 0

        tracker = MemoryTracker(device=target_device)
        tracker.start()

        for idx, rec in enumerate(eval_records):
            bill_id = rec["id"]
            title = rec.get("title", f"Bill #{idx+1}")
            bill_text = rec["text"]
            reference = rec["summary"]

            prompt = preprocessor.format_prompt(bill_text)
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            if target_device in ("mps", "cuda", "cpu"):
                input_ids = input_ids.to(target_device)

            input_tok_count = input_ids.shape[1]
            total_input_tokens += input_tok_count

            # Measure generation timing
            start_gen = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            end_gen = time.perf_counter()

            gen_latency_ms = round((end_gen - start_gen) * 1000, 2)
            latencies_ms.append(gen_latency_ms)

            # Extract generated tokens (excluding prompt)
            new_tokens = outputs[0][input_tok_count:]
            output_tok_count = len(new_tokens)
            total_output_tokens += output_tok_count

            # TTFT approximation: first token generation time (prefill phase)
            ttft_ms = round(gen_latency_ms / max(1, output_tok_count), 2)
            ttfts_ms.append(ttft_ms)

            tps = (
                round((output_tok_count / (gen_latency_ms / 1000.0)), 2)
                if gen_latency_ms > 0
                else 0.0
            )
            tokens_per_sec_list.append(tps)

            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Compute quality metrics for this pair
            m = self.metrics_calc.evaluate_summary(
                reference=reference,
                candidate=generated_text,
                source_text=bill_text,
            )
            sample_metrics.append(m)

            generations.append({
                "bill_id": bill_id,
                "title": title,
                "input_tokens": input_tok_count,
                "output_tokens": output_tok_count,
                "latency_ms": gen_latency_ms,
                "ttft_ms": ttft_ms,
                "tokens_per_second": tps,
                "reference_summary": reference,
                "generated_summary": generated_text,
                "rouge1": m["rouge1"],
                "rouge2": m["rouge2"],
                "rougeL": m["rougeL"],
                "factual_consistency": m["factual_consistency"],
            })

            tracker.update_peak()

        tracker.stop()
        mem_summary = tracker.get_summary()

        # 4. Aggregate operational and quality metrics
        avg_quality = self.metrics_calc.aggregate_metrics(sample_metrics)
        avg_latency = round(float(sum(latencies_ms) / len(latencies_ms)), 2) if latencies_ms else 0.0
        avg_ttft = round(float(sum(ttfts_ms) / len(ttfts_ms)), 2) if ttfts_ms else 0.0
        avg_tps = (
            round(float(sum(tokens_per_sec_list) / len(tokens_per_sec_list)), 2)
            if tokens_per_sec_list
            else 0.0
        )
        avg_input_len = round(total_input_tokens / max(1, len(eval_records)), 1)
        avg_output_len = round(total_output_tokens / max(1, len(eval_records)), 1)

        # Local cost calculation: Electricity + Hardware amortization
        # Electricity: 250W * (avg_latency_ms / 3.6e6) * $0.15/kWh
        local_kwh = 0.25 * (avg_latency / 3600000.0)
        cost_per_req = round(local_kwh * 0.15, 6)

        hw_info = detect_hardware().to_dict()

        result = {
            "evaluation_id": f"eval_{model_id.replace('/', '_')}_{int(time.time())}",
            "experiment_id": exp_id,
            "model_id": model_id,
            "dataset_name": dataset_name,
            "split": split,
            "num_samples": len(eval_records),
            "quality": avg_quality,
            "operational": {
                "avg_latency_ms": avg_latency,
                "avg_ttft_ms": avg_ttft,
                "tokens_per_second": avg_tps,
                "avg_input_tokens": avg_input_len,
                "avg_output_tokens": avg_output_len,
                "peak_memory_mb": mem_summary.get("peak_ram_mb", 0.0),
                "cost_per_request_usd": cost_per_req,
            },
            "hardware_info": hw_info,
            "generations_sample": generations[:5],
            "all_generations_count": len(generations),
            "created_at": datetime.datetime.utcnow().isoformat(),
        }

        # 5. Optional Frontier API Comparison
        frontier_comparison = None
        if compare_with_api:
            frontier_comparison = self._run_frontier_comparison(
                eval_records=eval_records,
                provider_name=compare_with_api,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            result["frontier_comparison"] = frontier_comparison

        # 6. Save raw benchmark file
        bench_file = self.output_dir / f"{result['evaluation_id']}.json"
        with open(bench_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved benchmark results to {bench_file}")

        # 7. Persist to SQLite Database
        self._save_to_database(result)

        # 8. Log experiment to MLflow
        self._log_to_mlflow(result)

        return result

    def _run_frontier_comparison(
        self,
        eval_records: List[Dict[str, Any]],
        provider_name: str,
        max_output_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Run the held-out BillSum sample set through the commercial frontier LLM."""
        provider = get_frontier_provider(provider_name=provider_name)
        frontier_gens = []
        frontier_metrics = []
        latencies = []
        costs = []

        for rec in eval_records:
            prompt = (
                "You are an expert legislative analyst. Provide a clear, factual, and concise summary "
                f"of the following congressional bill.\n\nBill Text:\n{rec['text']}\n\nSummary:"
            )
            resp = provider.generate(
                prompt=prompt,
                max_tokens=max_output_tokens,
                temperature=temperature,
            )
            latencies.append(resp.latency_ms)
            costs.append(resp.cost_usd)

            m = self.metrics_calc.evaluate_summary(
                reference=rec["summary"],
                candidate=resp.text,
                source_text=rec["text"],
            )
            frontier_metrics.append(m)

            frontier_gens.append({
                "bill_id": rec["id"],
                "generated_summary": resp.text,
                "latency_ms": resp.latency_ms,
                "cost_usd": resp.cost_usd,
                "rougeL": m["rougeL"],
            })

        avg_quality = self.metrics_calc.aggregate_metrics(frontier_metrics)
        avg_lat = round(float(sum(latencies) / len(latencies)), 2) if latencies else 0.0
        avg_cost = round(float(sum(costs) / len(costs)), 6) if costs else 0.0

        return {
            "provider": provider.model_id,
            "quality": avg_quality,
            "avg_latency_ms": avg_lat,
            "avg_cost_per_request_usd": avg_cost,
            "cost_per_1k_usd": round(avg_cost * 1000, 4),
            "samples": frontier_gens[:3],
        }

    def _save_to_database(self, result: Dict[str, Any]) -> None:
        """Store evaluation metrics in SQLite database."""
        session = SessionLocal()
        try:
            eval_rec = EvaluationRecord(
                id=result["evaluation_id"],
                experiment_id=result.get("experiment_id"),
                model_id=result["model_id"],
                dataset_name=result["dataset_name"],
                split=result["split"],
                num_samples=result["num_samples"],
                rouge1=result["quality"]["rouge1"],
                rouge2=result["quality"]["rouge2"],
                rougeL=result["quality"]["rougeL"],
                bert_score=result["quality"]["bert_score"],
                factual_consistency=result["quality"]["factual_consistency"],
                avg_input_length=result["operational"]["avg_input_tokens"],
                avg_output_length=result["operational"]["avg_output_tokens"],
                avg_latency_ms=result["operational"]["avg_latency_ms"],
                avg_ttft_ms=result["operational"]["avg_ttft_ms"],
                tokens_per_second=result["operational"]["tokens_per_second"],
                peak_memory_mb=result["operational"]["peak_memory_mb"],
                cost_per_request_usd=result["operational"]["cost_per_request_usd"],
                hardware_info=result["hardware_info"],
                generations_sample=result["generations_sample"],
            )
            session.add(eval_rec)

            exp_rec = ExperimentRecord(
                id=result.get("experiment_id", f"exp_{int(time.time())}"),
                name=f"Baseline: {result['model_id']}",
                model_id=result["model_id"],
                category="baseline",
                rouge1=result["quality"]["rouge1"],
                rouge2=result["quality"]["rouge2"],
                rougeL=result["quality"]["rougeL"],
                bert_score=result["quality"]["bert_score"],
                avg_latency_ms=result["operational"]["avg_latency_ms"],
                tokens_per_second=result["operational"]["tokens_per_second"],
                peak_memory_mb=result["operational"]["peak_memory_mb"],
                cost_per_1k_requests=round(result["operational"]["cost_per_request_usd"] * 1000, 4),
                status="completed",
                parameters={"split": result["split"], "num_samples": result["num_samples"]},
                metrics={**result["quality"], **result["operational"]},
            )
            session.merge(exp_rec)
            session.commit()
            logger.info("Saved evaluation to database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error persisting evaluation to database: {e}")
        finally:
            session.close()

    def _log_to_mlflow(self, result: Dict[str, Any]) -> None:
        """Log baseline evaluation run to MLflow."""
        try:
            import mlflow

            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///./mlruns.db")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("openllm_benchlab_evaluations")

            with mlflow.start_run(run_name=f"baseline_{result['model_id']}"):
                mlflow.log_param("model_id", result["model_id"])
                mlflow.log_param("dataset", result["dataset_name"])
                mlflow.log_param("split", result["split"])
                mlflow.log_param("num_samples", result["num_samples"])

                # Quality metrics
                for k, v in result["quality"].items():
                    mlflow.log_metric(f"quality_{k}", v)

                # Operational metrics
                for k, v in result["operational"].items():
                    mlflow.log_metric(f"operational_{k}", v)

            logger.info("Successfully logged evaluation to MLflow.")
        except Exception as e:
            logger.debug(f"MLflow logging skipped: {e}")

