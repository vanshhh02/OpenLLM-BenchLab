"""Concurrent Load Testing and Latency Percentile Benchmarking Engine for OpenLLM BenchLab.

Features:
- Sweeps across configurable concurrency levels (e.g. 1, 5, 10, 25 concurrent requests).
- Calculates empirical latency percentiles (p50, p95, p99), average TTFT, RPS, and TPS.
- Computes dynamic KV cache memory consumption per concurrency level.
- Compares standard sequential execution with concurrent serving.
- Persists results to SQLite database (`benchmarks` table) and saves benchmark artifacts.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.database.models import BenchmarkRecord
from backend.app.database.session import SessionLocal
from ml.core.hardware import detect_hardware, MemoryTracker
from ml.inference.engine import InferenceEngine, KVCacheCalculator

logger = logging.getLogger("benchlab.benchmarking")


class LoadTester:
    """Orchestrates multi-concurrency load testing and operational profiling."""

    ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "benchmarks"

    DEFAULT_PROMPT = (
        "You are an expert legislative analyst. Summarize the following congressional bill:\n\n"
        "Section 1. Short Title. This Act may be cited as the Open Government and Data Transparency Act.\n"
        "Section 2. Purpose. The purpose of this Act is to require all executive branch agencies to make "
        "their datasets publicly accessible in open, machine-readable formats without paywalls or restrictive licenses.\n\n"
        "Summary:"
    )

    def __init__(self, artifact_dir: Optional[Path] = None):
        self.artifact_dir = artifact_dir or self.ARTIFACT_DIR
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.hardware = detect_hardware()

    def run_load_test_sweep(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        concurrency_levels: Optional[List[int]] = None,
        requests_per_level: int = 10,
        prompt: Optional[str] = None,
        max_output_tokens: int = 64,
        engine_type: str = "pytorch_mps" if detect_hardware().accelerator_type == "mps" else "pytorch",
    ) -> Dict[str, Any]:
        """Execute load test sweep across multiple concurrency levels."""
        if concurrency_levels is None:
            # Scale defaults based on available system RAM
            if self.hardware.total_ram_gb < 16:
                concurrency_levels = [1, 5, 10]
            else:
                concurrency_levels = [1, 5, 10, 25]

        prompt_text = prompt or self.DEFAULT_PROMPT
        sweep_id = f"loadtest_{int(time.time())}"
        logger.info(
            f"Starting load test sweep '{sweep_id}' for model={model_id}, "
            f"concurrency_levels={concurrency_levels}, requests_per_level={requests_per_level}"
        )

        engine = InferenceEngine(model_id=model_id)
        sweep_results: List[Dict[str, Any]] = []

        for concurrency in concurrency_levels:
            logger.info(f"Running concurrency level: {concurrency} (total requests: {requests_per_level})...")
            level_metrics = self._run_level(
                engine=engine,
                model_id=model_id,
                concurrency=concurrency,
                num_requests=requests_per_level,
                prompt=prompt_text,
                max_output_tokens=max_output_tokens,
                engine_type=engine_type,
                sweep_id=sweep_id,
            )
            sweep_results.append(level_metrics)

        # Build comparison summary
        summary = {
            "sweep_id": sweep_id,
            "model_id": model_id,
            "engine": engine_type,
            "concurrency_levels": concurrency_levels,
            "requests_per_level": requests_per_level,
            "levels": sweep_results,
            "created_at": datetime.datetime.utcnow().isoformat(),
        }

        # Save artifact JSON
        artifact_file = self.artifact_dir / f"{sweep_id}.json"
        with open(artifact_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved load test sweep report to {artifact_file}")

        return summary

    def _run_level(
        self,
        engine: InferenceEngine,
        model_id: str,
        concurrency: int,
        num_requests: int,
        prompt: str,
        max_output_tokens: int,
        engine_type: str,
        sweep_id: str,
    ) -> Dict[str, Any]:
        """Run load test for a single concurrency level."""
        latencies_ms: List[float] = []
        ttfts_ms: List[float] = []
        total_tokens_generated = 0
        success_count = 0
        fail_count = 0

        tracker = MemoryTracker()
        tracker.start()
        start_wall_time = time.perf_counter()

        def _worker_task() -> Dict[str, Any]:
            try:
                res = engine.generate(
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=0.1,
                )
                return {"status": "ok", "res": res}
            except Exception as e:
                logger.error(f"Request failed under concurrency {concurrency}: {e}")
                return {"status": "error", "error": str(e)}

        # Execute concurrent requests using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_worker_task) for _ in range(num_requests)]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    out = fut.result()
                    if out["status"] == "ok":
                        res = out["res"]
                        success_count += 1
                        latencies_ms.append(res["total_latency_ms"])
                        ttfts_ms.append(res["time_to_first_token_ms"])
                        total_tokens_generated += res["completion_tokens"]
                    else:
                        fail_count += 1
                except Exception:
                    fail_count += 1

        end_wall_time = time.perf_counter()
        tracker.stop()
        mem_info = tracker.get_summary()

        total_wall_sec = max(0.001, end_wall_time - start_wall_time)

        # Calculate percentiles using numpy
        if latencies_ms:
            p50 = float(np.percentile(latencies_ms, 50))
            p95 = float(np.percentile(latencies_ms, 95))
            p99 = float(np.percentile(latencies_ms, 99))
            avg_lat = float(np.mean(latencies_ms))
            avg_ttft = float(np.mean(ttfts_ms))
        else:
            p50 = p95 = p99 = avg_lat = avg_ttft = 0.0

        rps = round(success_count / total_wall_sec, 2)
        tps = round(total_tokens_generated / total_wall_sec, 2)

        # Calculate KV cache memory footprint for this concurrency level
        # Assuming prompt length ~60 tokens + generated ~64 tokens = context ~128 tokens
        kv_stats = KVCacheCalculator.calculate_memory(
            batch_size=concurrency,
            context_length=128,
            num_layers=24,
            num_kv_heads=2,
            head_dim=64,
            precision="fp16",
        )
        kv_cache_mb = kv_stats["total_memory_mb"]

        level_record_id = f"bench_{sweep_id}_c{concurrency}"
        result = {
            "id": level_record_id,
            "model_id": model_id,
            "benchmark_type": "concurrent_load_test",
            "concurrency_level": concurrency,
            "num_requests": num_requests,
            "successful_requests": success_count,
            "failed_requests": fail_count,
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "avg_ttft_ms": round(avg_ttft, 2),
            "requests_per_second": rps,
            "tokens_per_second": tps,
            "peak_memory_mb": mem_info.get("peak_ram_mb", 0.0),
            "engine": engine_type,
            "kv_cache_usage_mb": kv_cache_mb,
            "metrics_json": {
                "total_wall_time_sec": round(total_wall_sec, 2),
                "total_tokens_generated": total_tokens_generated,
                "kv_cache_details": kv_stats,
            },
            "created_at": datetime.datetime.utcnow().isoformat(),
        }

        self._save_to_database(result)
        return result

    def _save_to_database(self, result: Dict[str, Any]) -> None:
        """Persist benchmark execution record to SQLite database."""
        session = SessionLocal()
        try:
            rec = BenchmarkRecord(
                id=result["id"],
                model_id=result["model_id"],
                benchmark_type=result["benchmark_type"],
                concurrency_level=result["concurrency_level"],
                num_requests=result["num_requests"],
                successful_requests=result["successful_requests"],
                failed_requests=result["failed_requests"],
                p50_latency_ms=result["p50_latency_ms"],
                p95_latency_ms=result["p95_latency_ms"],
                p99_latency_ms=result["p99_latency_ms"],
                avg_latency_ms=result["avg_latency_ms"],
                avg_ttft_ms=result["avg_ttft_ms"],
                requests_per_second=result["requests_per_second"],
                tokens_per_second=result["tokens_per_second"],
                peak_memory_mb=result["peak_memory_mb"],
                engine=result["engine"],
                kv_cache_usage_mb=result["kv_cache_usage_mb"],
                metrics_json=result["metrics_json"],
                created_at=datetime.datetime.utcnow(),
            )
            session.merge(rec)
            session.commit()
            logger.info(f"Persisted benchmark record {result['id']} to database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to persist benchmark record: {e}")
        finally:
            session.close()

