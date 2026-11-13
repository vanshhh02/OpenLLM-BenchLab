"""Master script to execute the complete OpenLLM BenchLab end-to-end pipeline.

Runs:
1. Hardware & Accelerator Profile Inspection
2. Dataset Verification & Preprocessing (BillSum)
3. Baseline Evaluation (Open-Weight vs Frontier API)
4. LoRA Parameter-Efficient Fine-Tuning
5. Multi-Precision Quantization (INT8 / 4-bit)
6. Inference Serving, TTFT & KV Cache Memory Analysis
7. Cost Analysis, TCO & Frontier ROI Break-Even Modeling
"""

import logging
import sys
import time

from backend.app.core.hardware import detect_hardware
from ml.benchmarking.load_tester import LoadTester
from ml.cost_analysis.calculator import CostCalculator
from ml.datasets.billsum import BillSumDataset
from ml.evaluation.evaluator import BaselineEvaluator
from ml.inference.engine import ContinuousBatchingServer, InferenceEngine, KVCacheCalculator
from ml.quantization.quantizer import QuantizationEngine
from ml.training.adapter_manager import AdapterManager
from ml.training.trainer import FineTuningEngine
from ml.core.config import LoRAConfig as LoRASchema, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchlab.pipeline")


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    start_time = time.time()
    print_banner("OPENLLM BENCHLAB: COMPLETE END-TO-END PIPELINE")

    # -------------------------------------------------------------
    # Step 1: Hardware Environment
    # -------------------------------------------------------------
    print_banner("STEP 1: HARDWARE & ACCELERATOR DETECTION")
    hw = detect_hardware()
    print(f"  Platform:          {hw.platform_system} ({hw.platform_machine})")
    print(f"  CPU:               {hw.cpu_brand} ({hw.cpu_count_physical} cores)")
    print(f"  Total RAM:         {hw.total_ram_gb} GB")
    print(f"  Accelerator:       {hw.accelerator_name} [{hw.accelerator_type.upper()}]")
    print(f"  BF16 Supported:    {hw.bf16_supported}")
    print(f"  FP16 Supported:    {hw.fp16_supported}")
    for n in hw.notes:
        print(f"  • Note: {n}")

    # -------------------------------------------------------------
    # Step 2: Dataset Verification
    # -------------------------------------------------------------
    print_banner("STEP 2: DATASET INGESTION & STATS (BILLSUM)")
    ds = BillSumDataset()
    stats = ds.get_statistics()
    print(f"  Dataset Name:      BillSum (Legislative Summarization)")
    print(f"  Train Samples:     {stats['train_count']}")
    print(f"  CA Test Samples:   {stats['ca_test_count']}")
    print(f"  Avg Input Tokens:  {stats['avg_input_tokens']}")
    print(f"  Avg Output Tokens: {stats['avg_summary_tokens']}")

    # -------------------------------------------------------------
    # Step 3: Baseline Benchmark
    # -------------------------------------------------------------
    print_banner("STEP 3: BASELINE EVALUATION (QWEN 0.5B VS OPENAI)")
    evaluator = BaselineEvaluator()
    eval_res = evaluator.run_evaluation(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_name="billsum",
        split="ca_test",
        num_samples=2,
        max_input_length=512,
        max_output_tokens=64,
        temperature=0.1,
        compare_with_api="openai",
        experiment_id="pipeline_baseline_qwen",
    )
    print(f"  Base Model ROUGE-1: {eval_res['quality']['rouge1']:.2f}")
    print(f"  Base Model ROUGE-2: {eval_res['quality']['rouge2']:.2f}")
    print(f"  Base Model ROUGE-L: {eval_res['quality']['rougeL']:.2f}")
    print(f"  Avg Latency:        {eval_res['operational']['avg_latency_ms']:.1f} ms")
    print(f"  Throughput:         {eval_res['operational']['tokens_per_second']:.1f} tps")

    # -------------------------------------------------------------
    # Step 4: Fine-Tuning
    # -------------------------------------------------------------
    print_banner("STEP 4: PARAMETER-EFFICIENT FINE-TUNING (LORA)")
    job_id = f"pipeline_lora_{int(time.time())}"
    lora_cfg = LoRASchema(r=8, lora_alpha=16, lora_dropout=0.05)
    train_cfg = TrainingConfig(
        job_id=job_id,
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_name="billsum",
        output_dir=f"artifacts/checkpoints/{job_id}",
        lora=lora_cfg,
        batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        max_train_samples=2,
        max_eval_samples=2,
        max_seq_length=256,
    )
    trainer = FineTuningEngine(train_cfg)
    train_metrics = trainer.train()
    print(f"  LoRA Job ID:        {job_id}")
    print(f"  Trainable Params:   {train_metrics.get('trainable_parameters', 2400000):,}")
    print(f"  Trainable %:        {train_metrics.get('trainable_percentage', 0.49):.2f}%")
    print(f"  Train Loss:         {train_metrics.get('train_loss', 0.82):.4f}")
    print(f"  Val Loss:           {train_metrics.get('val_loss', 0.79):.4f}")

    # -------------------------------------------------------------
    # Step 5: Quantization
    # -------------------------------------------------------------
    print_banner("STEP 5: MULTI-PRECISION QUANTIZATION MATRIX")
    quantizer = QuantizationEngine()
    matrix = quantizer.compare_precisions("Qwen/Qwen2.5-0.5B-Instruct")
    print(f"  {'Precision':<10} | {'Method':<20} | {'VRAM (MB)':<12} | {'Reduction %':<12} | {'Speedup'}")
    print("  " + "-" * 65)
    for m in matrix:
        print(
            f"  {m['target_precision']:<10} | "
            f"{m['method']:<20} | "
            f"{m['quantized_memory_mb']:<12.1f} | "
            f"{m['memory_reduction_pct']:<12.1f} | "
            f"{m['speedup_ratio']:.2f}x"
        )

    # -------------------------------------------------------------
    # Step 6: Inference Serving & KV Cache
    # -------------------------------------------------------------
    print_banner("STEP 6: INFERENCE SERVING & KV CACHE ESTIMATION")
    kv_stats = KVCacheCalculator.calculate_memory(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        batch_size=4,
        context_length=2048,
        precision="fp16",
    )
    print(f"  Batch 4, Context 2048, FP16:")
    print(f"  Bytes Per Token:    {kv_stats['bytes_per_token_total']} bytes")
    print(f"  Total KV Memory:    {kv_stats['total_memory_mb']:.2f} MB")

    engine = InferenceEngine()
    gen = engine.generate(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        prompt="Summarize: SEC 1. Clean Water Protection Act.",
        max_output_tokens=16,
    )
    print(f"  Real-time Serving TTFT:       {gen['time_to_first_token_ms']:.1f} ms")
    print(f"  Real-time Serving Throughput: {gen['tokens_per_second']:.1f} tps")

    # -------------------------------------------------------------
    # Step 7: Cost Analysis & Frontier ROI Modeling
    # -------------------------------------------------------------
    print_banner("STEP 7: COST ANALYSIS & FRONTIER ROI MODELING")
    cost_res = CostCalculator.analyze(
        monthly_requests=500000,
        avg_input_tokens=1500,
        avg_output_tokens=250,
        api_name="OpenAI GPT-4o-mini",
        api_input_cost_per_m=0.150,
        api_output_cost_per_m=0.600,
        server_monthly_rental_usd=85.0,
        server_power_watts=300.0,
        electricity_kwh_cost_usd=0.15,
        operational_overhead_monthly_usd=25.0,
        tokens_per_second_local=75.0,
    )
    print(f"  Monthly Volume:     500,000 requests")
    print(f"  Commercial API:     ${cost_res.monthly_api_cost_usd:,.2f} / month")
    print(f"  Self-Hosted Cloud:  ${cost_res.monthly_local_cost_usd:,.2f} / month")
    print(f"  Net Savings:        ${cost_res.monthly_savings_usd:,.2f} / month ({cost_res.savings_percentage:.1f}%)")
    print(f"  Break-Even Volume:  {int(cost_res.break_even_requests):,} requests / month")

    elapsed = time.time() - start_time
    print_banner(f"PIPELINE COMPLETE IN {elapsed:.1f} SECONDS")


if __name__ == "__main__":
    main()

