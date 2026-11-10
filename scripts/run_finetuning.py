"""Script to execute LoRA fine-tuning, evaluate adapter, and compare with baseline."""

import json
import logging
import time
from ml.core.config import LoRAConfig as LoRASchema, TrainingConfig
from ml.training.adapter_manager import AdapterManager
from ml.training.trainer import FineTuningEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    print("=================================================================")
    print(" OpenLLM BenchLab: LoRA Parameter-Efficient Fine-Tuning Pipeline ")
    print("=================================================================")

    job_id = f"lora_qwen_billsum_{int(time.time())}"
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"

    lora_cfg = LoRASchema(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        use_qlora=False,
    )

    # Fast reproducible training configuration for interactive validation
    train_cfg = TrainingConfig(
        job_id=job_id,
        base_model_id=base_model,
        dataset_name="billsum",
        output_dir=f"artifacts/checkpoints/{job_id}",
        lora=lora_cfg,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=15,  # 15 gradient steps for rapid verification
        max_seq_length=256,
        logging_steps=5,
    )

    print(f"\n[1/3] Launching LoRA Fine-Tuning ({job_id})...")
    print(f"Base Model:       {base_model}")
    print(f"Target Modules:   {lora_cfg.target_modules}")
    print(f"Rank (r):         {lora_cfg.r}, Alpha: {lora_cfg.lora_alpha}")
    print(f"Learning Rate:    {train_cfg.learning_rate}")
    print(f"Batch Size:       {train_cfg.per_device_train_batch_size} (Grad Accum: {train_cfg.gradient_accumulation_steps})")

    engine = FineTuningEngine()
    summary = engine.train(train_cfg)

    print("\n------------------ TRAINING RUN SUMMARY ------------------")
    print(f"Status:               {summary['status'].upper()}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,} / {summary['total_parameters']:,} ({summary['trainable_percentage']}%)")
    print(f"Final Train Loss:     {summary['train_loss']}")
    print(f"Final Val Loss:       {summary['val_loss']}")
    print(f"Training Duration:    {summary['duration_seconds']}s")
    print(f"Peak Memory:          {summary['peak_memory_mb']} MB")
    print(f"Checkpoint Size:      {summary['checkpoint_size_mb']} MB")
    print(f"Checkpoint Saved To:  {summary['checkpoint_path']}")

    print("\n[2/3] Evaluating Fine-Tuned LoRA Adapter on BillSum Test Set...")
    adapter_mgr = AdapterManager()
    eval_res = adapter_mgr.evaluate_adapter(
        base_model_id=base_model,
        adapter_path=summary["checkpoint_path"],
        split="ca_test",
        num_samples=5,
        max_input_length=512,
        max_output_tokens=128,
        experiment_id=f"exp_{job_id}",
    )

    print("\n----------------- FINE-TUNED QUALITY METRICS -----------------")
    print(f"Fine-Tuned ROUGE-1:   {eval_res['quality']['rouge1']:.2f}")
    print(f"Fine-Tuned ROUGE-2:   {eval_res['quality']['rouge2']:.2f}")
    print(f"Fine-Tuned ROUGE-L:   {eval_res['quality']['rougeL']:.2f}")
    print(f"Factual Consistency:  {eval_res['quality']['factual_consistency']:.2f}%")
    print(f"Inference Latency:    {eval_res['operational']['avg_latency_ms']:.2f} ms")
    print(f"Inference TPS:        {eval_res['operational']['tokens_per_second']:.2f} tokens/sec")

    if "baseline_delta" in eval_res and "delta_rougeL" in eval_res["baseline_delta"]:
        delta = eval_res["baseline_delta"]
        print("\n-------------- DELTA VS ORIGINAL BASELINE MODEL --------------")
        print(f"Baseline ROUGE-L:     {delta['baseline_rougeL']:.2f}")
        print(f"Fine-Tuned ROUGE-L:   {delta['fine_tuned_rougeL']:.2f}")
        print(f"ROUGE-L Improvement:  {'+' if delta['delta_rougeL'] >= 0 else ''}{delta['delta_rougeL']:.2f} points")
        print(f"Latency Difference:   {'+' if delta['delta_latency_ms'] >= 0 else ''}{delta['delta_latency_ms']:.2f} ms")

    print("\n=================================================================")
    print("Fine-tuning pipeline execution successfully validated.")
    print("=================================================================")


if __name__ == "__main__":
    main()

