"""Script to execute Phase 2 baseline evaluation on BillSum dataset."""

import json
import logging
from ml.evaluation.evaluator import BaselineEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    print("=================================================================")
    print(" OpenLLM BenchLab: Baseline Open-Weight vs Frontier Benchmarking ")
    print("=================================================================")

    evaluator = BaselineEvaluator()
    results = evaluator.run_evaluation(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_name="billsum",
        split="ca_test",
        num_samples=10,
        max_input_length=512,
        max_output_tokens=128,
        temperature=0.1,
        compare_with_api="openai",
        experiment_id="exp_baseline_qwen0.5b_billsum",
    )

    print("\n---------------------- QUALITY METRICS ----------------------")
    print(f"ROUGE-1:             {results['quality']['rouge1']:.2f}")
    print(f"ROUGE-2:             {results['quality']['rouge2']:.2f}")
    print(f"ROUGE-L:             {results['quality']['rougeL']:.2f}")
    print(f"BERTScore (est):     {results['quality']['bert_score']:.2f}")
    print(f"Factual Consistency: {results['quality']['factual_consistency']:.2f}%")

    print("\n-------------------- OPERATIONAL METRICS --------------------")
    print(f"Average Latency:     {results['operational']['avg_latency_ms']:.2f} ms")
    print(f"Time to First Token: {results['operational']['avg_ttft_ms']:.2f} ms")
    print(f"Throughput:          {results['operational']['tokens_per_second']:.2f} tokens/sec")
    print(f"Peak Memory:         {results['operational']['peak_memory_mb']:.2f} MB")
    print(f"Cost per 1k Req:     ${results['operational']['cost_per_request_usd'] * 1000:.6f}")

    if "frontier_comparison" in results:
        fc = results["frontier_comparison"]
        print("\n---------------- FRONTIER COMMERCIAL COMPARISON -------------")
        print(f"Frontier Model:      {fc['provider']}")
        print(f"Frontier ROUGE-L:    {fc['quality']['rougeL']:.2f}")
        print(f"Frontier Latency:    {fc['avg_latency_ms']:.2f} ms")
        print(f"Frontier Cost / 1k:  ${fc['cost_per_1k_usd']:.4f}")

    print("\n=================================================================")
    print(f"Evaluation ID: {results['evaluation_id']}")
    print("Benchmark results successfully registered in database and artifacts.")
    print("=================================================================")


if __name__ == "__main__":
    main()

