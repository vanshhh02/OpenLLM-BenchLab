"""Unit tests for quality metrics calculation: ROUGE, factual consistency, and aggregation."""

import pytest
from ml.evaluation.metrics import QualityMetricsCalculator


def test_compute_rouge_exact_match():
    calc = QualityMetricsCalculator()
    text = "The quick brown fox jumps over the lazy dog."
    scores = calc.compute_rouge(reference=text, candidate=text)
    assert scores["rouge1"] == 100.0
    assert scores["rouge2"] == 100.0
    assert scores["rougeL"] == 100.0


def test_compute_rouge_partial_match():
    calc = QualityMetricsCalculator()
    ref = "The President signs the legislative bill into federal law."
    cand = "The President signs a bill into law."
    scores = calc.compute_rouge(reference=ref, candidate=cand)
    assert 50.0 < scores["rouge1"] < 100.0
    assert scores["rougeL"] > 50.0


def test_factual_consistency_grounding():
    calc = QualityMetricsCalculator()
    source = (
        "SECTION 2. The Secretary shall authorize $500,000,000 for fiscal years 2026 through 2030 "
        "to modernize electricity grids and reduce outage rates."
    )
    # Factually consistent candidate (matches figures and key concepts)
    grounded_cand = "Authorizes $500,000,000 for fiscal years 2026-2030 to modernize electricity grids."
    score_grounded = calc.assess_factual_consistency(source, grounded_cand)

    # Hallucinated candidate with invented numbers
    hallucinated_cand = "Allocates $999,999,999 for NASA space exploration and orbital defense."
    score_hallucinated = calc.assess_factual_consistency(source, hallucinated_cand)

    assert score_grounded > score_hallucinated
    assert score_grounded >= 70.0


def test_aggregate_metrics():
    calc = QualityMetricsCalculator()
    samples = [
        {"rouge1": 40.0, "rouge2": 20.0, "rougeL": 35.0, "factual_consistency": 80.0},
        {"rouge1": 50.0, "rouge2": 30.0, "rougeL": 45.0, "factual_consistency": 90.0},
    ]
    agg = calc.aggregate_metrics(samples)
    assert agg["rouge1"] == 45.0
    assert agg["rouge2"] == 25.0
    assert agg["rougeL"] == 40.0
    assert agg["factual_consistency"] == 85.0

