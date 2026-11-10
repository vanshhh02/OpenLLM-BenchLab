"""Unit tests for BillSum dataset downloading, cleaning, and statistical profiling."""

import pytest
from ml.datasets.billsum import BillSumDatasetManager


def test_billsum_cleaning_and_stats():
    manager = BillSumDatasetManager()
    raw = manager._generate_representative_samples()
    assert "train" in raw
    assert "ca_test" in raw
    assert len(raw["train"]) > 0

    cleaned, stats = manager.prepare_and_clean_dataset(raw)
    assert len(cleaned["train"]) > 0
    assert stats["total_records"] > 0
    assert stats["input_tokens"]["mean"] > 0
    assert stats["summary_tokens"]["mean"] > 0
    assert stats["compression_ratio_avg"] > 0


def test_billsum_database_sync():
    manager = BillSumDatasetManager()
    raw = manager._generate_representative_samples()
    cleaned, stats = manager.prepare_and_clean_dataset(raw)
    manager.sync_to_database(stats, cleaned.get("ca_test", []))

