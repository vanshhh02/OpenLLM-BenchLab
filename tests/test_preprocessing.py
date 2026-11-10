"""Unit tests for bill text normalization and summarization preprocessing."""

import pytest
from ml.preprocessing.summarization import (
    BillSummarizationPreprocessor,
    normalize_bill_text,
)


def test_normalize_bill_text():
    raw_text = '  SECTION 1.   SHORT TITLE.\n\n\nThis Act may be cited as the "Test Act"—2026.  '
    normalized = normalize_bill_text(raw_text)
    assert 'SECTION 1. SHORT TITLE.' in normalized
    assert '"Test Act"-2026.' in normalized
    assert '\n\n\n' not in normalized
    assert normalized.startswith("SECTION 1.")


def test_truncation_strategy():
    preprocessor = BillSummarizationPreprocessor(max_input_length=50)
    long_text = "word " * 200
    truncated, was_trunc, orig_tokens, final_tokens = preprocessor.truncate_text(long_text, max_tokens=50)
    assert was_trunc is True
    assert orig_tokens >= 200
    assert final_tokens <= 50
    assert len(truncated.split()) <= 50


def test_chunking_strategy():
    preprocessor = BillSummarizationPreprocessor()
    text = "word " * 1000
    chunks = preprocessor.chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 3
    assert len(chunks[0].split()) == 300


def test_process_record():
    preprocessor = BillSummarizationPreprocessor(max_input_length=100)
    raw = {
        "id": "bill-101",
        "title": "Clean Energy Act",
        "text": "SECTION 1. SHORT TITLE.\nThis is a long congressional bill text describing energy goals. " * 10,
        "summary": "This bill sets forth federal energy goals.",
    }
    processed = preprocessor.process_record(raw, strategy="truncation")
    assert processed.bill_id == "bill-101"
    assert processed.title == "Clean Energy Act"
    assert processed.reference_summary == "This bill sets forth federal energy goals."
    assert processed.input_token_count > 0
    assert processed.reference_token_count > 0

