"""Preprocessing and prompt formatting pipeline for congressional bill summarization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PreprocessedBill:
    bill_id: str
    original_text: str
    normalized_text: str
    processed_text: str
    reference_summary: str
    title: str
    original_char_count: int
    normalized_char_count: int
    input_token_count: int
    reference_token_count: int
    was_truncated: bool = False
    strategy_used: str = "truncation"
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_bill_text(text: str) -> str:
    """Clean and normalize congressional bill text.

    Removes excessive whitespace, standardizes unicode characters,
    strips non-printable artifacts, and removes leading/trailing boilerplate.
    """
    if not text:
        return ""

    # Normalize unicode to NFKC
    text = unicodedata.normalize("NFKC", text)

    # Standardize quotation marks and hyphens
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r"[–—]", "-", text)

    # Replace multiple newlines or tabs with consistent formatting
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip non-printable characters except standard whitespace
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")

    return text.strip()


class BillSummarizationPreprocessor:
    """Handles text normalization, prompt formatting, and long-document strategies."""

    DEFAULT_PROMPT_TEMPLATE = (
        "You are an expert legislative analyst. Provide a clear, factual, and concise summary "
        "of the following congressional bill.\n\n"
        "Bill Text:\n{text}\n\n"
        "Summary:"
    )

    def __init__(
        self,
        tokenizer: Any = None,
        max_input_length: int = 1024,
        max_output_length: int = 256,
        prompt_template: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    def count_tokens(self, text: str) -> int:
        """Accurately count tokens using HF tokenizer or whitespace approximation."""
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fallback estimation: ~4 chars per token in English text
        return max(1, len(text.split()))

    def truncate_text(self, text: str, max_tokens: int) -> Tuple[str, bool, int, int]:
        """Truncate text to fit within max_tokens budget, recording original & final lengths."""
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            orig_len = len(tokens)
            if orig_len > max_tokens:
                truncated_ids = tokens[:max_tokens]
                truncated_text = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
                return truncated_text, True, orig_len, max_tokens
            return text, False, orig_len, orig_len

        # Word-level fallback
        words = text.split()
        orig_len = len(words)
        if orig_len > max_tokens:
            truncated_text = " ".join(words[:max_tokens])
            return truncated_text, True, orig_len, max_tokens
        return text, False, orig_len, orig_len

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split document into overlapping chunks for chunk-based summarization."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        stride = max(1, chunk_size - overlap)
        for i in range(0, len(words), stride):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def format_prompt(self, text: str, system_prompt: Optional[str] = None) -> str:
        """Format the input prompt using either the chat template or the standard prompt."""
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({
                    "role": "system",
                    "content": "You are an expert legislative analyst who summarizes congressional bills factually and concisely."
                })
            messages.append({"role": "user", "content": f"Summarize the following bill:\n\n{text}"})

            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        return self.prompt_template.format(text=text)

    def process_record(
        self,
        raw_record: Dict[str, Any],
        record_id: Optional[str] = None,
        strategy: str = "truncation",
    ) -> PreprocessedBill:
        """Process a raw BillSum record into a structured PreprocessedBill instance."""
        text = raw_record.get("text", "")
        summary = raw_record.get("summary", "")
        title = raw_record.get("title", "")
        bid = record_id or raw_record.get("id", "bill-0")

        normalized = normalize_bill_text(text)
        cleaned_summary = normalize_bill_text(summary)

        orig_char_count = len(text)
        norm_char_count = len(normalized)

        if strategy == "truncation":
            processed_text, was_truncated, orig_tokens, final_tokens = self.truncate_text(
                normalized, self.max_input_length
            )
        else:
            processed_text = normalized
            was_truncated = False
            orig_tokens = self.count_tokens(normalized)
            final_tokens = orig_tokens

        ref_tokens = self.count_tokens(cleaned_summary)

        return PreprocessedBill(
            bill_id=str(bid),
            original_text=text,
            normalized_text=normalized,
            processed_text=processed_text,
            reference_summary=cleaned_summary,
            title=title,
            original_char_count=orig_char_count,
            normalized_char_count=norm_char_count,
            input_token_count=final_tokens,
            reference_token_count=ref_tokens,
            was_truncated=was_truncated,
            strategy_used=strategy,
            metadata={
                "original_tokens": orig_tokens,
                "final_tokens": final_tokens,
                "compression_ratio": round(final_tokens / max(1, ref_tokens), 2),
            },
        )

