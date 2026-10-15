"""Tests for truncate_to_token_budget in dspy_logic/utils.py."""

import pytest

from nyha_temporal_identification.dspy_logic.utils import (
    SPAN_MARKER,
    SPAN_MARKER_END,
    build_classification_context,
    truncate_to_token_budget,
)


import re


class FakeTokenizer:
    """
    Deterministic fake tokenizer: each whitespace-separated word = 1 token.
    Supports the __call__ interface with return_offsets_mapping.
    """

    def _tokens(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def __call__(self, text: str, return_offsets_mapping: bool = False, add_special_tokens: bool = True) -> dict:
        offsets = self._tokens(text)
        result: dict = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def encode(self, text: str) -> list[int]:
        return list(range(len(self._tokens(text))))

    def decode(self, ids: list[int]) -> str:
        return " ".join(["x"] * len(ids))


class CharTokenizer:
    """
    Character-level tokenizer: each character = 1 token.
    Supports the __call__ interface with return_offsets_mapping.
    """

    def __call__(self, text: str, return_offsets_mapping: bool = False, add_special_tokens: bool = True) -> dict:
        result: dict = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_no_op_when_tokenizer_is_none():
    text = "hello world foo bar"
    result = truncate_to_token_budget(text, None, 3)
    assert result == text


def test_no_op_when_max_tokens_zero():
    tok = FakeTokenizer()
    text = "a b c d e f g"
    result = truncate_to_token_budget(text, tok, 0)
    assert result == text


def test_no_op_when_within_budget():
    tok = FakeTokenizer()
    text = "a b c"
    result = truncate_to_token_budget(text, tok, 100)
    assert result == text


def test_tail_truncation_without_markers():
    tok = CharTokenizer()
    text = "abcdefghij"  # 10 chars = 10 tokens
    result = truncate_to_token_budget(text, tok, 5)
    assert len(result) == 5
    assert result == "abcde"


def test_markers_preserved_under_truncation():
    """
    When the text has markers, they must survive truncation.
    The function should keep content around the >>> ... <<< block.
    """
    tok = CharTokenizer()
    # Build text where the span sits exactly in the middle
    prefix = "a" * 20
    span_content = f"{SPAN_MARKER}NYHA{SPAN_MARKER_END}"
    suffix = "b" * 20
    text = prefix + span_content + suffix

    budget = len(span_content) + 10  # only room for markers + 10 chars padding
    result = truncate_to_token_budget(text, tok, budget)

    assert SPAN_MARKER in result
    assert SPAN_MARKER_END in result


def test_output_length_respects_budget():
    """Output token count must be ≤ max_tokens."""
    tok = CharTokenizer()
    text = "x" * 100
    budget = 30
    result = truncate_to_token_budget(text, tok, budget)
    assert len(tok.encode(result)) <= budget


def test_short_text_returned_unchanged():
    tok = CharTokenizer()
    text = "abc"
    result = truncate_to_token_budget(text, tok, 10)
    assert result == text


def test_truncation_is_verbatim_prefix():
    """Result must be a verbatim substring of input, not a decode-reconstructed string."""
    tok = CharTokenizer()
    text = "abcdefghij"
    result = truncate_to_token_budget(text, tok, 5)
    assert result == "abcde"
    assert text.startswith(result)


def test_char_position_preserved_after_truncation():
    """Span at a fixed offset inside the window must still be readable after budget truncation.

    Regression: tokenizer.decode() was used to produce the truncated string, which
    could shift character positions and break span validation in normalise_examples.
    """
    tok = CharTokenizer()
    span = "NYHA II"
    span_pos = 1000
    # Window: 1000 prefix chars + span + 500 suffix = 1507 chars > budget of 1005
    text = "x" * span_pos + span + "y" * 500
    budget = span_pos + len(span)  # just enough to include the span
    result = truncate_to_token_budget(text, tok, budget)
    assert len(result) == budget
    assert result[span_pos : span_pos + len(span)] == span


# ── build_classification_context ──────────────────────────────────────────────


def test_bcc_injects_markers_with_window():
    span = "NYHA II"
    prefix = "patient is "
    text = prefix + span + " this year."
    start = len(prefix)
    end = start + len(span)
    result = build_classification_context(text, start, end, window=100)
    assert SPAN_MARKER in result
    assert SPAN_MARKER_END in result
    marker_pos = result.find(SPAN_MARKER)
    inner_start = marker_pos + len(SPAN_MARKER)
    assert result[inner_start : inner_start + len(span)] == span


def test_bcc_full_text_no_window_still_marks():
    span = "NYHA III"
    prefix = "note: "
    text = prefix + span + " historical."
    start = len(prefix)
    end = start + len(span)
    result = build_classification_context(text, start, end, window=None)
    assert result == text[: start] + SPAN_MARKER + span + SPAN_MARKER_END + text[end:]


def test_bcc_none_offsets_returns_raw():
    text = "some clinical note without a resolved span"
    result = build_classification_context(text, None, None, window=100)
    assert result == text
    assert SPAN_MARKER not in result


def test_bcc_token_cap_applied():
    tok = CharTokenizer()
    span = "NYHA I"
    prefix = "a" * 200
    text = prefix + span + "b" * 200
    start = len(prefix)
    end = start + len(span)
    budget = 50
    result = build_classification_context(text, start, end, window=1000, tokenizer=tok, max_tokens=budget)
    assert len(tok.encode(result)) <= budget


def test_bcc_markers_survive_cap():
    tok = CharTokenizer()
    span = "NYHA IV"
    prefix = "x" * 300
    suffix = "y" * 300
    text = prefix + span + suffix
    start = len(prefix)
    end = start + len(span)
    budget = 100
    result = build_classification_context(text, start, end, window=1000, tokenizer=tok, max_tokens=budget)
    assert SPAN_MARKER in result
    assert SPAN_MARKER_END in result
