"""Tests for data_handling/data_conversion.py (subsample_examples + explode)."""

from types import SimpleNamespace

import dspy
import pytest

from nyha_temporal_identification.data_handling.data_conversion import (
    documents_to_dspy_examples,
    subsample_examples,
)
from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.dspy_logic.utils import SPAN_MARKER


def _make_example(nyha_class: str) -> dspy.Example:
    return dspy.Example(nyha_class=nyha_class, text="dummy").with_inputs("text")


def _examples(class_counts: dict[str, int]) -> list[dspy.Example]:
    result = []
    for cls, n in class_counts.items():
        result.extend(_make_example(cls) for _ in range(n))
    return result


def test_no_op_when_max_size_zero():
    examples = _examples({"I": 5, "II": 5})
    result = subsample_examples(examples, max_size=0)
    assert result is examples


def test_no_op_when_within_budget():
    examples = _examples({"I": 3, "II": 3})
    result = subsample_examples(examples, max_size=10)
    assert result is examples


def test_output_size_capped():
    examples = _examples({"I": 20, "II": 20, "III": 20, "IV": 20})
    result = subsample_examples(examples, max_size=30)
    assert len(result) <= 30


def test_min_per_class_floor_respected():
    # 1 example of class IV among many of class I
    examples = _examples({"I": 40, "IV": 1})
    result = subsample_examples(examples, max_size=10, min_per_class=1)
    classes = [ex.nyha_class for ex in result]
    assert "IV" in classes


def test_deterministic_with_same_seed():
    examples = _examples({"I": 50, "II": 50})
    r1 = subsample_examples(examples, max_size=20, seed=42)
    r2 = subsample_examples(examples, max_size=20, seed=42)
    assert [ex.nyha_class for ex in r1] == [ex.nyha_class for ex in r2]


def test_different_seeds_may_differ():
    examples = _examples({"I": 50, "II": 50})
    r1 = subsample_examples(examples, max_size=20, seed=1)
    r2 = subsample_examples(examples, max_size=20, seed=2)
    # It's extremely unlikely both orderings are identical; assert at least one differs
    assert [ex.nyha_class for ex in r1] != [ex.nyha_class for ex in r2]


def test_no_class_attribute_falls_back_to_other_bucket():
    """Examples without nyha_class should go into 'other' bucket without raising."""
    examples = [
        dspy.Example(text="no class here").with_inputs("text")
        for _ in range(20)
    ]
    result = subsample_examples(examples, max_size=5, min_per_class=1)
    assert len(result) <= 5


def test_all_in_one_class_still_capped():
    examples = _examples({"I": 100})
    result = subsample_examples(examples, max_size=10)
    assert len(result) <= 10


# ── documents_to_dspy_examples: explode parameter ──────────────────────────────


def _cfg(windowed: bool) -> SimpleNamespace:
    """Minimal Config stub; documents_to_dspy_examples only reads this field."""
    return SimpleNamespace(use_windowed_classification_context=windowed)


def _single_mention_note() -> NYHANote:
    text = "patient has nyha ii noted in 2020 here."
    #               ^12     ^19
    return NYHANote(
        old_note_id="n1",
        deidentified_text=text,
        authored_datetime="2021-01-01",
        mentions=[
            NYHAMention(
                char_start=12, char_end=19, span="nyha ii",
                nyha_class="II", temporality="current",
            )
        ],
    )


def _two_mention_note() -> NYHANote:
    text = "nyha i in 2019 and nyha iii currently."
    #       ^0   ^6            ^19      ^27
    return NYHANote(
        old_note_id="n2",
        deidentified_text=text,
        authored_datetime="2021-01-01",
        mentions=[
            NYHAMention(
                char_start=0, char_end=6, span="nyha i",
                nyha_class="I", temporality="historical",
            ),
            NYHAMention(
                char_start=19, char_end=27, span="nyha iii",
                nyha_class="III", temporality="current",
            ),
        ],
    )


def test_explode_false_keeps_document_level_examples():
    """explode=False → one example per note, all gold mentions preserved,
    even when the windowed-context config is on."""
    docs = [_single_mention_note(), _two_mention_note()]
    result = documents_to_dspy_examples(docs, _cfg(windowed=True), explode=False)

    assert len(result) == 2  # one example per document
    assert [len(ex.mentions) for ex in result] == [1, 2]
    # Document-level examples carry the full note text, not a window/excerpt.
    assert result[0].deidentified_text == docs[0].deidentified_text
    # Document-level examples do not carry per-mention single-span fields.
    assert not hasattr(result[0], "span")


def test_explode_true_explodes_one_example_per_mention():
    """explode=True → one example per mention regardless of the config flag."""
    docs = [_single_mention_note(), _two_mention_note()]
    result = documents_to_dspy_examples(docs, _cfg(windowed=False), explode=True)

    assert len(result) == 3  # 1 + 2 mentions
    # Exploded examples carry per-mention single-span fields.
    assert all(hasattr(ex, "span") and hasattr(ex, "nyha_class") for ex in result)
    assert all(len(ex.mentions) == 1 for ex in result)


def test_explode_none_defaults_to_config_flag():
    """explode=None mirrors config.use_windowed_classification_context."""
    docs = [_single_mention_note(), _two_mention_note()]

    windowed = documents_to_dspy_examples(docs, _cfg(windowed=True))  # explode=None
    full = documents_to_dspy_examples(docs, _cfg(windowed=False))  # explode=None

    assert len(windowed) == 3  # exploded per mention
    assert len(full) == 2  # document-level


# ── Marker injection in exploded demo examples ─────────────────────────────────


def test_explode_true_demo_text_has_markers():
    """Windowed demo examples must carry >>> / <<< markers — same format as live inference."""
    docs = [_single_mention_note(), _two_mention_note()]
    result = documents_to_dspy_examples(docs, _cfg(windowed=True), explode=True)
    for ex in result:
        assert SPAN_MARKER in ex.deidentified_text, (
            f"Missing >>> marker in demo text: {ex.deidentified_text!r}"
        )


def test_explode_true_rebased_offsets_read_span():
    """char_start/char_end in the rebased mention must index the span inside the marked demo text."""
    docs = [_single_mention_note(), _two_mention_note()]
    result = documents_to_dspy_examples(docs, _cfg(windowed=True), explode=True)
    for ex in result:
        m = ex.mentions[0]
        actual = ex.deidentified_text[m.char_start : m.char_end]
        assert actual.strip().lower() == m.span.strip().lower(), (
            f"Rebased offsets ({m.char_start}, {m.char_end}) read {actual!r}, "
            f"expected {m.span!r}"
        )


def test_explode_true_linked_date_dropped():
    """linked_date must be None after normalisation — offsets are invalid in marked coords."""
    docs = [_single_mention_note(), _two_mention_note()]
    result = documents_to_dspy_examples(docs, _cfg(windowed=True), explode=True)
    for ex in result:
        assert ex.mentions[0].linked_date is None
