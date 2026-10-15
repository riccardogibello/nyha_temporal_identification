"""Tests for leakage-safe failsafe re-injection.

Covers:
- reinject_dropped_mentions: coverage-based gate, correct promotion
- assess_current_consistency: balanced / conservative / off policies
- NYHAMention.source round-trip (back-compat default)
- spacy_candidate_recall_by_temporality diagnostic
- NYHAExtractor.forward() with failsafe_dropped_current on/off
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nyha_temporal_identification.data_model.dspy_model import (
    NYHAMention,
    NYHANote,
)
from nyha_temporal_identification.data_model.general import NyhaMention, ParsedDate
from nyha_temporal_identification.dspy_logic.failsafe import (
    assess_current_consistency,
    reinject_dropped_mentions,
    reinject_uncovered_regex_spans,
)


# ── Helper factories ──────────────────────────────────────────────────────────


def _nyha_mention(
    start: int = 0,
    end: int = 8,
    span: str = "NYHA III",
    nyha_class: str = "III",
    temporality: str = "current",
    source: str = "llm",
) -> NYHAMention:
    return NYHAMention(
        char_start=start,
        char_end=end,
        span=span,
        nyha_class=nyha_class,
        temporality=temporality,
        source=source,
    )


def _spacy_mention(
    original_start: int = 0,
    original_end: int = 8,
    text: str = "NYHA III",
    classes: list[int] | None = None,
) -> NyhaMention:
    m = NyhaMention(
        original_text=text,
        cleaned_text=text,
        classes=classes or [3],
        one_hot_vector=[0, 0, 1, 0],
        start=original_start,
        end=original_end,
        original_start=original_start,
        original_end=original_end,
        linked_date=None,
        char_distance=None,
    )
    return m


# ── NYHAMention.source round-trip ─────────────────────────────────────────────


def test_source_default_is_llm():
    m = _nyha_mention()
    assert m.source == "llm"


def test_source_failsafe_roundtrip():
    m = _nyha_mention(source="failsafe")
    dumped = m.model_dump()
    restored = NYHAMention(**dumped)
    assert restored.source == "failsafe"


def test_source_back_compat_absent_key():
    """Deserialising old JSON without 'source' key must default to 'llm'."""
    data = {
        "char_start": 0,
        "char_end": 8,
        "span": "NYHA II",
        "nyha_class": "II",
        "temporality": "current",
        # no 'source' key
    }
    m = NYHAMention(**data)
    assert m.source == "llm"


def test_source_not_in_dedupe_key():
    """source field must not affect deduplication equality."""
    from nyha_temporal_identification.data_model.nyha_mention_deduplication import (
        unique_mentions,
    )

    m_llm = _nyha_mention(source="llm")
    m_failsafe = _nyha_mention(source="failsafe")
    # They differ only in source; unique_mentions must keep both (source excluded
    # from key means they ARE distinct — the spec says source is NOT in key).
    # Actually re-read spec: "source is NOT part of _mention_dedupe_key" means
    # two mentions identical in all other fields but differing in source count
    # as ONE unique mention (dedup removes the duplicate).
    unique = unique_mentions([m_llm, m_failsafe])
    assert len(unique) == 1


# ── reinject_dropped_mentions ─────────────────────────────────────────────────


def test_covered_candidate_not_reinjected():
    candidate = _spacy_mention(0, 8, "NYHA III")
    survivor = _nyha_mention(0, 8, "NYHA III", source="llm")
    result = reinject_dropped_mentions([candidate], [survivor])
    assert result == []


def test_uncovered_candidate_is_reinjected_as_current_failsafe():
    candidate = _spacy_mention(0, 8, "NYHA III", classes=[3])
    result = reinject_dropped_mentions([candidate], [])
    assert len(result) == 1
    m = result[0]
    assert m.source == "failsafe"
    assert m.temporality == "current"
    assert m.char_start == 0
    assert m.char_end == 8
    assert m.get_nyha_class() == "III"


def test_reinject_uses_drop_reasons_in_evidence():
    candidate = _spacy_mention(0, 8, "NYHA II", classes=[2])
    drop_reasons = {id(candidate): "classify_failed"}
    result = reinject_dropped_mentions([candidate], [], drop_reasons=drop_reasons)
    assert len(result) == 1
    assert "classify_failed" in result[0].evidence


def test_reinject_uses_uncovered_when_no_reason():
    candidate = _spacy_mention(0, 8, "NYHA II", classes=[2])
    result = reinject_dropped_mentions([candidate], [])
    assert "uncovered" in result[0].evidence


def test_reinject_partial_coverage():
    c1 = _spacy_mention(0, 8, "NYHA II", classes=[2])
    c2 = _spacy_mention(20, 30, "NYHA III", classes=[3])
    survivor = _nyha_mention(0, 8, "NYHA II", nyha_class="II")
    result = reinject_dropped_mentions([c1, c2], [survivor])
    # c1 is covered; c2 is not
    assert len(result) == 1
    assert result[0].char_start == 20


def test_broad_range_candidate_reinjects_via_normal_path():
    """
    Under the canonical convention a spaCy "NYHA II-IV" candidate collapses to a
    single class II ([0,1,0,0]), so it promotes via the normal from_nyha_mention
    path — keeping its class — rather than the bare fallback.
    """
    candidate = NyhaMention(
        original_text="NYHA II-IV",
        cleaned_text="NYHA II-IV",
        classes=[2],
        one_hot_vector=[0, 1, 0, 0],
        start=0,
        end=10,
        original_start=1133,
        original_end=1143,
        linked_date=None,
        char_distance=691,
    )
    result = reinject_dropped_mentions([candidate], [])
    assert len(result) == 1
    m = result[0]
    assert m.source == "failsafe"
    assert m.temporality == "current"
    assert m.char_start == 1133
    assert m.char_end == 1143
    assert m.get_nyha_class() == "II"  # class preserved (normal path)
    assert "class dropped" not in m.evidence
    NYHAMention(**m.model_dump())  # round-trips without raising


def test_malformed_noncontiguous_candidate_falls_back_to_bare_span():
    """
    Defensive-only: a hand-built non-contiguous candidate (which spaCy can no
    longer produce) must NOT crash re-injection — it re-injects the bare span as
    current/failsafe with the class cleared.
    """
    candidate = NyhaMention(
        original_text="NYHA II-IV",
        cleaned_text="NYHA II-IV",
        classes=[2, 4],
        one_hot_vector=[0, 1, 0, 1],
        start=0,
        end=10,
        original_start=1133,
        original_end=1143,
        linked_date=None,
        char_distance=691,
    )
    result = reinject_dropped_mentions([candidate], [])
    assert len(result) == 1
    m = result[0]
    assert m.source == "failsafe"
    assert m.temporality == "current"
    assert m.get_nyha_class() is None  # class dropped — irrelevant to masking
    assert "class dropped" in m.evidence
    NYHAMention(**m.model_dump())


def test_malformed_candidate_still_reinjected_alongside_valid():
    """One bad-vector candidate must not stop a valid candidate being re-injected."""
    bad = NyhaMention(
        original_text="NYHA I-IV",
        cleaned_text="NYHA I-IV",
        classes=[1, 4],
        one_hot_vector=[1, 0, 0, 1],
        start=0,
        end=9,
        original_start=0,
        original_end=9,
        linked_date=None,
        char_distance=None,
    )
    good = _spacy_mention(20, 28, "NYHA II", classes=[2])
    good.one_hot_vector = [0, 1, 0, 0]
    result = reinject_dropped_mentions([bad, good], [])
    assert len(result) == 2
    assert all(m.source == "failsafe" and m.temporality == "current" for m in result)


def test_regex_reinject_broad_range_span_keeps_collapsed_vector():
    """
    A regex "NYHA II-IV" match collapses to single class II ([0,1,0,0]) — a valid
    vector — so the regex pass stores it and the span round-trips.
    """
    text = "Patient met NYHA II-IV klachten."
    result = reinject_uncovered_regex_spans(text, survivors=[], existing_failsafe=[])
    assert len(result) >= 1
    m = result[0]
    assert m.source == "failsafe"
    assert m.temporality == "current"
    assert m.nyha_vector == [0, 1, 0, 0]  # collapsed to class II
    NYHAMention(**m.model_dump())  # round-trips without raising


def test_regex_reinject_i_iii_span_is_bare():
    """
    An "NYHA I-III" surface is not a recognised class (parse → None), so the regex
    pass re-injects a bare span (no vector) — still masked, leakage-safe.
    """
    text = "Patient met NYHA I-III klachten."
    result = reinject_uncovered_regex_spans(text, survivors=[], existing_failsafe=[])
    assert len(result) >= 1
    m = result[0]
    assert m.source == "failsafe"
    assert m.temporality == "current"
    assert m.nyha_vector == []  # I-III has no class
    NYHAMention(**m.model_dump())  # round-trips without raising


# ── assess_current_consistency ────────────────────────────────────────────────


def test_balanced_no_conflict_single_confident():
    mentions = [_nyha_mention(temporality="current", source="llm", nyha_class="II")]
    r = assess_current_consistency(mentions, policy="balanced")
    assert r["note_status"] == "ok"
    assert r["policy"] == "balanced"
    assert r["reinjected_count"] == 0


def test_balanced_conflict_on_two_confident_currents():
    m1 = _nyha_mention(0, 8, "NYHA II", "II", "current", "llm")
    m2 = _nyha_mention(10, 18, "NYHA III", "III", "current", "llm")
    r = assess_current_consistency([m1, m2], policy="balanced")
    assert r["note_status"] == "conflict"


def test_balanced_no_conflict_failsafe_only_discrepancy():
    """Balanced ignores failsafe-vs-confident disagreement."""
    m_llm = _nyha_mention(0, 8, "NYHA II", "II", "current", "llm")
    m_fs = _nyha_mention(10, 18, "NYHA III", "III", "current", "failsafe")
    r = assess_current_consistency([m_llm, m_fs], policy="balanced")
    assert r["note_status"] == "ok"
    assert r["reinjected_count"] == 1
    assert "III" in r["failsafe_current_classes"]


def test_conservative_flags_failsafe_vs_confident():
    m_llm = _nyha_mention(0, 8, "NYHA II", "II", "current", "llm")
    m_fs = _nyha_mention(10, 18, "NYHA III", "III", "current", "failsafe")
    r = assess_current_consistency([m_llm, m_fs], policy="conservative")
    assert r["note_status"] == "conflict"


def test_off_always_ok():
    m1 = _nyha_mention(0, 8, "NYHA II", "II", "current", "llm")
    m2 = _nyha_mention(10, 18, "NYHA III", "III", "current", "llm")
    r = assess_current_consistency([m1, m2], policy="off")
    assert r["note_status"] == "ok"


def test_balanced_two_failsafe_same_class_no_conflict():
    """Two failsafe mentions with the same class → ok under balanced."""
    m1 = _nyha_mention(0, 8, "NYHA II", "II", "current", "failsafe")
    m2 = _nyha_mention(10, 18, "NYHA II", "II", "current", "failsafe")
    r = assess_current_consistency([m1, m2], policy="balanced")
    assert r["note_status"] == "ok"


def test_historical_mentions_ignored():
    """Historical mentions (llm) must never contribute to conflict detection."""
    m_hist = _nyha_mention(0, 8, "NYHA II", "II", "historical", "llm")
    m_curr = _nyha_mention(10, 18, "NYHA III", "III", "current", "llm")
    r = assess_current_consistency([m_hist, m_curr], policy="balanced")
    assert r["note_status"] == "ok"


# ── spacy_candidate_recall_by_temporality ─────────────────────────────────────


def test_spacy_recall_covered_current():
    """Gold current mention covered by a spaCy candidate → recall=1.0."""
    gold_m = _nyha_mention(0, 8, "NYHA III", "III", "current")
    doc = NYHANote(
        old_note_id="n1",
        deidentified_text="NYHA III patient",
        mentions=[gold_m],
    )

    fake_candidate = _spacy_mention(0, 8, "NYHA III", [3])

    from nyha_temporal_identification.evaluation.evaluation import (
        spacy_candidate_recall_by_temporality,
    )

    with patch(
        "nyha_temporal_identification.dspy_logic.regex.spacy_preprocess",
        return_value=[fake_candidate],
    ):
        result = spacy_candidate_recall_by_temporality([doc])

    assert result["current"]["gold"] == 1
    assert result["current"]["covered_by_spacy"] == 1
    assert result["current"]["recall"] == 1.0


def test_spacy_recall_uncovered_current_lowers_recall():
    """Gold current not covered by any spaCy candidate → recall < 1.0."""
    gold_m = _nyha_mention(100, 108, "NYHA III", "III", "current")
    doc = NYHANote(
        old_note_id="n1",
        deidentified_text="A" * 200,
        mentions=[gold_m],
    )

    from nyha_temporal_identification.evaluation.evaluation import (
        spacy_candidate_recall_by_temporality,
    )

    # spaCy returns a candidate at a completely different position (no overlap).
    non_overlapping = _spacy_mention(0, 8, "NYHA II", [2])

    with patch(
        "nyha_temporal_identification.dspy_logic.regex.spacy_preprocess",
        return_value=[non_overlapping],
    ):
        result = spacy_candidate_recall_by_temporality([doc])

    assert result["current"]["gold"] == 1
    assert result["current"]["covered_by_spacy"] == 0
    assert result["current"]["recall"] == 0.0


def test_spacy_recall_empty_spacy_output():
    gold_m = _nyha_mention(0, 8, "NYHA III", "III", "current")
    doc = NYHANote(
        old_note_id="n1",
        deidentified_text="NYHA III",
        mentions=[gold_m],
    )

    from nyha_temporal_identification.evaluation.evaluation import (
        spacy_candidate_recall_by_temporality,
    )

    with patch(
        "nyha_temporal_identification.dspy_logic.regex.spacy_preprocess",
        return_value=[],
    ):
        result = spacy_candidate_recall_by_temporality([doc])

    assert result["current"]["recall"] == 0.0


def test_spacy_recall_no_gold_defaults_to_one():
    doc = NYHANote(
        old_note_id="n1",
        deidentified_text="No NYHA here",
        mentions=[],
    )

    from nyha_temporal_identification.evaluation.evaluation import (
        spacy_candidate_recall_by_temporality,
    )

    with patch(
        "nyha_temporal_identification.dspy_logic.regex.spacy_preprocess",
        return_value=[],
    ):
        result = spacy_candidate_recall_by_temporality([doc])

    # No gold → both temporalities absent from result (empty dict) or recall=1.0
    # depending on which temps appeared. With no gold, counts dict is empty.
    assert "current" not in result or result["current"]["gold"] == 0


# ── NYHAExtractor.forward() integration ───────────────────────────────────────


class _StubExtractor:
    """Minimal stub that mimics the NYHAExtractor interface for failsafe tests."""

    def __init__(self, failsafe_dropped_current: bool = False):
        self.failsafe_dropped_current = failsafe_dropped_current
        self.use_identification = False
        self.use_windowed_classification_context = False
        self.tokenizer = None
        self.max_eval_tokens = 0
        self.max_retries = 1


def test_forward_failsafe_off_no_reinsertion():
    """With failsafe_dropped_current=False, forward() returns only LLM survivors."""
    from nyha_temporal_identification.dspy_logic.modules import NYHAExtractorSpacyOnly

    extractor = NYHAExtractorSpacyOnly(failsafe_dropped_current=False)

    text = "Patiënt heeft NYHA III klachten."
    spacy_candidate = _spacy_mention(15, 23, "NYHA III", [3])

    # Mock all LLM/spaCy calls: spaCy returns one candidate, Layer 2 returns None (fails).
    with (
        patch(
            "nyha_temporal_identification.dspy_logic.modules.spacy_preprocess",
            return_value=[spacy_candidate],
        ),
        patch.object(extractor, "_classify_with_retry", return_value=None),
    ):
        result = extractor.forward(text)

    # Layer 2 failed → no survivors, and failsafe is off → empty result
    assert result == []


def test_forward_failsafe_on_reinserts_dropped_candidate():
    """With failsafe_dropped_current=True, a dropped spaCy candidate is re-injected."""
    from nyha_temporal_identification.dspy_logic.modules import NYHAExtractorSpacyOnly

    extractor = NYHAExtractorSpacyOnly(failsafe_dropped_current=True)

    text = "Patiënt heeft NYHA III klachten."
    spacy_candidate = _spacy_mention(15, 23, "NYHA III", [3])

    # spaCy returns candidate; Layer 2 classification fails → nothing survives Layer 3
    with (
        patch(
            "nyha_temporal_identification.dspy_logic.modules.spacy_preprocess",
            return_value=[spacy_candidate],
        ),
        patch.object(extractor, "_classify_with_retry", return_value=None),
    ):
        result = extractor.forward(text)

    # Failsafe re-injection must produce exactly the dropped candidate as current/failsafe
    assert len(result) >= 1
    sources = {m.source for m in result}
    assert "failsafe" in sources
    failsafe_m = next(m for m in result if m.source == "failsafe")
    assert failsafe_m.temporality == "current"


# ── reinject_uncovered_regex_spans ────────────────────────────────────────────


def test_regex_reinject_uncovered_span():
    """An uncovered NYHA span is re-injected as current/failsafe with regex_uncovered evidence."""
    text = "NYHA III rest of note"
    result = reinject_uncovered_regex_spans(text, survivors=[], existing_failsafe=[])
    assert len(result) >= 1
    m = result[0]
    assert m.source == "failsafe"
    assert m.temporality == "current"
    assert "regex_uncovered" in m.evidence


def test_regex_reinject_covered_by_survivor():
    """A span already covered by a survivor is not re-injected."""
    text = "NYHA III"
    survivor = _nyha_mention(0, 8, "NYHA III", "III", "current", "llm")
    result = reinject_uncovered_regex_spans(text, survivors=[survivor], existing_failsafe=[])
    assert result == []


def test_regex_reinject_covered_by_pass1_failsafe():
    """A span covered by a pass-1 failsafe mention is not re-injected."""
    text = "NYHA III"
    fs_mention = _nyha_mention(0, 8, "NYHA III", "III", "current", "failsafe")
    result = reinject_uncovered_regex_spans(text, survivors=[], existing_failsafe=[fs_mention])
    assert result == []


def test_regex_reinject_invalid_parse_still_reinjected():
    """An invalid NYHA value (reversed range) is re-injected; get_nyha_class() returns None."""
    text = "NYHA IV-I"
    result = reinject_uncovered_regex_spans(text, survivors=[], existing_failsafe=[])
    assert len(result) >= 1
    m = result[0]
    assert m.source == "failsafe"
    assert m.get_nyha_class() is None


def test_regex_reinject_class_set_on_valid_parse():
    """A valid NYHA span gets the correct class derived from the regex parse."""
    text = "NYHA II rest of note"
    result = reinject_uncovered_regex_spans(text, survivors=[], existing_failsafe=[])
    assert len(result) >= 1
    m = result[0]
    assert m.get_nyha_class() == "II"


# ── forward() with regex pass-2 ───────────────────────────────────────────────


def test_forward_failsafe_on_regex_span_recovered():
    """
    With failsafe_dropped_current=True, a span missed by spaCy (char_span=None)
    but found by find_raw_nyha_spans is re-injected as a failsafe mention.
    """
    from nyha_temporal_identification.dspy_logic.modules import NYHAExtractorSpacyOnly

    extractor = NYHAExtractorSpacyOnly(failsafe_dropped_current=True)
    text = "NYHA III rest of note"

    # spaCy returns nothing (simulating char_span=None drop)
    with (
        patch(
            "nyha_temporal_identification.dspy_logic.modules.spacy_preprocess",
            return_value=[],
        ),
        patch.object(extractor, "_classify_with_retry", return_value=None),
    ):
        result = extractor.forward(text)

    # Pass-2 regex failsafe must still find the span
    assert any(m.source == "failsafe" for m in result)
    failsafe_m = next(m for m in result if m.source == "failsafe")
    assert "NYHA" in text[failsafe_m.char_start : failsafe_m.char_end].upper()


def test_forward_failsafe_off_find_raw_not_called():
    """With failsafe_dropped_current=False, find_raw_nyha_spans is never called."""
    from nyha_temporal_identification.dspy_logic.modules import NYHAExtractorSpacyOnly

    extractor = NYHAExtractorSpacyOnly(failsafe_dropped_current=False)
    text = "NYHA III rest of note"

    with (
        patch(
            "nyha_temporal_identification.dspy_logic.modules.spacy_preprocess",
            return_value=[],
        ),
        patch.object(extractor, "_classify_with_retry", return_value=None),
        patch(
            "nyha_temporal_identification.extraction.parsing.find_raw_nyha_spans"
        ) as mock_raw,
    ):
        result = extractor.forward(text)

    mock_raw.assert_not_called()
    assert result == []


# ── regex_candidate_recall_by_temporality ─────────────────────────────────────


def test_regex_recall_covered_current():
    """Gold current at (0,8) covered by regex → recall=1.0."""
    from nyha_temporal_identification.evaluation.evaluation import (
        regex_candidate_recall_by_temporality,
    )

    gold_m = _nyha_mention(0, 8, "NYHA III", "III", "current")
    doc = NYHANote(
        old_note_id="n1",
        deidentified_text="NYHA III patient",
        mentions=[gold_m],
    )

    result = regex_candidate_recall_by_temporality([doc])
    assert result["current"]["gold"] == 1
    assert result["current"]["covered_by_regex"] == 1
    assert result["current"]["recall"] == 1.0


def test_regex_recall_uncovered_current():
    """Gold current at (100,108) with no regex match → recall=0.0."""
    from nyha_temporal_identification.evaluation.evaluation import (
        regex_candidate_recall_by_temporality,
    )

    gold_m = _nyha_mention(100, 108, "NYHA III", "III", "current")
    doc = NYHANote(
        old_note_id="n1",
        deidentified_text="A" * 200,
        mentions=[gold_m],
    )

    result = regex_candidate_recall_by_temporality([doc])
    assert result["current"]["gold"] == 1
    assert result["current"]["covered_by_regex"] == 0
    assert result["current"]["recall"] == 0.0


def test_regex_recall_above_spacy_recall():
    """
    A gold mention found by regex but not by spaCy contributes to regex recall
    but not spaCy recall.
    """
    from nyha_temporal_identification.evaluation.evaluation import (
        regex_candidate_recall_by_temporality,
        spacy_candidate_recall_by_temporality,
    )

    gold_m = _nyha_mention(0, 8, "NYHA III", "III", "current")
    doc = NYHANote(
        old_note_id="n1",
        deidentified_text="NYHA III patient",
        mentions=[gold_m],
    )

    # spaCy returns nothing; regex finds the span naturally
    with patch(
        "nyha_temporal_identification.dspy_logic.regex.spacy_preprocess",
        return_value=[],
    ):
        spacy_result = spacy_candidate_recall_by_temporality([doc])

    regex_result = regex_candidate_recall_by_temporality([doc])

    assert spacy_result["current"]["recall"] == 0.0
    assert regex_result["current"]["recall"] == 1.0
