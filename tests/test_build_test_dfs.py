"""build_test_dfs resilience to unsupported spaCy predictions."""

from __future__ import annotations

import pytest

from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.data_model.general import NyhaMention
from nyha_temporal_identification.models import data_loading


def _spacy_mention(classes, one_hot, start, end, text):
    return NyhaMention(
        original_text=text,
        cleaned_text=text,
        classes=classes,
        one_hot_vector=one_hot,
        start=start,
        end=end,
        original_start=start,
        original_end=end,
    )


def test_from_nyha_mention_rejects_non_adjacent_classes():
    """Non-adjacent classes (e.g. I and III) form an unsupported NYHA range
    and must not silently produce a malformed NYHAMention."""
    bad = _spacy_mention([1, 3], [1, 0, 1, 0], 0, 9, "nyha i-iii")
    with pytest.raises(Exception):
        NYHAMention.from_nyha_mention(other_nyha_mention=bad)


def test_build_test_dfs_skips_invalid_prediction(monkeypatch):
    """A document whose spaCy output includes an unsupported extraction is
    handled gracefully: the invalid prediction is dropped, the valid one is
    kept, and no exception propagates."""
    text = "patient has nyha ii here."
    note = NYHANote(
        old_note_id="doc1",
        deidentified_text=text,
        authored_datetime="2021-01-01",
        mentions=[
            NYHAMention(
                char_start=12, char_end=19, span="nyha ii",
                nyha_class="II", temporality="current",
            )
        ],
    )

    valid = _spacy_mention([2], [0, 1, 0, 0], 12, 19, "nyha ii")
    invalid = _spacy_mention([1, 3], [1, 0, 1, 0], 12, 19, "nyha i-iii")

    monkeypatch.setattr(
        data_loading, "spacy_preprocess", lambda **kwargs: [valid, invalid]
    )

    valid_test_df, report_documents, invalid_preds = data_loading.build_test_dfs(
        [note], ["doc1"]
    )

    # The valid prediction is matched to the gold mention; the invalid one is
    # dropped entirely (not counted as a false positive either).
    assert valid_test_df.height == 1
    assert report_documents == [note]
    assert invalid_preds.get("doc1", []) == []


def test_build_test_dfs_dedupes_double_overlap(monkeypatch):
    """Two spaCy spans overlapping the same gold mention must not both emit a
    valid test row. The first claims the gold mention; the second is routed to
    the invalid predictions as a false positive instead of duplicating the
    gold span/label."""
    text = "patient has nyha ii here."
    note = NYHANote(
        old_note_id="doc1",
        deidentified_text=text,
        authored_datetime="2021-01-01",
        mentions=[
            NYHAMention(
                char_start=12, char_end=19, span="nyha ii",
                nyha_class="II", temporality="current",
            )
        ],
    )

    # Two predictions both overlapping the single gold mention above threshold.
    first = _spacy_mention([2], [0, 1, 0, 0], 12, 19, "nyha ii")
    second = _spacy_mention([2], [0, 1, 0, 0], 12, 18, "nyha i")

    monkeypatch.setattr(
        data_loading, "spacy_preprocess", lambda **kwargs: [first, second]
    )

    valid_test_df, report_documents, invalid_preds = data_loading.build_test_dfs(
        [note], ["doc1"]
    )

    # Only one valid row for the single gold mention; the duplicate overlapping
    # prediction is recorded as an invalid (false-positive) prediction.
    assert valid_test_df.height == 1
    assert report_documents == [note]
    assert len(invalid_preds.get("doc1", [])) == 1
