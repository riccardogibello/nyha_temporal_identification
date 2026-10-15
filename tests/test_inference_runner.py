"""Tests for the inference cache behaviour in dspy_logic.runners."""

import json

from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.dspy_logic.runners import _process_single_document


def _make_mention() -> NYHAMention:
    return NYHAMention(
        char_start=0,
        char_end=8,
        span="NYHA III",
        nyha_class="III",
        temporality="current",
    )


def _make_doc(note_id: str = "note-1") -> NYHANote:
    return NYHANote(
        old_note_id=note_id,
        deidentified_text="Patient met NYHA III klachten.",
        mentions=[],
        authored_datetime="2024-01-01",
    )


class _CountingExtractor:
    """Stand-in extractor that records how many times it is invoked."""

    failsafe_dropped_current = False

    def __init__(self, predictions):
        self._predictions = predictions
        self.calls = 0

    def __call__(self, deidentified_text, authored_datetime=None):
        self.calls += 1
        return list(self._predictions)


def test_non_empty_result_is_written_and_returned(tmp_path):
    doc = _make_doc()
    extractor = _CountingExtractor([_make_mention()])

    doc_id, preds, was_cached = _process_single_document(doc, extractor, tmp_path)

    assert doc_id == "note-1"
    assert len(preds) == 1
    assert was_cached is False
    assert extractor.calls == 1
    # The result file is written for a non-empty extraction.
    assert (tmp_path / "note-1.json").exists()


def test_non_empty_cache_hit_skips_extractor(tmp_path):
    doc = _make_doc()
    first = _CountingExtractor([_make_mention()])
    _process_single_document(doc, first, tmp_path)

    # Second run with a fresh extractor must hit the cache, not call the LLM.
    second = _CountingExtractor([_make_mention()])
    doc_id, preds, was_cached = _process_single_document(doc, second, tmp_path)

    assert was_cached is True
    assert len(preds) == 1
    assert second.calls == 0


def test_empty_extraction_is_not_written(tmp_path):
    doc = _make_doc()
    extractor = _CountingExtractor([])

    doc_id, preds, was_cached = _process_single_document(doc, extractor, tmp_path)

    assert preds == []
    assert was_cached is False
    assert extractor.calls == 1
    # No file is persisted for an empty result, so it stays uncached.
    assert not (tmp_path / "note-1.json").exists()


def test_failed_extraction_is_not_written(tmp_path):
    doc = _make_doc()

    class _FailingExtractor:
        failsafe_dropped_current = False

        def __init__(self):
            self.calls = 0

        def __call__(self, deidentified_text, authored_datetime=None):
            self.calls += 1
            raise RuntimeError("boom")

    extractor = _FailingExtractor()
    doc_id, preds, was_cached = _process_single_document(doc, extractor, tmp_path)

    assert preds == []
    assert was_cached is False
    assert extractor.calls == 1
    assert not (tmp_path / "note-1.json").exists()


def test_empty_cached_file_is_recomputed(tmp_path):
    doc = _make_doc()
    # Simulate a stale file left by an earlier run with empty predictions.
    stale = {
        "old_note_id": "note-1",
        "deidentified_text": doc.deidentified_text,
        "authored_datetime": doc.authored_datetime,
        "predictions": [],
    }
    out_path = tmp_path / "note-1.json"
    with open(out_path, "w") as f:
        json.dump(stale, f)

    extractor = _CountingExtractor([_make_mention()])
    doc_id, preds, was_cached = _process_single_document(doc, extractor, tmp_path)

    # The empty cached file is treated as a miss: extractor runs and overwrites it.
    assert was_cached is False
    assert extractor.calls == 1
    assert len(preds) == 1
    with open(out_path) as f:
        rewritten = json.load(f)
    assert len(rewritten["predictions"]) == 1
