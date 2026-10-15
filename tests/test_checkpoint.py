"""Tests for PipelineCheckpoint.assert_config_matches tiered behaviour."""

import pytest

from nyha_temporal_identification.dspy_logic.checkpoint import PipelineCheckpoint

BASE_SNAPSHOT = {
    "task_lm": "google/gemma-3-12b-it",
    "optimiser": "both",
    "max_tokens": 1536,
    "identify_max_tokens": 2048,
    "seed": 42,
}


def _make_checkpoint(tmp_path, snapshot=None):
    cp = PipelineCheckpoint(tmp_path)
    if snapshot is not None:
        cp.save(step=1, config_snapshot=snapshot)
    return cp


def test_critical_key_mismatch_raises(tmp_path):
    cp = _make_checkpoint(tmp_path, BASE_SNAPSHOT)
    current = {**BASE_SNAPSHOT, "task_lm": "google/gemma-3-27b-it"}
    with pytest.raises(ValueError) as exc_info:
        cp.assert_config_matches(current)
    msg = str(exc_info.value)
    assert "google/gemma-3-12b-it" in msg
    assert "google/gemma-3-27b-it" in msg


def test_non_critical_mismatch_warns_and_proceeds(tmp_path, capsys):
    cp = _make_checkpoint(tmp_path, BASE_SNAPSHOT)
    current = {**BASE_SNAPSHOT, "max_tokens": 512}
    cp.assert_config_matches(current)  # must not raise
    captured = capsys.readouterr()
    assert "[CHECKPOINT]" in captured.out
    assert "max_tokens" in captured.out


def test_exact_match_silent(tmp_path, capsys):
    cp = _make_checkpoint(tmp_path, BASE_SNAPSHOT)
    capsys.readouterr()  # drain save() output
    cp.assert_config_matches(dict(BASE_SNAPSHOT))  # identical copy
    captured = capsys.readouterr()
    assert captured.out == ""


def test_missing_snapshot_raises(tmp_path):
    cp = _make_checkpoint(tmp_path, snapshot=None)  # no checkpoint file written
    with pytest.raises(ValueError, match="does not contain a config snapshot"):
        cp.assert_config_matches(BASE_SNAPSHOT)
