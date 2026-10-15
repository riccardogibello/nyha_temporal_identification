"""Tests for document-level bootstrap confidence intervals."""

from __future__ import annotations

import json

from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.evaluation.bootstrap import bootstrap_metric_cis
from nyha_temporal_identification.evaluation.evaluation import print_evaluation_report

_EXPECTED_METRICS = {
    "composite_macro_f1",
    "composite_micro_f1",
    "span_macro_f1",
    "class_macro_f1",
    "temp_macro_f1",
    "temp_macro_mcc",
}


def _note(note_id: str, temporality: str) -> NYHANote:
    text = "patient has nyha ii noted in 2020 here."
    return NYHANote(
        old_note_id=note_id,
        deidentified_text=text,
        authored_datetime="2021-01-01",
        mentions=[
            NYHAMention(
                char_start=12, char_end=19, span="nyha ii",
                nyha_class="II", temporality=temporality,
            )
        ],
    )


def _pred(temporality: str) -> list[NYHAMention]:
    return [
        NYHAMention(
            char_start=12, char_end=19, span="nyha ii",
            nyha_class="II", temporality=temporality,
        )
    ]


def _mixed_corpus():
    """Five notes; predictions correct on the first three, wrong temporality
    on the last two — gives a non-degenerate metric distribution."""
    docs = [
        _note("a", "current"),
        _note("b", "current"),
        _note("c", "historical"),
        _note("d", "historical"),
        _note("e", "current"),
    ]
    preds = [
        _pred("current"),
        _pred("current"),
        _pred("historical"),
        _pred("current"),   # wrong temporality
        _pred("historical"),  # wrong temporality
    ]
    return docs, preds


def test_bootstrap_is_deterministic_for_same_seed():
    docs, preds = _mixed_corpus()
    a = bootstrap_metric_cis(docs, preds, n_bootstrap=200, seed=42)
    b = bootstrap_metric_cis(docs, preds, n_bootstrap=200, seed=42)
    assert a == b


def test_bootstrap_seed_changes_result():
    docs, preds = _mixed_corpus()
    a = bootstrap_metric_cis(docs, preds, n_bootstrap=200, seed=1)
    b = bootstrap_metric_cis(docs, preds, n_bootstrap=200, seed=2)
    # Different seeds should not produce identical interval bounds everywhere.
    assert a["metrics"] != b["metrics"]


def test_bootstrap_ci_brackets_mean():
    docs, preds = _mixed_corpus()
    result = bootstrap_metric_cis(docs, preds, n_bootstrap=500, seed=7)
    assert set(result["metrics"]) == _EXPECTED_METRICS
    for name, stats in result["metrics"].items():
        assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"], name
        assert 0.0 <= stats["ci_low"] <= 1.0, name
        assert 0.0 <= stats["ci_high"] <= 1.0, name


def test_bootstrap_perfect_predictions_collapse_to_one():
    docs = [_note("a", "current"), _note("b", "historical")]
    preds = [_pred("current"), _pred("historical")]
    result = bootstrap_metric_cis(docs, preds, n_bootstrap=100, seed=0)
    for name in ("composite_macro_f1", "composite_micro_f1", "temp_macro_f1"):
        stats = result["metrics"][name]
        assert stats["mean"] == 1.0, name
        assert stats["ci_low"] == 1.0 and stats["ci_high"] == 1.0, name
        assert stats["std"] == 0.0, name


def test_bootstrap_zero_resamples_returns_empty_metrics():
    docs, preds = _mixed_corpus()
    result = bootstrap_metric_cis(docs, preds, n_bootstrap=0, seed=42)
    assert result["metrics"] == {}
    assert result["n_bootstrap"] == 0


def test_bootstrap_empty_corpus_returns_empty_metrics():
    result = bootstrap_metric_cis([], [], n_bootstrap=100, seed=42)
    assert result["metrics"] == {}


def _load_report(tmp_path, stage):
    slug = stage.lower().replace(" ", "_")
    with open(tmp_path / f"evaluation_report_{slug}.json") as f:
        return json.load(f)


def test_report_has_no_ci_block_by_default(tmp_path):
    docs, preds = _mixed_corpus()
    print_evaluation_report("stage_default", docs, preds, tmp_path)
    report = _load_report(tmp_path, "stage_default")
    assert "confidence_intervals" not in report


def test_report_includes_ci_block_when_requested(tmp_path):
    docs, preds = _mixed_corpus()
    print_evaluation_report(
        "stage_ci", docs, preds, tmp_path, n_bootstrap=200, bootstrap_seed=42
    )
    report = _load_report(tmp_path, "stage_ci")
    ci = report["confidence_intervals"]
    assert ci["method"] == "document-level bootstrap"
    assert ci["n_bootstrap"] == 200
    assert ci["seed"] == 42
    assert set(ci["metrics"]) == _EXPECTED_METRICS
