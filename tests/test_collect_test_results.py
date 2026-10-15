"""Tests for scripts/collect_test_results.py — report discovery, metric
extraction, and output generation. No model calls; uses synthetic reports."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_test_results.py"
_spec = importlib.util.spec_from_file_location("collect_test_results", _SCRIPT)
ctr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctr)


def _report(
    composite_macro=0.8,
    composite_micro=0.7,
    span=0.9,
    cls=0.6,
    temp_f1=0.75,
    temp_mcc=0.5,
    with_ci=True,
):
    report = {
        "composite": {"macro_f1": composite_macro, "micro": {"f1": composite_micro}},
        "span_identification": {"metrics": {"macro_f1": span}},
        "nyha_classification": {"metrics": {"macro_f1": cls}},
        "temporality_classification": {
            "metrics": {"macro_temp_macro_f1": temp_f1, "macro_mcc": temp_mcc}
        },
    }
    if with_ci:
        report["confidence_intervals"] = {
            "n_bootstrap": 100,
            "metrics": {
                "composite_macro_f1": {"mean": composite_macro, "ci_low": 0.75,
                                       "ci_high": 0.85, "std": 0.02},
                "composite_micro_f1": {"mean": composite_micro, "ci_low": 0.65,
                                       "ci_high": 0.75, "std": 0.02},
                "span_macro_f1": {"mean": span, "ci_low": 0.88, "ci_high": 0.92,
                                  "std": 0.01},
                "class_macro_f1": {"mean": cls, "ci_low": 0.55, "ci_high": 0.65,
                                   "std": 0.02},
                "temp_macro_f1": {"mean": temp_f1, "ci_low": 0.70, "ci_high": 0.80,
                                  "std": 0.02},
                "temp_macro_mcc": {"mean": temp_mcc, "ci_low": 0.45, "ci_high": 0.55,
                                   "std": 0.02},
            },
        }
    return report


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def data_folder(tmp_path: Path) -> Path:
    root = tmp_path / "temporal_status_prediction"
    # One LLM stage report (with CI) and one baseline report (without CI).
    _write(root / "opt_subset" / "evaluation_report_final_base.json", _report())
    _write(
        root / "baselines" / "single" / "base_text" / "threshold_classifier"
        / "evaluation_report_test_threshold_classifier_k_fold_merged.json",
        _report(composite_macro=0.5, with_ci=False),
    )
    return tmp_path


@pytest.fixture
def multi_mode_folder(tmp_path: Path) -> Path:
    """Two feature-mode sub-folders with partial model coverage."""
    root = tmp_path / "temporal_status_prediction"
    _write(root / "opt_subset" / "evaluation_report_final_base.json", _report())
    # base_text mode: threshold + LR
    for model in ("threshold_classifier", "logistic_regression"):
        _write(
            root / "baselines" / "single" / "base_text" / model
            / f"evaluation_report_{model}.json",
            _report(composite_macro=0.6),
        )
    # text mode: LR + GB (no threshold — expected gap)
    for model in ("logistic_regression", "gradient_boosting"):
        _write(
            root / "baselines" / "single" / "text" / model
            / f"evaluation_report_{model}.json",
            _report(composite_macro=0.55),
        )
    return tmp_path


def test_discover_finds_existing_and_skips_missing(data_folder, capsys):
    paths = ctr.discover_report_paths(data_folder)
    assert set(paths) == {"Zero-shot", "Threshold (base+text)"}
    # Missing systems are warned about, not fatal.
    out = capsys.readouterr().out
    assert "no LLM report found for MIPRO" in out
    assert "no baseline report found for Logistic Reg. (base+text)" in out


def test_discover_include_optimised(data_folder):
    root = data_folder / "temporal_status_prediction"
    _write(root / "opt_subset" / "evaluation_report_final_optimised.json", _report())
    paths = ctr.discover_report_paths(data_folder, include_optimised=True)
    assert "Optimised" in paths


def test_discover_multi_mode(multi_mode_folder, capsys):
    paths = ctr.discover_report_paths(multi_mode_folder)
    labels = set(paths)
    # base_text has threshold and LR; text has LR and GB (no threshold — no warn)
    assert "Threshold (base+text)" in labels
    assert "Logistic Reg. (base+text)" in labels
    assert "Logistic Reg. (text)" in labels
    assert "Gradient Boost. (text)" in labels
    # Threshold not present in text mode — should generate no warning for that gap
    out = capsys.readouterr().out
    assert "Threshold (text)" not in out or "no baseline report" not in out


def test_discover_text_mode_threshold_no_warn(tmp_path, capsys):
    """text-mode threshold gap must NOT generate a warn line."""
    root = tmp_path / "temporal_status_prediction"
    _write(
        root / "baselines" / "single" / "text" / "logistic_regression"
        / "evaluation_report_lr.json",
        _report(),
    )
    ctr.discover_report_paths(tmp_path)
    out = capsys.readouterr().out
    assert "Threshold (text)" not in out


def test_baseline_labels_are_colour_grouped(multi_mode_folder):
    """All baseline variants must be treated as ML baselines in the colour map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = ctr.discover_report_paths(multi_mode_folder)
    df = ctr.build_long_table(paths)
    systems = [s for s in ctr.SYSTEM_ORDER if s in df["system"].to_list()]
    systems += [s for s in df["system"].unique().to_list() if s not in ctr.SYSTEM_ORDER]
    colours = [
        "#4C72B0"
        if any(s.startswith(m) for m in ctr._BASELINE_SHORT_NAMES)
        else "#C44E52"
        for s in systems
    ]
    llm_systems = {"Zero-shot"}
    for sys, col in zip(systems, colours):
        if sys in llm_systems:
            assert col == "#C44E52", f"{sys} should be LLM colour"
        elif not sys.startswith("Zero") and not sys.startswith("Few") and sys not in {"MIPRO", "Optimised"}:
            assert col == "#4C72B0", f"{sys} should be baseline colour"


def test_collect_writes_outputs(data_folder):
    out_dir = data_folder / "out"
    df = ctr.collect(data_folder, out_dir)
    assert (out_dir / "test_results_comparison.csv").exists()
    assert (out_dir / "test_results_comparison.tex").exists()
    assert (out_dir / "test_results_comparison.png").exists()
    assert (out_dir / "test_results_comparison.pdf").exists()
    # Long table has one row per (system, metric) present.
    assert set(df["system"].to_list()) == {"Zero-shot", "Threshold (base+text)"}
    assert df.height == 2 * len(ctr.METRICS)


def test_extract_metrics_with_ci():
    metrics = ctr.extract_metrics(_report())
    assert set(metrics) == set(ctr.METRICS)
    comp = metrics["composite_macro_f1"]
    assert comp["point"] == pytest.approx(0.8)
    assert comp["ci_low"] == pytest.approx(0.75)
    assert comp["ci_high"] == pytest.approx(0.85)
    assert comp["has_ci"] == 1.0
    # micro pulled from the nested composite.micro.f1
    assert metrics["composite_micro_f1"]["point"] == pytest.approx(0.7)


def test_extract_metrics_without_ci():
    metrics = ctr.extract_metrics(_report(with_ci=False))
    comp = metrics["composite_macro_f1"]
    assert comp["point"] == pytest.approx(0.8)
    assert math.isnan(comp["ci_low"])
    assert comp["has_ci"] == 0.0


def test_extract_metrics_missing_field_skipped():
    report = _report()
    del report["span_identification"]
    metrics = ctr.extract_metrics(report)
    assert "span_macro_f1" not in metrics
    assert "composite_macro_f1" in metrics


def test_latex_cell_formatting():
    assert ctr._fmt_cell(0.812, 0.78, 0.84) == "0.812 [0.780, 0.840]"
    assert ctr._fmt_cell(0.812, math.nan, math.nan) == "0.812"
    assert ctr._fmt_cell(math.nan, math.nan, math.nan) == "--"
