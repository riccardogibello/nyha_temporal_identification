"""Document-level bootstrap confidence intervals for evaluation metrics.

The evaluation reports macro-average per-document ``mention_f1`` scores and
micro-aggregate the composite tp/fp/fn counts. This module resamples the
**documents** (not individual mentions) with replacement to estimate the
sampling distribution of the headline metrics, so a test report can carry a
mean ± 95% CI instead of a bare point estimate.

Resampling at the document level (rather than the mention level) respects the
correlation between mentions in the same note and matches the unit over which
the report macro-averages.

The bootstrap reuses the per-document ``mention_f1`` results, computed once, so
it adds no model calls and is effectively free.
"""

from __future__ import annotations

from typing import List

import numpy as np

from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.evaluation.metrics_computation import mention_f1

# Headline metrics carried in the confidence_intervals block. Keys are the
# names exposed in the report; values name the per-document fields used to
# recompute them on each resample.
_MACRO_METRICS = {
    "composite_macro_f1": "f1",
    "span_macro_f1": "span_f1",
    "class_macro_f1": "class_macro_f1",
    "temp_macro_f1": "temp_macro_f1",
    "temp_macro_mcc": "temp_mcc",
}


def _micro_f1(tp: float, fp: float, fn: float) -> float:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def bootstrap_metric_cis(
    documents: List[NYHANote],
    predictions: List[List[NYHAMention]],
    n_bootstrap: int,
    seed: int,
    ci: float = 0.95,
) -> dict:
    """Compute document-level bootstrap CIs for the headline metrics.

    Args:
        documents:   NYHANote objects carrying gold mentions.
        predictions: Predicted mention lists, one per document, same order.
        n_bootstrap: Number of bootstrap resamples (>0).
        seed:        Seed for the resampling RNG (reproducible CIs).
        ci:          Central interval width (default 0.95 → 2.5/97.5 pct).

    Returns:
        ``{method, n_bootstrap, seed, ci, metrics: {name: {mean, ci_low,
        ci_high, std}}}``. Returns an empty ``metrics`` dict when there are no
        documents.
    """
    assert len(documents) == len(predictions), (
        "documents and predictions must be the same length"
    )

    n = len(documents)
    metric_names = list(_MACRO_METRICS) + ["composite_micro_f1"]

    if n == 0 or n_bootstrap <= 0:
        return {
            "method": "document-level bootstrap",
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
            "ci": ci,
            "metrics": {},
        }

    # Cache per-document scores once.
    per_doc = [mention_f1(doc.mentions, pred) for doc, pred in zip(documents, predictions)]
    macro_arrays = {
        name: np.array([row[field] for row in per_doc], dtype=float)
        for name, field in _MACRO_METRICS.items()
    }
    tp_arr = np.array([row["tp"] for row in per_doc], dtype=float)
    fp_arr = np.array([row["fp"] for row in per_doc], dtype=float)
    fn_arr = np.array([row["fn"] for row in per_doc], dtype=float)

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        for name, arr in macro_arrays.items():
            samples[name].append(float(arr[idx].mean()))
        samples["composite_micro_f1"].append(
            _micro_f1(float(tp_arr[idx].sum()), float(fp_arr[idx].sum()), float(fn_arr[idx].sum()))
        )

    lo_pct = 100 * (1 - ci) / 2
    hi_pct = 100 * (1 + ci) / 2

    metrics: dict[str, dict[str, float]] = {}
    for name in metric_names:
        draws = np.array(samples[name], dtype=float)
        metrics[name] = {
            "mean": round(float(draws.mean()), 4),
            "ci_low": round(float(np.percentile(draws, lo_pct)), 4),
            "ci_high": round(float(np.percentile(draws, hi_pct)), 4),
            "std": round(float(draws.std()), 4),
        }

    return {
        "method": "document-level bootstrap",
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "ci": ci,
        "metrics": metrics,
    }
