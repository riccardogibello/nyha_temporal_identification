"""Evaluation utilities for NYHA mention extraction.

This module provides:
- mention-level scoring with span overlap matching (IoU threshold),
- composite and sub-task metrics (span F1, class macro-F1, temporality MCC),
- DSPy optimisation metric weighting,
- error taxonomy and record export,
- task-specific breakdown reports for span, class, temporality, and composite views.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List

from nyha_temporal_identification.constants import OVERLAP_THRESHOLD
from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.data_model.general import NyhaMention
from nyha_temporal_identification.data_model.nyha_mention_deduplication import (
    unique_mentions,
)
from nyha_temporal_identification.evaluation.bootstrap import bootstrap_metric_cis
from nyha_temporal_identification.evaluation.entities import ErrorRecord
from nyha_temporal_identification.evaluation.error_analysis import analyse_errors
from nyha_temporal_identification.evaluation.metrics_computation import (
    best_gold_match,
    mention_f1,
)


def store_records(
    records: List[ErrorRecord],
    output_dir_path: Path,
) -> None:
    """Store error records as JSON for later manual review."""
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_file = output_dir_path / "error_records.json"
    records_list = [
        {
            "old_note_id": record.old_note_id,
            "gold": [m.model_dump() if m is not None else None for m in record.gold],
            "pred": [m.model_dump() if m is not None else None for m in record.pred],
            "error_types": record.error_types,
        }
        for record in records
    ]
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records_list, f, indent=2, ensure_ascii=False)


# ── Per-cell reporting ────────────────────────────────────────────────────────


def _cells_to_report(cells: dict) -> dict:
    """Derive precision/recall/F1 from raw TP/FP/FN counts per cell."""
    report = {}
    for key, counts in sorted(cells.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        report[str(key)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return report


def per_class_report(
    documents: List[NYHANote],
    predictions: List[List[NYHAMention]],
) -> dict:
    """
    Joint match (span + class + temporality), keyed by (class, temporality).
    Used for the composite section.

    On a mismatch, FP is charged to the pred's cell and FN to the gold's cell
    in the same operation. The gold mention is always consumed on a span match
    regardless of label correctness, preventing double-matching.
    """
    cells: dict[tuple, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for doc, pred_mentions in zip(documents, predictions):
        gold_mentions = doc.mentions
        matched_gold: set[int] = set()

        for p in pred_mentions:
            best_idx, best_ov = best_gold_match(p, gold_mentions, matched_gold)
            if best_idx is not None and best_ov >= OVERLAP_THRESHOLD:
                g = gold_mentions[best_idx]
                matched_gold.add(best_idx)  # always consume on span match
                if (
                    p.get_nyha_class() == g.get_nyha_class()
                    and p.temporality == g.temporality
                ):
                    cells[(g.get_nyha_class(), g.temporality)]["tp"] += 1
                else:
                    # FP charged to pred cell, FN charged to gold cell
                    cells[(p.get_nyha_class(), p.temporality)]["fp"] += 1
                    cells[(g.get_nyha_class(), g.temporality)]["fn"] += 1
            else:
                # No span match: pure FP on pred side
                cells[(p.get_nyha_class(), p.temporality)]["fp"] += 1

        for i, g in enumerate(gold_mentions):
            if i not in matched_gold:
                # Span missed entirely: pure FN on gold side
                cells[(g.get_nyha_class(), g.temporality)]["fn"] += 1

    return _cells_to_report(
        {f"NYHA_{cls}_{temp}": v for (cls, temp), v in cells.items()}
    )


def per_class_report_by_class_only(
    documents: List[NYHANote],
    predictions: List[List[NYHAMention]],
) -> dict:
    """
    Span + class match, temporality ignored in both matching and cell key.
    Used for the nyha_classification section.
    Rows keyed by NYHA class only (e.g. "I", "II", "III").

    On a class mismatch, FP is charged to the pred's class and FN to the
    gold's class. The gold mention is always consumed on a span match,
    guaranteeing FP[pred_class] and FN[gold_class] are symmetric across
    all mismatch pairs.
    """
    cells: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for doc, pred_mentions in zip(documents, predictions):
        gold_mentions = doc.mentions
        matched_gold: set[int] = set()

        for p in pred_mentions:
            best_idx, best_ov = best_gold_match(p, gold_mentions, matched_gold)
            if best_idx is not None and best_ov >= OVERLAP_THRESHOLD:
                g = gold_mentions[best_idx]
                matched_gold.add(best_idx)  # always consume on span match
                if p.get_nyha_class() == g.get_nyha_class():
                    cells[g.get_nyha_class() or "unknown"]["tp"] += 1
                else:
                    # FP charged to pred class, FN charged to gold class
                    cells[p.get_nyha_class() or "unknown"]["fp"] += 1
                    cells[g.get_nyha_class() or "unknown"]["fn"] += 1
            else:
                # No span match: pure FP on pred side
                cells[p.get_nyha_class() or "unknown"]["fp"] += 1

        for i, g in enumerate(gold_mentions):
            if i not in matched_gold:
                # Span missed entirely: pure FN on gold side
                cells[g.get_nyha_class() or "unknown"]["fn"] += 1

    return _cells_to_report(cells)


def per_class_report_by_temporality_only(
    documents: List[NYHANote],
    predictions: List[List[NYHAMention]],
) -> dict:
    """
    Span + temporality match, class ignored in both matching and cell key.
    Used for the temporality_classification section.
    Rows keyed by temporality only: "current" / "historical".

    On a temporality mismatch, FP is charged to the pred's temporality and
    FN to the gold's temporality. The gold mention is always consumed on a
    span match, guaranteeing:
        FP[current] == FN[historical]
        FP[historical] == FN[current]
    by construction.
    """
    cells: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for doc, pred_mentions in zip(documents, predictions):
        gold_mentions = doc.mentions
        matched_gold: set[int] = set()

        for p in pred_mentions:
            best_idx, best_ov = best_gold_match(p, gold_mentions, matched_gold)
            if best_idx is not None and best_ov >= OVERLAP_THRESHOLD:
                g = gold_mentions[best_idx]
                matched_gold.add(best_idx)  # always consume on span match
                if p.temporality == g.temporality:
                    cells[g.temporality or "unknown"]["tp"] += 1
                else:
                    # FP charged to pred temporality, FN to gold temporality
                    cells[p.temporality or "unknown"]["fp"] += 1
                    cells[g.temporality or "unknown"]["fn"] += 1
            else:
                # No span match: pure FP on pred side
                cells[p.temporality or "unknown"]["fp"] += 1

        for i, g in enumerate(gold_mentions):
            if i not in matched_gold:
                # Span missed entirely: pure FN on gold side
                cells[g.temporality or "unknown"]["fn"] += 1

    return _cells_to_report(cells)


def _temporality_error_breakdown(
    documents: List[NYHANote],
    predictions: List[List[NYHAMention]],
    error_records: List[ErrorRecord],
) -> dict:
    """
    Build the temporality error breakdown from the confusion matrix,
    not from the gold-first error taxonomy.

    Counts are derived by scanning matched span pairs directly so that
    per-label counts are consistent with the breakdown FP/FN values.

    Special error types (implicit_date_missed, implicit_no_date) are
    preserved from the gold-first error taxonomy because they require
    document context not available in the confusion matrix. They are
    attributed to the gold label and represent a sub-classification of
    false negatives, not of wrong_temporality.

    In per_label, the gold temporality is used as the key, consistent
    with FN attribution in the breakdown.
    """
    # Collect special sub-types from existing error records
    special_subtypes = {"implicit_date_missed", "implicit_no_date"}
    special_counts: dict[str, int] = defaultdict(int)
    special_per_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for record in error_records:
        for gold_m, etype in zip(record.gold, record.error_types):
            if etype in special_subtypes and gold_m is not None:
                special_counts[etype] += 1
                special_per_label[gold_m.temporality or "unknown"][etype] += 1

    # Derive wrong_temporality counts from confusion matrix
    wrong_temporality_total = 0
    wrong_temporality_per_label: dict[str, int] = defaultdict(int)

    for doc, pred_mentions in zip(documents, predictions):
        gold_mentions = doc.mentions
        matched_gold: set[int] = set()

        for p in pred_mentions:
            best_idx, best_ov = best_gold_match(p, gold_mentions, matched_gold)
            if best_idx is not None and best_ov >= OVERLAP_THRESHOLD:
                g = gold_mentions[best_idx]
                matched_gold.add(best_idx)
                if p.temporality != g.temporality:
                    wrong_temporality_total += 1
                    wrong_temporality_per_label[g.temporality or "unknown"] += 1

    total_special = sum(special_counts.values())
    total_errors = wrong_temporality_total + total_special

    if total_errors == 0:
        return {
            "total": 0,
            "counts": {},
            "percentages": {},
            "descriptions": _TEMP_ERROR_DESCRIPTIONS,
            "per_label": {},
        }

    counts: dict[str, int] = {}
    if wrong_temporality_total > 0:
        counts["wrong_temporality"] = wrong_temporality_total
    counts.update({k: v for k, v in special_counts.items() if v > 0})

    percentages = {k: round(100 * v / total_errors, 2) for k, v in counts.items()}

    per_label: dict[str, dict[str, int]] = defaultdict(dict)
    for label, count in wrong_temporality_per_label.items():
        if count > 0:
            per_label[label]["wrong_temporality"] = count
    for label, subtypes in special_per_label.items():
        for etype, count in subtypes.items():
            if count > 0:
                per_label[label][etype] = count

    return {
        "total": total_errors,
        "counts": counts,
        "percentages": percentages,
        "descriptions": {
            k: v for k, v in _TEMP_ERROR_DESCRIPTIONS.items() if k in counts
        },
        "per_label": dict(per_label),
    }


_TEMP_ERROR_DESCRIPTIONS = {
    "wrong_temporality": "Span + class matched but temporality wrong",
    "implicit_date_missed": (
        "No anchor words but a linked date was present — "
        "date-linkage inference did not resolve temporality"
    ),
    "implicit_no_date": "No temporal anchor AND no linked date available",
}


# ── Aggregation helpers ───────────────────────────────────────────────────────


def _aggregate_mention_f1(
    documents: List[NYHANote],
    predictions: List[List[NYHAMention]],
) -> dict:
    """
    Aggregate mention_f1 sub-scores across documents.
    Returns micro-averaged counts and macro-averaged rates for all sub-tasks.
    """
    total_tp, total_fp, total_fn = 0.0, 0, 0
    doc_f1s: List[float] = []
    doc_span_f1s: List[float] = []
    doc_class_macro_f1s: List[float] = []
    doc_temp_mccs: List[float] = []
    doc_temp_macro_f1s: List[float] = []

    for doc, pred_mentions in zip(documents, predictions):
        result = mention_f1(doc.mentions, pred_mentions)
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        doc_f1s.append(result["f1"])
        doc_span_f1s.append(result["span_f1"])
        doc_class_macro_f1s.append(result["class_macro_f1"])
        doc_temp_mccs.append(result["temp_mcc"])
        doc_temp_macro_f1s.append(result["temp_macro_f1"])

    n = len(documents)

    def _micro_f1(tp, fp, fn) -> tuple[float, float, float]:
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f, 4)

    def _macro(values: List[float]) -> float:
        return round(sum(values) / n, 4) if n else 0.0

    comp_p, comp_r, comp_f1 = _micro_f1(total_tp, total_fp, total_fn)
    macro_mcc = _macro(doc_temp_mccs)

    return {
        "composite": {
            "micro": {
                "precision": comp_p,
                "recall": comp_r,
                "f1": comp_f1,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
            },
            "macro_f1": _macro(doc_f1s),
        },
        "span_identification": {
            "macro_f1": _macro(doc_span_f1s),
        },
        "nyha_classification": {
            "macro_f1": _macro(doc_class_macro_f1s),
        },
        "temporality_classification": {
            "macro_mcc": macro_mcc,
            "macro_mcc_normalized": round((macro_mcc + 1) / 2, 4),
            "macro_temp_macro_f1": _macro(doc_temp_macro_f1s),
        },
    }


def print_evaluation_report(
    stage: str,
    documents: List[NYHANote],
    predictions: List[List[NYHAMention]],
    output_dir_path: Path,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> None:
    """
    Store a structured evaluation report for a given pipeline stage.

    Report structure:
      - span_identification:        NER macro F1 only (no breakdown — label-agnostic)
      - nyha_classification:        Class macro F1 + breakdown by NYHA class
                                    + class error breakdown
      - temporality_classification: Macro F1 (optimisation target) + MCC (diagnostic)
                                    + breakdown by current/historical
                                    + temporality error breakdown (confusion-matrix-derived)
      - composite:                  Joint F1 + full (class × temporality) breakdown

    All three breakdown functions (per_class_report, per_class_report_by_class_only,
    per_class_report_by_temporality_only) now guarantee FP/FN symmetry across
    mismatch pairs by always consuming the gold mention on a span match and
    recording both FP on the pred cell and FN on the gold cell simultaneously.

    Args:
        stage:           Stage label used in report metadata and output filename.
        documents:       NYHANote objects carrying gold mentions.
        predictions:     Predicted mention lists, one per document, same order.
        output_dir_path: Directory where report and error records are stored.
        n_bootstrap:     Document-level bootstrap resamples for confidence
                         intervals on the headline metrics; 0 disables the
                         confidence_intervals block (default).
        bootstrap_seed:  Seed for the bootstrap RNG (reproducible CIs).
    """
    assert len(documents) == len(predictions), (
        "documents and predictions must be the same length"
    )

    doc_count = len(documents)

    doc_f1s = [
        mention_f1(doc.mentions, pred)["f1"]
        for doc, pred in zip(documents, predictions)
    ]
    exact_match_count = sum(1 for f in doc_f1s if f >= 1.0 - 1e-9)
    total_gold = sum(len(doc.mentions) for doc in documents)
    total_pred = sum(len(p) for p in predictions)
    total_gold_unique = sum(len(unique_mentions(doc.mentions)) for doc in documents)
    total_pred_unique = sum(len(unique_mentions(p)) for p in predictions)

    aggregated = _aggregate_mention_f1(documents, predictions)

    class_breakdown = per_class_report_by_class_only(documents, predictions)
    temp_breakdown = per_class_report_by_temporality_only(documents, predictions)
    joint_breakdown = per_class_report(documents, predictions)

    # Gold-first error taxonomy: used for class errors and special temp subtypes
    records, error_summary = analyse_errors(documents, predictions)
    store_records(records, output_dir_path)

    # Temporality error breakdown: confusion-matrix-derived for FP/FN consistency
    temp_error_breakdown = _temporality_error_breakdown(documents, predictions, records)

    report = {
        "stage": stage,
        "documents": {
            "count": doc_count,
            "exact_match_count": exact_match_count,
            "exact_match_rate": round(exact_match_count / doc_count, 4)
            if doc_count
            else 0.0,
        },
        "mentions": {
            "gold_total": total_gold,
            "predicted_total": total_pred,
            "gold_unique": total_gold_unique,
            "predicted_unique": total_pred_unique,
            "avg_gold_per_doc": round(total_gold / doc_count, 4) if doc_count else 0.0,
            "avg_predicted_per_doc": round(total_pred / doc_count, 4)
            if doc_count
            else 0.0,
        },
        "span_identification": {
            "description": "Did the model find the right character spans? (label-agnostic)",
            "metrics": {
                "macro_f1": aggregated["span_identification"]["macro_f1"],
            },
        },
        "nyha_classification": {
            "description": (
                "NYHA class accuracy conditioned on matched spans. "
                "Macro F1 over classes to penalise ignoring rare labels (e.g., class IV)."
            ),
            "metrics": {
                "macro_f1": aggregated["nyha_classification"]["macro_f1"],
            },
            "breakdown": class_breakdown,
            "errors": error_summary.as_class_partition(),
        },
        "temporality_classification": {
            "description": (
                "Temporality accuracy conditioned on matched spans. "
                "Macro F1 gives equal weight to current and historical regardless "
                "of frequency and is the optimisation target. "
                "MCC is retained as a diagnostic to capture binary correlation structure."
            ),
            "metrics": {
                "macro_temp_macro_f1": aggregated["temporality_classification"][
                    "macro_temp_macro_f1"
                ],
                "macro_mcc": aggregated["temporality_classification"]["macro_mcc"],
                "macro_mcc_normalized": aggregated["temporality_classification"][
                    "macro_mcc_normalized"
                ],
            },
            "breakdown": temp_breakdown,
            "errors": temp_error_breakdown,
        },
        "composite": {
            "description": "0.10×span_f1 + 0.10×class_macro_f1 + 0.80×temp_macro_f1.",
            "micro": aggregated["composite"]["micro"],
            "macro_f1": aggregated["composite"]["macro_f1"],
            "breakdown": joint_breakdown,
        },
    }

    if n_bootstrap > 0:
        report["confidence_intervals"] = bootstrap_metric_cis(
            documents,
            predictions,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )

    output_dir_path.mkdir(parents=True, exist_ok=True)
    stage_slug = re.sub(r"[^a-z0-9]+", "_", stage.lower()).strip("_") or "stage"
    report_path = output_dir_path / f"evaluation_report_{stage_slug}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def regex_candidate_recall_by_temporality(documents: List[NYHANote]) -> dict:
    """
    Residual-leak diagnostic: quantify gold mentions the regex pass never finds.

    Runs find_raw_nyha_spans on each note's raw text and matches intervals to
    gold mentions by span overlap (OVERLAP_THRESHOLD).  Returns per gold
    temporality:

        {"current":    {"gold": int, "covered_by_regex": int, "recall": float},
         "historical": {"gold": int, "covered_by_regex": int, "recall": float}}

    current.recall is the ceiling on leakage protection after both failsafe
    passes: (1 - recall) of current gold mentions can still leak.
    """
    from nyha_temporal_identification.extraction.parsing import find_raw_nyha_spans

    counts: dict[str, dict[str, int]] = {
        "current": {"gold": 0, "covered_by_regex": 0},
        "historical": {"gold": 0, "covered_by_regex": 0},
    }

    for doc in documents:
        raw_spans = find_raw_nyha_spans(doc.deidentified_text)
        for gold in doc.mentions:
            temporality = gold.temporality or "current"
            if temporality not in counts:
                counts[temporality] = {"gold": 0, "covered_by_regex": 0}
            counts[temporality]["gold"] += 1
            covered = any(
                gold.overlap(
                    NYHAMention(
                        char_start=s,
                        char_end=e,
                        span=doc.deidentified_text[s:e],
                    )
                )
                >= OVERLAP_THRESHOLD
                for (s, e) in raw_spans
            )
            if covered:
                counts[temporality]["covered_by_regex"] += 1

    result: dict = {}
    for temp, c in counts.items():
        gold_n = c["gold"]
        covered_n = c["covered_by_regex"]
        result[temp] = {
            "gold": gold_n,
            "covered_by_regex": covered_n,
            "recall": round(covered_n / gold_n, 4) if gold_n > 0 else 1.0,
        }
    return result


def spacy_candidate_recall_by_temporality(documents: List[NYHANote]) -> dict:
    """
    Residual-leak diagnostic: quantify gold mentions spaCy never proposed.

    Runs spacy_preprocess on each note and matches candidates to gold mentions
    by span overlap (OVERLAP_THRESHOLD). Returns per gold temporality:

        {"current":    {"gold": int, "covered_by_spacy": int, "recall": float},
         "historical": {"gold": int, "covered_by_spacy": int, "recall": float}}

    current.recall is the ceiling on failsafe's leakage protection:
    (1 - recall) of current gold mentions can still leak even with failsafe.
    """
    from nyha_temporal_identification.dspy_logic.regex import spacy_preprocess

    counts: dict[str, dict[str, int]] = {
        "current": {"gold": 0, "covered_by_spacy": 0},
        "historical": {"gold": 0, "covered_by_spacy": 0},
    }

    for doc in documents:
        candidates: list[NyhaMention] = spacy_preprocess(
            doc.deidentified_text, doc.authored_datetime
        )
        for gold in doc.mentions:
            temporality = gold.temporality or "current"
            if temporality not in counts:
                counts[temporality] = {"gold": 0, "covered_by_spacy": 0}
            counts[temporality]["gold"] += 1
            covered = any(gold.overlap(c) >= OVERLAP_THRESHOLD for c in candidates)
            if covered:
                counts[temporality]["covered_by_spacy"] += 1

    result: dict = {}
    for temp, c in counts.items():
        gold_n = c["gold"]
        covered_n = c["covered_by_spacy"]
        result[temp] = {
            "gold": gold_n,
            "covered_by_spacy": covered_n,
            "recall": round(covered_n / gold_n, 4) if gold_n > 0 else 1.0,
        }
    return result
