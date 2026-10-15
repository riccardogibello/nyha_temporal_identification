from typing import List, Optional

from nyha_temporal_identification.constants import (
    ERROR_WINDOW_ANALYZED,
    OVERLAP_THRESHOLD,
)
from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.data_model.nyha_mention_deduplication import (
    unique_mentions,
)
from nyha_temporal_identification.evaluation.constants import (
    IMPROVEMENT_RE,
    TELEGRAPHIC_DENSITY,
)
from nyha_temporal_identification.evaluation.entities import ErrorRecord, ErrorSummary
from nyha_temporal_identification.evaluation.metrics_computation import mention_f1
from nyha_temporal_identification.dspy_logic.regex import CURRENT_RE, HISTORICAL_RE


def _classify_error(
    gold_m: Optional[NYHAMention],
    pred_m: Optional[NYHAMention],
    deidentified_text: str,
    all_pred_mentions: Optional[List[NYHAMention]] = None,
) -> str:
    """
    Classify the type of error for a single (gold, predicted) mention pair.

    Decision tree:
      1. pred exists, gold does not             → false_positive
      2. gold exists, pred does not:
         a. partial overlap found               → span_boundary_mismatch
         b. improvement narrative in context    → improvement_narrative
         c. no temporal anchor:
            - linked date present               → implicit_date_missed
            - no linked date                    → implicit_no_date
         d. telegraphic prose                   → telegraphic_prose
         e. otherwise                           → false_negative
      3. both exist but disagree:
         - class wrong, temporality ok          → wrong_class
         - temporality wrong, class ok          → wrong_temporality
         - both wrong                           → wrong_both
    """
    if gold_m is None:
        return "false_positive"

    if pred_m is None:
        if all_pred_mentions:
            for p in all_pred_mentions:
                if gold_m.overlap(p) >= OVERLAP_THRESHOLD:
                    return "span_boundary_mismatch"

        ctx_start = max(0, gold_m.char_start - ERROR_WINDOW_ANALYZED)
        ctx_end = min(len(deidentified_text), gold_m.char_end + ERROR_WINDOW_ANALYZED)
        context = deidentified_text[ctx_start:ctx_end]

        if IMPROVEMENT_RE.search(context):
            return "improvement_narrative"

        has_anchor = bool(CURRENT_RE.search(context) or HISTORICAL_RE.search(context))
        if not has_anchor:
            if gold_m.linked_date is not None:
                return "implicit_date_missed"
            return "implicit_no_date"

        context_density = len(context.split()) / max(len(context), 1)
        if context_density < TELEGRAPHIC_DENSITY:
            return "telegraphic_prose"

        return "false_negative"

    class_ok = pred_m.get_nyha_class() == gold_m.get_nyha_class()
    temp_ok = pred_m.temporality == gold_m.temporality

    if class_ok and not temp_ok:
        return "wrong_temporality"
    elif not class_ok and temp_ok:
        return "wrong_class"
    elif not class_ok and not temp_ok:
        return "wrong_both"
    else:
        return "correct"


def analyse_errors(
    documents: List[NYHANote],
    predictions: List[List[NYHAMention]],
) -> tuple[List[ErrorRecord], ErrorSummary]:
    """
    Gold-first error taxonomy over a document collection.

    Errors are routed into class and temporality partitions during iteration —
    no post-processing step is needed.

    Returns:
        records: One ErrorRecord per imperfect document (f1 < 1.0).
        summary: ErrorSummary with class and temporality partitions pre-built.
    """
    assert len(documents) == len(predictions), (
        "documents and predictions must be the same length"
    )

    records: List[ErrorRecord] = []
    summary = ErrorSummary()

    for doc, pred_mentions in zip(documents, predictions):
        pred_mentions = unique_mentions(pred_mentions)
        gold_mentions = unique_mentions(doc.mentions)

        metrics = mention_f1(gold_mentions, pred_mentions)
        if metrics["f1"] >= 1.0 - 1e-9:
            continue

        claimed_pred_indices: set[int] = set()
        error_record = ErrorRecord(
            old_note_id=doc.old_note_id,
            gold=[],
            pred=[],
            error_types=[],
        )

        for g in gold_mentions:
            best_pred_idx, best_ov = max(
                ((j, g.overlap(p)) for j, p in enumerate(pred_mentions)),
                key=lambda x: x[1],
                default=(None, 0.0),
            )

            matched = best_pred_idx is not None and best_ov >= OVERLAP_THRESHOLD

            if matched:
                assert best_pred_idx is not None
                etype = _classify_error(
                    g,
                    pred_mentions[best_pred_idx],
                    doc.deidentified_text,
                    all_pred_mentions=pred_mentions,
                )
                claimed_pred_indices.add(best_pred_idx)
                if etype == "correct":
                    continue
                matched_pred_idx: int | None = best_pred_idx
            else:
                etype = _classify_error(
                    g,
                    None,
                    doc.deidentified_text,
                    all_pred_mentions=pred_mentions,
                )
                matched_pred_idx = None

            error_record.gold.append(g)
            error_record.pred.append(
                pred_mentions[matched_pred_idx]
                if matched_pred_idx is not None
                else None
            )
            error_record.error_types.append(etype)
            summary.record(etype, g)

        for j, p in enumerate(pred_mentions):
            if j not in claimed_pred_indices:
                error_record.gold.append(None)
                error_record.pred.append(p)
                error_record.error_types.append("false_positive")
                # false_positive has no reliable gold mention to key on;
                # it is counted in class_error_counts only (no label breakdown).
                summary.class_error_counts["false_positive"] += 1
                summary.total_errors += 1

        records.append(error_record)

    return records, summary
