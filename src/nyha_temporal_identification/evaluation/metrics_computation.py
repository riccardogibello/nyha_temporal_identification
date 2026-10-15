import logging
from typing import List, Literal, Optional

import dspy
from sklearn.metrics import f1_score, matthews_corrcoef

from nyha_temporal_identification.constants import OVERLAP_THRESHOLD
from nyha_temporal_identification.data_model.dspy_model import NYHAMention


def _spans_text_match(p: NYHAMention, g: NYHAMention) -> bool:
    """Normalized text match fallback when IoU is just below threshold."""
    return p.span.strip().lower() == g.span.strip().lower()


def best_gold_match(
    pred: NYHAMention,
    gold_mentions: List[NYHAMention],
    already_matched: set[int],
    use_text_fallback: bool = True,
) -> tuple[Optional[int], float]:
    """
    Find the best matching gold mention for a prediction by IoU overlap,
    with an optional text-normalization fallback for near-miss boundaries.

    Args:
        pred:              Predicted mention to match.
        gold_mentions:     Full gold mention list.
        already_matched:   Indices of gold mentions already claimed.
        use_text_fallback: If True, accept a text-normalized match when IoU
                           is below OVERLAP_THRESHOLD.

    Returns:
        (best_idx, best_overlap) or (None, 0.0) if no match found.
    """
    best_idx: Optional[int] = None
    best_ov: float = 0.0

    for i, g in enumerate(gold_mentions):
        if i in already_matched:
            continue
        ov = pred.overlap(g)
        if ov > best_ov:
            best_ov = ov
            best_idx = i

    if use_text_fallback and (best_idx is None or best_ov < OVERLAP_THRESHOLD):
        for i, g in enumerate(gold_mentions):
            if i in already_matched:
                continue
            if _spans_text_match(pred, g):
                return i, OVERLAP_THRESHOLD
    return best_idx, best_ov


def _compute_class_f1(
    gold_mentions: List[NYHAMention],
    pred_mentions: List[NYHAMention],
    matched_pairs: List[tuple[int, int]],
    average: Literal["binary", "micro", "macro", "weighted", "samples"] = "macro",
) -> float:
    """
    Macro (or weighted) F1 over NYHA class labels for matched span pairs.

    Args:
        gold_mentions:  Full gold mention list.
        pred_mentions:  Full predicted mention list.
        matched_pairs:  List of (pred_idx, gold_idx) pairs from span matching.
        average:        Averaging strategy passed to sklearn.

    Returns:
        F1 score in [0.0, 1.0], or 0.0 if no matched pairs.
    """
    if not matched_pairs:
        return 0.0
    gold_labels: list[str] = [
        gold_mentions[g].get_nyha_class() or "unknown" for _, g in matched_pairs
    ]
    pred_labels: list[str] = [
        pred_mentions[p].get_nyha_class() or "unknown" for p, _ in matched_pairs
    ]
    return float(f1_score(gold_labels, pred_labels, average=average, zero_division=0))


def _compute_temporality_mcc(
    gold_mentions: List[NYHAMention],
    pred_mentions: List[NYHAMention],
    matched_pairs: List[tuple[int, int]],
) -> float:
    """
    Matthews Correlation Coefficient for binary temporality classification.

    MCC is preferred over accuracy for imbalanced binary labels.
    Returns 1.0/0.0 for the degenerate single-class case.

    Args:
        gold_mentions:  Full gold mention list.
        pred_mentions:  Full predicted mention list.
        matched_pairs:  List of (pred_idx, gold_idx) pairs from span matching.

    Returns:
        MCC in [-1.0, 1.0].
    """
    if not matched_pairs:
        return 0.0
    gold_labels = [gold_mentions[g].temporality for _, g in matched_pairs]
    pred_labels = [pred_mentions[p].temporality for p, _ in matched_pairs]
    if len(set(gold_labels)) < 2:
        return 1.0 if gold_labels == pred_labels else 0.0
    return float(matthews_corrcoef(gold_labels, pred_labels))


def _compute_temporality_macro_f1(
    gold_mentions: List[NYHAMention],
    pred_mentions: List[NYHAMention],
    matched_pairs: List[tuple[int, int]],
) -> float:
    """
    Macro F1 over temporality labels for matched span pairs.

    Gives equal weight to 'current' and 'historical' regardless of frequency,
    directly penalising both precision and recall failures on each class.
    Preferred over MCC as the optimisation signal because it is symmetric with
    how class_macro_f1 is computed and makes the historical failure mode
    visible to the optimiser.

    Returns 0.0 if no matched pairs exist.
    Returns 1.0 for the degenerate single-class case when all predictions are correct.
    """
    if not matched_pairs:
        return 0.0
    gold_labels = [gold_mentions[g].temporality for _, g in matched_pairs]
    pred_labels = [pred_mentions[p].temporality for p, _ in matched_pairs]
    if len(set(gold_labels)) < 2:
        return 1.0 if gold_labels == pred_labels else 0.0
    return float(f1_score(gold_labels, pred_labels, average="macro", zero_division=0))


def mention_f1(
    gold_mentions: List[NYHAMention],
    pred_mentions: List[NYHAMention],
    partial_credit: bool = False,
) -> dict:
    """
    Compute mention-level precision, recall and F1 between two mention lists,
    decomposed into three sub-scores for diagnostic insight:

      1. span_f1         — pure NER score: did we find the right spans?
      2. class_macro_f1  — macro F1 over NYHA classes for matched spans.
      3. temp_macro_f1   — macro F1 over current/historical (optimisation target)
      4. temp_mcc        — Matthews Correlation Coefficient (diagnostic)

    The composite F1 awards credit as follows:
      Full credit  (1.0): span match + class correct + temporality correct
      Half credit  (0.5): span match + class correct + temporality wrong
                          only when partial_credit=True
      No credit    (0.0): span match + class wrong, or no span match

    Edge cases:
      - Both empty              → perfect score (1.0 / 1.0 / 1.0), all counts 0
      - Gold empty, preds exist → all FP, all scores 0.0
      - Preds empty, gold exist → all FN, all scores 0.0

    Returns:
        {
            "precision", "recall", "f1",                      # composite
            "tp", "fp", "fn",                                  # composite counts
            "span_precision", "span_recall", "span_f1",        # NER sub-score
            "class_macro_f1", "class_weighted_f1",             # class sub-score
            "temp_macro_f1",                                   # temporality optimisation target
            "temp_mcc", "temp_mcc_normalized",                 # temporality diagnostics
            "class_correct", "class_wrong",
            "temp_correct", "temp_wrong",
            "span_matched",
        }
    """
    _zero = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "span_precision": 0.0,
        "span_recall": 0.0,
        "span_f1": 0.0,
        "class_macro_f1": 0.0,
        "class_weighted_f1": 0.0,
        "temp_mcc": 0.0,
        "temp_mcc_normalized": 0.5,
        "temp_macro_f1": 0.0,
        "class_correct": 0,
        "class_wrong": 0,
        "temp_correct": 0,
        "temp_wrong": 0,
        "span_matched": 0,
    }
    if not gold_mentions and not pred_mentions:
        return {
            **_zero,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "span_precision": 1.0,
            "span_recall": 1.0,
            "span_f1": 1.0,
            "class_macro_f1": 1.0,
            "class_weighted_f1": 1.0,
            "temp_mcc": 1.0,
            "temp_mcc_normalized": 1.0,
            "temp_macro_f1": 1.0,
        }
    if not gold_mentions:
        return {**_zero, "fp": len(pred_mentions)}
    if not pred_mentions:
        return {**_zero, "fn": len(gold_mentions)}

    matched_gold: set[int] = set()
    span_matched_gold: set[int] = set()
    matched_pairs: List[tuple[int, int]] = []

    true_positives = 0.0
    class_correct = 0
    class_wrong = 0
    temp_correct = 0
    temp_wrong = 0

    for pred_idx, p in enumerate(pred_mentions):
        # Pure NER matching (label-agnostic)
        span_idx, span_ov = best_gold_match(p, gold_mentions, span_matched_gold)
        if span_idx is not None and span_ov >= OVERLAP_THRESHOLD:
            span_matched_gold.add(span_idx)

        # Composite matching (label-aware)
        best_idx, best_ov = best_gold_match(p, gold_mentions, matched_gold)
        if best_idx is None or best_ov < OVERLAP_THRESHOLD:
            continue

        g = gold_mentions[best_idx]
        class_match = p.get_nyha_class() == g.get_nyha_class()
        temp_match = p.temporality == g.temporality

        matched_pairs.append((pred_idx, best_idx))
        if class_match:
            class_correct += 1
        else:
            class_wrong += 1
        if temp_match:
            temp_correct += 1
        else:
            temp_wrong += 1

        if class_match and temp_match:
            true_positives += 1.0
            matched_gold.add(best_idx)
        elif partial_credit and class_match:
            true_positives += 0.5
            matched_gold.add(best_idx)

    # Composite F1
    fp = len(pred_mentions) - len(matched_gold)
    fn = len(gold_mentions) - len(matched_gold)
    precision = true_positives / len(pred_mentions)
    recall = true_positives / len(gold_mentions)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Span F1 (NER only)
    span_tp = len(span_matched_gold)
    span_precision = span_tp / len(pred_mentions)
    span_recall = span_tp / len(gold_mentions)
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if (span_precision + span_recall) > 0
        else 0.0
    )

    # Label sub-scores
    class_macro_f1 = _compute_class_f1(
        gold_mentions, pred_mentions, matched_pairs, average="macro"
    )
    class_weighted_f1 = _compute_class_f1(
        gold_mentions, pred_mentions, matched_pairs, average="weighted"
    )
    temp_mcc = _compute_temporality_mcc(gold_mentions, pred_mentions, matched_pairs)
    temp_mcc_normalized = (temp_mcc + 1) / 2
    temp_macro_f1 = _compute_temporality_macro_f1(
        gold_mentions, pred_mentions, matched_pairs
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": true_positives,
        "fp": fp,
        "fn": fn,
        "span_precision": round(span_precision, 4),
        "span_recall": round(span_recall, 4),
        "span_f1": round(span_f1, 4),
        "class_macro_f1": round(class_macro_f1, 4),
        "class_weighted_f1": round(class_weighted_f1, 4),
        "temp_mcc": round(temp_mcc, 4),
        "temp_mcc_normalized": round(temp_mcc_normalized, 4),
        "temp_macro_f1": round(temp_macro_f1, 4),
        "class_correct": class_correct,
        "class_wrong": class_wrong,
        "temp_correct": temp_correct,
        "temp_wrong": temp_wrong,
        "span_matched": class_correct + class_wrong,
    }


def nyha_dspy_metric(
    example: dspy.Example,
    prediction,
    trace=None,
) -> float:
    """
    DSPy-compatible metric for BootstrapFewShot and MIPROv2.

    Returns a weighted composite score:
      0.10 × span_f1           — span detection is the primary task
      0.10 × class_macro_f1    — macro F1 penalises ignoring rare classes (I, IV)
      0.80 × temp_macro_f1     — macro F1 over current/historical, equal weight
                                  to both classes regardless of frequency;
                                  replaces temp_mcc_normalized which masked the
                                  historical recall collapse under imbalance

    temp_mcc and temp_mcc_normalized are still computed and logged as
    diagnostics but no longer drive optimisation.
    """
    gold = example.mentions
    pred = prediction if isinstance(prediction, list) else []
    scores = mention_f1(gold, pred, partial_credit=False)

    temp_macro_f1_weight = 0.8
    span_f1_weight = 0.1
    class_macro_f1_weight = 1 - temp_macro_f1_weight - span_f1_weight
    composite = (
        span_f1_weight * scores["span_f1"]
        + class_macro_f1_weight * scores["class_macro_f1"]
        + temp_macro_f1_weight * scores["temp_macro_f1"]
    )

    logging.debug(
        "nyha_metric | composite=%.4f  span_f1=%.4f  class_macro_f1=%.4f  "
        "temp_macro_f1=%.4f  temp_mcc=%.4f (norm=%.4f) | "
        "tp=%.1f fp=%d fn=%d | span_matched=%d",
        composite,
        scores["span_f1"],
        scores["class_macro_f1"],
        scores["temp_macro_f1"],
        scores["temp_mcc"],
        scores["temp_mcc_normalized"],
        scores["tp"],
        scores["fp"],
        scores["fn"],
        scores["span_matched"],
    )

    return composite
