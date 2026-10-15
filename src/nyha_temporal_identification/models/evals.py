from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.ensemble import GradientBoostingClassifier

from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.evaluation.evaluation import (
    mention_f1,
    print_evaluation_report,
)
from nyha_temporal_identification.models.constants import (
    NOTE_ID_COL,
    NYHA_PRED_END_INDEX,
    NYHA_PRED_START_INDEX,
    NYHA_PRED_VEC_COL,
)
from nyha_temporal_identification.models.models import ThresholdClassifier


def _predict(
    model: Any,
    X: np.ndarray | sp.csr_matrix,
    numeric_indices: list[int] | None = None,
) -> np.ndarray:
    """
    Unified predict dispatcher.

    ThresholdClassifier has a custom signature (needs numeric_indices);
    all other sklearn estimators use the standard .predict(X) interface.
    GradientBoostingClassifier requires dense input.
    """
    if isinstance(model, ThresholdClassifier):
        assert numeric_indices is not None
        return model.predict(X, numeric_indices)

    # Dense conversion for GBM if needed
    if isinstance(model, GradientBoostingClassifier) and sp.issparse(X):
        X_tv_sparse = cast(sp.csr_matrix, X)
        X = X_tv_sparse.toarray()

    return model.predict(X)


def evaluate_model(
    name: str,
    models: list[Any],
    X_tests: list[np.ndarray | sp.csr_matrix],
    test_reports_folds: list[list[NYHANote]],
    invalid_doc_id_report_predictions_folds: list[dict[str, list[NYHAMention]]],
    numeric_indices_folds: list[list[int]] | None,
    save_dir: Path,
    test_df_valid_folds: list[pl.DataFrame],
    verbose: bool = False,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, float]:
    """
    Evaluate K fitted fold-models on K held-out test folds and merge predictions.

    Each fold contributes predictions for its own held-out test documents.
    The fold-level mention predictions are merged and passed to the report
    generator as one combined corpus-level evaluation.

    Parameters
    ----------
    name                                : str
    models                              : list[fitted classifier], length = K
    X_tests                             : list[combined test matrices], length = K
    test_reports_folds                  : list of per-fold report document lists
    invalid_doc_id_report_predictions_folds : list of per-fold invalid mentions
    numeric_indices_folds               : per-fold numeric indices for threshold model
                                          (None for sklearn estimators)
    save_dir                            : Path
    test_df_valid_folds                 : list of per-fold valid test dataframes
    verbose                             : bool  Print per-fold metrics logs.
    """
    k = len(models)
    if not (
        len(X_tests) == k
        and len(test_reports_folds) == k
        and len(invalid_doc_id_report_predictions_folds) == k
        and len(test_df_valid_folds) == k
    ):
        raise ValueError("All fold inputs to evaluate_model must have the same length")

    if numeric_indices_folds is not None and len(numeric_indices_folds) != k:
        raise ValueError(
            "numeric_indices_folds must be None or have one entry per fold"
        )

    merged_test_reports: list[NYHANote] = []
    merged_report_predictions: list[list[NYHAMention]] = []
    fold_micro_f1_scores: list[float] = []
    fold_gold_mentions: list[float] = []

    def _micro_f1_from_counts(tp: float, fp: float, fn: float) -> float:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    for fold_idx, (
        model,
        X_test,
        test_reports,
        invalid_predictions,
        test_df_valid,
    ) in enumerate(
        zip(
            models,
            X_tests,
            test_reports_folds,
            invalid_doc_id_report_predictions_folds,
            test_df_valid_folds,
            strict=True,
        )
    ):
        numeric_indices = (
            numeric_indices_folds[fold_idx]
            if numeric_indices_folds is not None
            else None
        )
        y_pred = _predict(model, X_test, numeric_indices)

        doc_id_pred_mentions: dict[str, list[NYHAMention]] = {}
        for index, row in enumerate(test_df_valid.iter_rows(named=True)):
            note_id = row[NOTE_ID_COL]
            nyha_pred_start_index = row[NYHA_PRED_START_INDEX]
            nyha_pred_end_index = row[NYHA_PRED_END_INDEX]
            predicted_nyha_vector = row[NYHA_PRED_VEC_COL]
            pred_value = y_pred[index]
            if pred_value == 1:
                pred_temporality = "current"
            elif pred_value == 0:
                pred_temporality = "historical"
            else:
                raise ValueError(f"Unexpected predicted value: {pred_value}")

            pred_mention = NYHAMention(
                char_start=nyha_pred_start_index,
                char_end=nyha_pred_end_index,
                span="",
                nyha_vector=predicted_nyha_vector,
                temporality=pred_temporality,
            )
            doc_id_pred_mentions.setdefault(note_id, []).append(pred_mention)

        for doc_id, mentions in invalid_predictions.items():
            doc_id_pred_mentions.setdefault(doc_id, []).extend(mentions)

        fold_report_predictions = [
            doc_id_pred_mentions.get(doc.old_note_id, []) for doc in test_reports
        ]

        fold_tp = 0.0
        fold_fp = 0.0
        fold_fn = 0.0
        for doc, pred_mentions in zip(
            test_reports, fold_report_predictions, strict=True
        ):
            metrics = mention_f1(doc.mentions, pred_mentions)
            fold_tp += float(metrics["tp"])
            fold_fp += float(metrics["fp"])
            fold_fn += float(metrics["fn"])
        fold_micro_f1 = _micro_f1_from_counts(fold_tp, fold_fp, fold_fn)
        fold_micro_f1_scores.append(fold_micro_f1)
        fold_gold_mentions.append(fold_tp + fold_fn)

        if verbose:
            print(
                f"[eval] {name} fold {fold_idx + 1}/{k} "
                f"micro-F1: {fold_micro_f1:.4f} "
                f"(tp={fold_tp:.1f}, fp={fold_fp:.1f}, fn={fold_fn:.1f})"
            )

        merged_test_reports.extend(test_reports)
        merged_report_predictions.extend(fold_report_predictions)

    merged_tp = 0.0
    merged_fp = 0.0
    merged_fn = 0.0
    for doc, pred_mentions in zip(
        merged_test_reports,
        merged_report_predictions,
        strict=True,
    ):
        metrics = mention_f1(doc.mentions, pred_mentions)
        merged_tp += float(metrics["tp"])
        merged_fp += float(metrics["fp"])
        merged_fn += float(metrics["fn"])

    merged_micro_f1 = _micro_f1_from_counts(merged_tp, merged_fp, merged_fn)
    fold_f1_mean = float(np.mean(fold_micro_f1_scores)) if fold_micro_f1_scores else 0.0
    fold_f1_std = float(np.std(fold_micro_f1_scores)) if fold_micro_f1_scores else 0.0
    fold_f1_weighted_mean = (
        float(np.average(fold_micro_f1_scores, weights=fold_gold_mentions))
        if fold_micro_f1_scores and sum(fold_gold_mentions) > 0
        else 0.0
    )

    print_evaluation_report(
        stage=f"Test {name} (K-fold merged)",
        documents=merged_test_reports,
        predictions=merged_report_predictions,
        output_dir_path=save_dir,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )

    return {
        "fold_micro_f1_mean": fold_f1_mean,
        "fold_micro_f1_std": fold_f1_std,
        "fold_micro_f1_weighted_mean": fold_f1_weighted_mean,
        "merged_micro_f1": float(merged_micro_f1),
        "n_folds": float(k),
        "total_gold_mentions": float(sum(fold_gold_mentions)),
    }
