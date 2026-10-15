"""
Binary Classification Pipeline: Classify temporal_status as "historical" or "current".

Uses Polars for data handling and scikit-learn for three classifiers: Threshold,
Logistic Regression, and Gradient Boosting. Supports optional text features via
bag-of-words extraction with Dutch stemming.

Default features: char_distance, days_distance. Pass additional features via extra_features.
Text branch: Governed by feature_mode ("base" / "base_text" / "text").
"""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from nyha_temporal_identification.constants import (
    BASELINES_SPLITS_FILE_NAME,
    ID_SPLITS_FILE_NAME,
    OUTPUT_IMAGES_FOLDER_NAME,
    SINGLE_SPLIT_FILE_NAME,
    SPLITS_FOLDER_NAME,
    TEMPORAL_STATUS_PREDICTION_FOLDER_NAME,
)
from nyha_temporal_identification.data_model.dspy_model import NYHANote
from nyha_temporal_identification.models.constants import (
    BASE_FEATURES,
    BASELINES_FOLDER_NAME,
    DEFAULT_WINDOW_SIZE,
    FEATURE_MODE_BASE,
    FEATURE_MODE_BASE_TEXT,
    FEATURE_MODE_TEXT,
    FEATURE_MODES,
    POSITIVE_CLASS,
    RANDOM_STATE,
    TARGET_COL,
)
from nyha_temporal_identification.data_handling.data_loading import load_note_patient_groups
from nyha_temporal_identification.models.data_loading import (
    build_test_dfs,
    build_train_val_dfs,
)
from nyha_temporal_identification.models.evals import evaluate_model
from nyha_temporal_identification.models.features_handling import (
    analyse_features,
    build_feature_matrix,
)
from nyha_temporal_identification.models.models import (
    ThresholdClassifier,
    build_gradient_boosting,
    build_logistic_regression,
)
from nyha_temporal_identification.models.output import (
    plot_feature_importance,
)
from nyha_temporal_identification.models.text_feature_extractor import (
    TextFeatureExtractor,
)

matplotlib.use("Agg")

warnings.filterwarnings("ignore")


def _train_kfold_cv(
    documents: list[NYHANote],
    document_ids: list[str],
    notes_by_id: dict[str, NYHANote],
    numeric_features: list[str],
    output_image_dir: Path,
    splits_folder_path: Path,
    comparisons_folder_path: Path,
    threshold_classifier_folder_path: Path | None,
    logistic_regression_folder_path: Path,
    gradient_boosting_folder_path: Path,
    feature_mode: str,
    window_size: int,
    apply_stemming: bool,
    max_bow_features: int | None,
    n_splits: int,
    eval_verbose: bool,
    doc_label_tie_break: str,
    n_bootstrap: int,
    note_to_group: dict[str, str],
    _document_label,
    _split_mentions_stats,
) -> dict[str, dict[str, float]]:
    """Run K-fold cross-validation pipeline."""
    has_text = feature_mode in {FEATURE_MODE_BASE_TEXT, FEATURE_MODE_TEXT}
    has_numeric = feature_mode in {FEATURE_MODE_BASE, FEATURE_MODE_BASE_TEXT}  # noqa: F841
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for K-fold evaluation")

    doc_labels = np.array([_document_label(doc_id) for doc_id in document_ids])
    groups = np.array([note_to_group[doc_id] for doc_id in document_ids])

    try:
        outer_splitter = StratifiedGroupKFold(n_splits=n_splits)
        # Dry-run to verify feasibility before consuming the iterator
        list(outer_splitter.split(document_ids, doc_labels, groups))
        outer_splits = outer_splitter.split(document_ids, doc_labels, groups)
        outer_splitter_name = "stratified_group_kfold"
    except ValueError:
        outer_splitter = GroupKFold(n_splits=n_splits)
        outer_splits = outer_splitter.split(document_ids, groups=groups)
        outer_splitter_name = "group_kfold"
        print(
            "[warning] StratifiedGroupKFold not feasible; falling back to GroupKFold."
        )

    split_folds: list[dict[str, Any]] = []
    threshold_models: list[ThresholdClassifier] = []
    lr_models: list[Any] = []
    gb_models: list[Any] = []
    X_tests_folds: list[np.ndarray | sp.csr_matrix] = []
    test_df_valid_folds = []
    test_reports_folds = []
    invalid_predictions_folds = []
    threshold_numeric_indices_folds: list[list[int]] = []

    for fold_index, (train_val_idx, test_idx) in enumerate(outer_splits):
        fold_number = fold_index + 1
        fold_output_dir = output_image_dir / f"fold_{fold_number:02d}"
        os.makedirs(fold_output_dir, exist_ok=True)

        train_val_ids = [document_ids[i] for i in train_val_idx]
        test_ids = [document_ids[i] for i in test_idx]

        if len(train_val_ids) < 2:
            raise ValueError(
                "Each fold requires at least 2 train+val documents to split train/val"
            )

        # Patient-grouped inner train/val split (~10% of fold docs go to val)
        patient_to_ids: dict[str, list[str]] = {}
        for doc_id in train_val_ids:
            g = note_to_group[doc_id]
            patient_to_ids.setdefault(g, []).append(doc_id)

        patient_groups = list(patient_to_ids.keys())
        random.Random(RANDOM_STATE + fold_index).shuffle(patient_groups)

        target_val = max(1, int(round(0.1 * len(train_val_ids))))
        val_ids: list[str] = []
        for pg in reversed(patient_groups):
            val_ids.extend(patient_to_ids[pg])
            if len(val_ids) >= target_val:
                break

        val_ids_set = set(val_ids)
        train_ids = [doc_id for doc_id in train_val_ids if doc_id not in val_ids_set]

        if not train_ids:
            # Edge case: move first patient group from val back to train
            first_pg = next(
                pg for pg in patient_groups if patient_to_ids[pg][0] in val_ids_set
            )
            for doc_id in patient_to_ids[first_pg]:
                val_ids.remove(doc_id)
                train_ids.append(doc_id)

        split_folds.append(
            {
                "fold": fold_number,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "stats": {
                    "train": _split_mentions_stats(train_ids),
                    "val": _split_mentions_stats(val_ids),
                    "test": _split_mentions_stats(test_ids),
                },
                "stratification": {
                    "outer_splitter": outer_splitter_name,
                    "inner_grouping": "patient",
                    "target_definition": (
                        "1 if current mentions > historical mentions, else 0; "
                        "ties resolved by doc_label_tie_break"
                    ),
                    "document_label_strategy": "majority_vote",
                    "doc_label_tie_break": doc_label_tie_break,
                },
            }
        )

        train_df, val_df = build_train_val_dfs(
            documents,
            train_ids,
            val_ids,
        )
        test_df_valid, test_reports, invalid_doc_id_report_predictions = build_test_dfs(
            documents,
            test_ids,
        )

        selected_numeric, mi_scores = analyse_features(train_df, numeric_features)

        bow_train: sp.csr_matrix | None = None
        bow_val: sp.csr_matrix | None = None
        bow_test: sp.csr_matrix | None = None

        if has_text:
            text_extractor = TextFeatureExtractor(
                window_size=window_size,
                apply_stemming=apply_stemming,
                max_features=max_bow_features,
            )
            bow_train = text_extractor.fit_transform(train_df)
            bow_val = text_extractor.transform(val_df)
            bow_test = text_extractor.transform(test_df_valid)
            print(
                f"[fold {fold_number}/{n_splits}] [text] BoW matrix shapes — "
                f"train: {bow_train.shape}, val: {bow_val.shape}, test: {bow_test.shape}"
            )

        X_train = build_feature_matrix(train_df, selected_numeric, bow_train)
        X_val = build_feature_matrix(val_df, selected_numeric, bow_val)
        X_test = build_feature_matrix(test_df_valid, selected_numeric, bow_test)

        y_train = train_df[TARGET_COL].to_numpy()
        y_val = val_df[TARGET_COL].to_numpy()
        numeric_indices = list(range(len(selected_numeric)))

        if feature_mode != FEATURE_MODE_TEXT:
            threshold_clf = ThresholdClassifier()
            threshold_clf.fit(train_df, val_df, selected_numeric)
            threshold_clf.plot_threshold_search(
                train_df, val_df, selected_numeric, fold_output_dir
            )
            threshold_models.append(threshold_clf)
            threshold_numeric_indices_folds.append(numeric_indices)

        lr_clf = build_logistic_regression(X_train, y_train, X_val, y_val, has_text)
        gb_clf = build_gradient_boosting(X_train, y_train, X_val, y_val)
        plot_feature_importance(
            mi_scores, gb_clf, selected_numeric, has_text, fold_output_dir
        )

        lr_models.append(lr_clf)
        gb_models.append(gb_clf)
        X_tests_folds.append(X_test)
        test_df_valid_folds.append(test_df_valid)
        test_reports_folds.append(test_reports)
        invalid_predictions_folds.append(invalid_doc_id_report_predictions)

    split_output_path = Path(splits_folder_path) / BASELINES_SPLITS_FILE_NAME
    with open(split_output_path, "w") as fp:
        json.dump(
            {
                "cv_type": "kfold",
                "grouping": "patient",
                "outer_splitter": outer_splitter_name,
                "n_splits": n_splits,
                "total_documents": len(document_ids),
                "folds": split_folds,
            },
            fp,
            indent=2,
        )

    results: dict[str, dict[str, float]] = {}
    if feature_mode != FEATURE_MODE_TEXT:
        results["threshold_classifier"] = evaluate_model(
            name="threshold_classifier",
            models=threshold_models,
            X_tests=X_tests_folds,
            test_reports_folds=test_reports_folds,
            invalid_doc_id_report_predictions_folds=invalid_predictions_folds,
            numeric_indices_folds=threshold_numeric_indices_folds,
            save_dir=threshold_classifier_folder_path,
            test_df_valid_folds=test_df_valid_folds,
            verbose=eval_verbose,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=RANDOM_STATE,
        )
    results["logistic_regression"] = evaluate_model(
        name="logistic_regression",
        models=lr_models,
        X_tests=X_tests_folds,
        test_reports_folds=test_reports_folds,
        invalid_doc_id_report_predictions_folds=invalid_predictions_folds,
        numeric_indices_folds=None,
        save_dir=logistic_regression_folder_path,
        test_df_valid_folds=test_df_valid_folds,
        verbose=eval_verbose,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=RANDOM_STATE,
    )
    results["gradient_boosting"] = evaluate_model(
        name="gradient_boosting",
        models=gb_models,
        X_tests=X_tests_folds,
        test_reports_folds=test_reports_folds,
        invalid_doc_id_report_predictions_folds=invalid_predictions_folds,
        numeric_indices_folds=None,
        save_dir=gradient_boosting_folder_path,
        test_df_valid_folds=test_df_valid_folds,
        verbose=eval_verbose,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=RANDOM_STATE,
    )

    return results


def _train_single_split(
    documents: list[NYHANote],
    document_ids: list[str],
    notes_by_id: dict[str, NYHANote],
    numeric_features: list[str],
    splits_folder_path: Path,
    comparisons_folder_path: Path,
    threshold_classifier_folder_path: Path | None,
    logistic_regression_folder_path: Path,
    gradient_boosting_folder_path: Path,
    feature_mode: str,
    window_size: int,
    apply_stemming: bool,
    max_bow_features: int | None,
    eval_verbose: bool,
    doc_label_tie_break: str,
    n_bootstrap: int,
    _document_label,
    _split_mentions_stats,
) -> dict[str, dict[str, float]]:
    """Train on fixed train+val split, evaluate on held-out test split."""
    has_text = feature_mode in {FEATURE_MODE_BASE_TEXT, FEATURE_MODE_TEXT}
    # Load split IDs saved by the DSPy runner at splits/id_splits.json.
    # splits_folder_path = .../splits/baselines/<eval_mode>/<feature_mode>/
    # so we need three .parent calls to reach .../splits/
    split_file = splits_folder_path.parent.parent.parent / ID_SPLITS_FILE_NAME
    print(f"[single-split] Looking for split file at: {split_file}")

    if not split_file.exists():
        raise FileNotFoundError(
            f"Cannot find split file: {split_file}. "
            f"Run run_dspy_pipeline first to generate {ID_SPLITS_FILE_NAME}."
        )

    with open(split_file, "r") as f:
        split_data = json.load(f)

    train_ids = split_data.get("train_ids", [])
    val_ids = split_data.get("val_ids", [])
    test_ids = split_data.get("test_ids", [])

    print(
        f"[single-split] Loaded split data: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )

    if not train_ids or not test_ids:
        raise ValueError(
            f"Split file must contain non-empty 'train_ids' and 'test_ids'. "
            f"Got: train_ids={len(train_ids)}, test_ids={len(test_ids)}"
        )

    # Combine train and val for single-split training
    combined_train_ids = train_ids + val_ids
    print(
        f"[single-split] Loaded splits: train+val={len(combined_train_ids)}, test={len(test_ids)}"
    )

    # Build dataframes
    combined_train_df, _ = build_train_val_dfs(documents, combined_train_ids, [])
    test_df_valid, test_reports, invalid_doc_id_report_predictions = build_test_dfs(
        documents, test_ids
    )

    # Feature analysis
    selected_numeric, mi_scores = analyse_features(combined_train_df, numeric_features)

    # Initialize text extractor if needed
    text_extractor: TextFeatureExtractor | None = None
    bow_train: sp.csr_matrix | None = None
    bow_test: sp.csr_matrix | None = None

    if has_text:
        text_extractor = TextFeatureExtractor(
            window_size=window_size,
            apply_stemming=apply_stemming,
            max_features=max_bow_features,
        )
        bow_train = text_extractor.fit_transform(combined_train_df)
        bow_test = text_extractor.transform(test_df_valid)
        print(
            f"[single-split] [text] BoW matrix shapes — "
            f"train: {bow_train.shape}, test: {bow_test.shape}"
        )

    X_test = build_feature_matrix(test_df_valid, selected_numeric, bow_test)

    numeric_indices = list(range(len(selected_numeric)))

    # For single split, use a simple val split within training for threshold/LR tuning
    val_frac = min(
        0.15, max(0.1, int(len(combined_train_ids) * 0.1) / len(combined_train_ids))
    )
    combined_train_list = list(combined_train_ids)
    split_idx = int(len(combined_train_list) * (1 - val_frac))
    internal_train_ids = combined_train_list[:split_idx]
    internal_val_ids = combined_train_list[split_idx:]
    internal_train_df, internal_val_df = build_train_val_dfs(
        documents, internal_train_ids, internal_val_ids
    )

    if feature_mode != FEATURE_MODE_TEXT:
        threshold_clf = ThresholdClassifier()
        threshold_clf.fit(internal_train_df, internal_val_df, selected_numeric)

    # Build feature matrices for internal train/val split
    bow_internal_train: sp.csr_matrix | None = None
    bow_internal_val: sp.csr_matrix | None = None

    if has_text and text_extractor is not None:
        bow_internal_train = text_extractor.transform(internal_train_df)
        bow_internal_val = text_extractor.transform(internal_val_df)

    X_internal_train = build_feature_matrix(
        internal_train_df, selected_numeric, bow_internal_train
    )
    X_internal_val = build_feature_matrix(
        internal_val_df, selected_numeric, bow_internal_val
    )

    y_internal_train = internal_train_df[TARGET_COL].to_numpy()
    y_internal_val = internal_val_df[TARGET_COL].to_numpy()

    lr_clf = build_logistic_regression(
        X_internal_train, y_internal_train, X_internal_val, y_internal_val, has_text
    )
    gb_clf = build_gradient_boosting(
        X_internal_train, y_internal_train, X_internal_val, y_internal_val
    )

    # Evaluate single model on test set
    results: dict[str, dict[str, float]] = {}
    if feature_mode != FEATURE_MODE_TEXT:
        results["threshold_classifier"] = evaluate_model(
            name="threshold_classifier",
            models=[threshold_clf],
            X_tests=[X_test],
            test_reports_folds=[test_reports],
            invalid_doc_id_report_predictions_folds=[invalid_doc_id_report_predictions],
            numeric_indices_folds=[numeric_indices],
            save_dir=threshold_classifier_folder_path,
            test_df_valid_folds=[test_df_valid],
            verbose=eval_verbose,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=RANDOM_STATE,
        )
    results["logistic_regression"] = evaluate_model(
        name="logistic_regression",
        models=[lr_clf],
        X_tests=[X_test],
        test_reports_folds=[test_reports],
        invalid_doc_id_report_predictions_folds=[invalid_doc_id_report_predictions],
        numeric_indices_folds=None,
        save_dir=logistic_regression_folder_path,
        test_df_valid_folds=[test_df_valid],
        verbose=eval_verbose,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=RANDOM_STATE,
    )
    results["gradient_boosting"] = evaluate_model(
        name="gradient_boosting",
        models=[gb_clf],
        X_tests=[X_test],
        test_reports_folds=[test_reports],
        invalid_doc_id_report_predictions_folds=[invalid_doc_id_report_predictions],
        numeric_indices_folds=None,
        save_dir=gradient_boosting_folder_path,
        test_df_valid_folds=[test_df_valid],
        verbose=eval_verbose,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=RANDOM_STATE,
    )

    return results


def train_baselines(
    base_folder_path: str,
    documents: list[NYHANote],
    extra_features: list[str] | None = None,
    feature_mode: str = FEATURE_MODE_BASE,
    window_size: int = DEFAULT_WINDOW_SIZE,
    apply_stemming: bool = False,
    max_bow_features: int | None = None,
    n_splits: int = 5,
    eval_verbose: bool = False,
    doc_label_tie_break: str = "current",
    use_kfold_cv: bool = True,
    n_bootstrap: int = 0,
) -> dict[str, dict[str, float]]:
    """
    End-to-end pipeline entry point for temporality classification.

    Supports two evaluation modes:
    1. K-fold cross-validation (use_kfold_cv=True): splits data into K folds,
       trains and evaluates K models, saves results to cv/ subfolder.
    2. Single hold-out split (use_kfold_cv=False): loads fixed train/val/test IDs
       from ID_SPLITS_FILE_NAME, trains once on train+val, evaluates on held-out
       test, saves results to single/ subfolder.

    Parameters
    ----------
    base_folder_path : str            Path to project root folder.
    documents        : list[NYHANote] Pre-loaded documents with NYHA annotations.
    extra_features   : list[str]|None Additional numeric feature columns (ignored
                                      in "text" mode).
    feature_mode     : str            One of "base", "base_text", "text".
                                      Controls which inputs are used.
    window_size      : int            Characters on each side of span.
    apply_stemming   : bool           Apply Dutch SnowballStemmer (text branch).
    max_bow_features : int|None       Cap CountVectorizer vocabulary size.
    n_splits         : int            Number of folds for K-fold CV (ignored if
                                      use_kfold_cv=False).
    eval_verbose     : bool           Print per-fold/model evaluation logs.
    doc_label_tie_break : str         Majority-vote tie-breaker:
                                      "current" or "historical".
    use_kfold_cv     : bool           If True, run K-fold CV; else load single
                                      split from ID_SPLITS_FILE_NAME.
    n_bootstrap      : int            Document-level bootstrap resamples for
                                      confidence intervals on the evaluation
                                      report metrics; 0 disables (default).

    Returns
    -------
    results : dict[str, dict[str, float]]
    """

    # ── Validate feature_mode ─────────────────────────────────────────────────
    if feature_mode not in FEATURE_MODES:
        raise ValueError(
            f"Invalid feature_mode '{feature_mode}'. Must be one of {FEATURE_MODES}."
        )

    has_text = feature_mode in {FEATURE_MODE_BASE_TEXT, FEATURE_MODE_TEXT}  # noqa: F841

    # ── Determine evaluation mode and setup folder structure ───────────────────
    eval_mode = "cv" if use_kfold_cv else "single"

    output_image_dir = (
        Path(base_folder_path)
        / OUTPUT_IMAGES_FOLDER_NAME
        / TEMPORAL_STATUS_PREDICTION_FOLDER_NAME
        / BASELINES_FOLDER_NAME
        / eval_mode
        / feature_mode
    )
    splits_folder_path = (
        Path(base_folder_path)
        / TEMPORAL_STATUS_PREDICTION_FOLDER_NAME
        / SPLITS_FOLDER_NAME
        / BASELINES_FOLDER_NAME
        / eval_mode
        / feature_mode
    )
    comparisons_folder_path = (
        Path(base_folder_path)
        / TEMPORAL_STATUS_PREDICTION_FOLDER_NAME
        / BASELINES_FOLDER_NAME
        / eval_mode
        / feature_mode
    )
    threshold_classifier_folder_path: Path | None = None
    logistic_regression_folder_path = comparisons_folder_path / "logistic_regression"
    gradient_boosting_folder_path = comparisons_folder_path / "gradient_boosting"

    paths = [
        output_image_dir,
        splits_folder_path,
        comparisons_folder_path,
        logistic_regression_folder_path,
        gradient_boosting_folder_path,
    ]
    if feature_mode != FEATURE_MODE_TEXT:
        threshold_classifier_folder_path = (
            comparisons_folder_path / "threshold_classifier"
        )
        paths.append(threshold_classifier_folder_path)
    for path in paths:
        os.makedirs(path, exist_ok=True)

    # ── Compose numeric feature list ─────────────────────────────────────────
    if feature_mode == FEATURE_MODE_TEXT:
        numeric_features: list[str] = []
    else:
        numeric_features = list(
            dict.fromkeys(BASE_FEATURES + (extra_features or []))
        )
    print(f"[config] Numeric features : {numeric_features}")
    print(f"[config] feature_mode     : {feature_mode}")
    print(f"[config] Evaluation mode  : {eval_mode.upper()}")

    # ── Validate configuration ───────────────────────────────────────────────
    if doc_label_tie_break not in {"current", "historical"}:
        raise ValueError("doc_label_tie_break must be either 'current' or 'historical'")
    if use_kfold_cv and n_splits < 2:
        raise ValueError("n_splits must be >= 2 for K-fold evaluation")

    document_ids = [note.old_note_id for note in documents]
    notes_by_id = {note.old_note_id: note for note in documents}

    def _document_label(doc_id: str) -> int:
        # Majority-vote document label built from mention-level temporality.
        # Tie-breaker is configurable to support sensitivity analyses.
        mentions = notes_by_id[doc_id].mentions
        current_mentions = sum(1 for m in mentions if m.temporality == POSITIVE_CLASS)
        historical_mentions = len(mentions) - current_mentions
        if current_mentions > historical_mentions:
            return 1
        if historical_mentions > current_mentions:
            return 0
        return 1 if doc_label_tie_break == "current" else 0

    def _split_mentions_stats(ids: list[str]) -> dict[str, float]:
        total_mentions = 0
        current_mentions = 0
        for doc_id in ids:
            for mention in notes_by_id[doc_id].mentions:
                total_mentions += 1
                if mention.temporality == POSITIVE_CLASS:
                    current_mentions += 1
        historical_mentions = total_mentions - current_mentions
        current_ratio = current_mentions / total_mentions if total_mentions > 0 else 0.0
        return {
            "documents": float(len(ids)),
            "mentions_total": float(total_mentions),
            "mentions_current": float(current_mentions),
            "mentions_historical": float(historical_mentions),
            "current_ratio": float(current_ratio),
        }

    # ── Route to evaluation mode ──────────────────────────────────────────────
    if use_kfold_cv:
        note_to_group = load_note_patient_groups(base_folder_path, document_ids)
        results = _train_kfold_cv(
            documents=documents,
            document_ids=document_ids,
            notes_by_id=notes_by_id,
            numeric_features=numeric_features,
            output_image_dir=output_image_dir,
            splits_folder_path=splits_folder_path,
            comparisons_folder_path=comparisons_folder_path,
            threshold_classifier_folder_path=threshold_classifier_folder_path,
            logistic_regression_folder_path=logistic_regression_folder_path,
            gradient_boosting_folder_path=gradient_boosting_folder_path,
            feature_mode=feature_mode,
            window_size=window_size,
            apply_stemming=apply_stemming,
            max_bow_features=max_bow_features,
            n_splits=n_splits,
            eval_verbose=eval_verbose,
            doc_label_tie_break=doc_label_tie_break,
            n_bootstrap=n_bootstrap,
            note_to_group=note_to_group,
            _document_label=_document_label,
            _split_mentions_stats=_split_mentions_stats,
        )
    else:
        results = _train_single_split(
            documents=documents,
            document_ids=document_ids,
            notes_by_id=notes_by_id,
            numeric_features=numeric_features,
            splits_folder_path=splits_folder_path,
            comparisons_folder_path=comparisons_folder_path,
            threshold_classifier_folder_path=threshold_classifier_folder_path,
            logistic_regression_folder_path=logistic_regression_folder_path,
            gradient_boosting_folder_path=gradient_boosting_folder_path,
            feature_mode=feature_mode,
            window_size=window_size,
            apply_stemming=apply_stemming,
            max_bow_features=max_bow_features,
            eval_verbose=eval_verbose,
            doc_label_tie_break=doc_label_tie_break,
            n_bootstrap=n_bootstrap,
            _document_label=_document_label,
            _split_mentions_stats=_split_mentions_stats,
        )

    results_output_path = comparisons_folder_path / "cv_metrics_summary.json"
    with open(results_output_path, "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"Evaluation complete. Results saved to {results_output_path}")
    return results
