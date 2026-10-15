import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_selection import mutual_info_classif

from nyha_temporal_identification.models.constants import (
    MI_DROP_THRESHOLD,
    RANDOM_STATE,
    TARGET_COL,
)


def to_numpy_features(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """
    Extract columns from a Polars DataFrame as a dense 2-D NumPy array.
    This is the primary Polars → sklearn bridge for numeric features.
    """
    return df.select(cols).to_numpy()


def build_feature_matrix(
    df: pl.DataFrame,
    numeric_features: list[str],
    bow_matrix: sp.csr_matrix | None,
) -> np.ndarray | sp.csr_matrix:
    """
    Assemble the final feature matrix by horizontally stacking:
      [dense numeric block] | [sparse BoW block (optional)]

    When bow_matrix is None (use_text=False), returns a plain dense ndarray.
    When bow_matrix is provided, returns a scipy.sparse.csr_matrix so that
    memory-efficient sparse operations are preserved end-to-end.

    All sklearn estimators in this pipeline (LR, GBM, ThresholdClassifier)
    accept both dense and sparse input without modification.

    Parameters
    ----------
    df              : pl.DataFrame       Source split.
    numeric_features: list[str]          Numeric columns to include.
    bow_matrix      : csr_matrix | None  Pre-computed BoW block (same row order).

    Returns
    -------
    np.ndarray or scipy.sparse.csr_matrix
    """
    if bow_matrix is None:
        if not numeric_features:
            raise ValueError(
                "build_feature_matrix: numeric_features is empty and bow_matrix is None — "
                "no features available to build the matrix."
            )
        return to_numpy_features(df, numeric_features)

    if not numeric_features:
        return bow_matrix

    X_num_sparse = sp.csr_matrix(to_numpy_features(df, numeric_features))
    return sp.csr_matrix(sp.hstack([X_num_sparse, bow_matrix], format="csr"))


def analyse_features(
    train_df: pl.DataFrame,
    features: list[str],
) -> tuple[list[str], dict[str, float]]:
    """
    Compute mutual information between each NUMERIC feature and the target.
    Features scoring below MI_DROP_THRESHOLD are dropped.

    Note: MI analysis runs on numeric features only. BoW features are added
    AFTER this step and are not individually filtered by MI (see §3 comments
    in the module docstring for rationale).

    Parameters
    ----------
    train_df : pl.DataFrame
    features : list[str]   Candidate numeric feature columns.

    Returns
    -------
    selected_features : list[str]
    mi_scores         : dict[str, float]
    """
    if not features:
        return [], {}

    X_tr = to_numpy_features(train_df, features)
    y_tr = train_df[TARGET_COL].to_numpy()

    mi = mutual_info_classif(X_tr, y_tr, random_state=RANDOM_STATE)
    mi_scores = dict(zip(features, mi))

    print("\n[feature_analysis] Mutual Information scores (numeric features):")
    for feat, score in mi_scores.items():
        flag = "  <- DROP (MI < threshold)" if score < MI_DROP_THRESHOLD else ""
        print(f"  {feat:30s}: {score:.4f}{flag}")

    selected = [f for f, s in mi_scores.items() if s >= MI_DROP_THRESHOLD]
    dropped = [f for f in features if f not in selected]
    if dropped:
        print(f"[feature_analysis] Dropped: {dropped}")
    else:
        print("[feature_analysis] All numeric features retained.")

    return selected, mi_scores
