import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from nyha_temporal_identification.models.constants import (
    NYHA_PRED_END_INDEX,
    NYHA_PRED_START_INDEX,
    POSITIVE_CLASS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    TEXT_COL,
    VAL_SIZE,
)


def encode_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Encode TARGET_COL: POSITIVE_CLASS → 1, other class → 0 (Int32).

    Uses a Polars when/then/otherwise expression; no sklearn LabelEncoder needed.
    """
    other_class = next(c for c in ["historical", "current"] if c != POSITIVE_CLASS)
    df = df.with_columns(
        pl.when(pl.col(TARGET_COL) == POSITIVE_CLASS)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int32)
        .alias(TARGET_COL)
    )
    print(f"[preprocess] Label mapping: {{'{other_class}': 0, '{POSITIVE_CLASS}': 1}}")
    return df


def _stratified_split_indices(
    labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (keep_idx, held_idx) for a stratified split on row indices."""
    idx = np.arange(len(labels))
    keep_idx, held_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return keep_idx, held_idx


def split_dataset(
    df: pl.DataFrame,
    features: list[str],
    use_text: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Stratified three-way split: train / validation / test.

    When use_text=True the text columns are retained in all splits so that
    TextFeatureExtractor can process them downstream.

    Parameters
    ----------
    df       : pl.DataFrame   Encoded full dataset.
    features : list[str]      Numeric feature columns.
    use_text : bool           Whether to retain text columns in splits.

    Returns
    -------
    train_df, val_df, test_df, test_invalid_df : pl.DataFrame
    """
    keep_cols = features + [TARGET_COL]
    if use_text:
        keep_cols += [TEXT_COL, NYHA_PRED_START_INDEX, NYHA_PRED_END_INDEX]
    keep_cols = list(dict.fromkeys(keep_cols))  # deduplicate, preserve order

    df_slim = df.select(keep_cols)
    labels = df_slim[TARGET_COL].to_numpy()

    trainval_idx, test_idx = _stratified_split_indices(labels, TEST_SIZE, RANDOM_STATE)
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    train_idx, val_idx = _stratified_split_indices(
        labels[trainval_idx], val_frac, RANDOM_STATE
    )
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    df_idx = df_slim.with_row_index("_row_idx")
    train_df = df_idx.filter(pl.col("_row_idx").is_in(train_idx)).drop("_row_idx")
    val_df = df_idx.filter(pl.col("_row_idx").is_in(val_idx)).drop("_row_idx")
    test_df = df_idx.filter(pl.col("_row_idx").is_in(test_idx)).drop("_row_idx")

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pos = part[TARGET_COL].mean()
        print(f"[split] {name:5s}: {len(part):4d} rows | positive rate = {pos:.2%}")

    return train_df, val_df, test_df
