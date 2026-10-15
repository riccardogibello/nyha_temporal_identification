import itertools
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nyha_temporal_identification.models.constants import (
    RANDOM_STATE,
    TARGET_COL,
    THRESHOLD_GRID_STEPS,
)
from nyha_temporal_identification.models.features_handling import to_numpy_features


def build_gradient_boosting(
    X_train: np.ndarray | sp.csr_matrix,
    y_train: np.ndarray,
    X_val: np.ndarray | sp.csr_matrix,
    y_val: np.ndarray,
) -> GradientBoostingClassifier:
    """
    Fit a GradientBoostingClassifier with CV-tuned hyperparameters.

    Accepts both dense and sparse feature matrices.
    Converts sparse to dense internally if sklearn version < 1.0 requires it.
    """
    if sp.issparse(X_train):
        X_train_sparse = cast(sp.csr_matrix, X_train)
        X_val_sparse = cast(sp.csr_matrix, X_val)
        X_tv = sp.csr_matrix(sp.vstack([X_train_sparse, X_val_sparse]))
    else:
        X_train_dense = cast(np.ndarray, X_train)
        X_val_dense = cast(np.ndarray, X_val)
        X_tv = np.vstack([X_train_dense, X_val_dense])
    y_tv = np.concatenate([y_train, y_val])

    # GradientBoostingClassifier requires dense input
    if sp.issparse(X_tv):
        X_tv_sparse = cast(sp.csr_matrix, X_tv)
        X_tv = X_tv_sparse.toarray()

    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [2, 3],
        "subsample": [0.8, 1.0],
    }
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    gs.fit(X_tv, y_tv)

    print(
        f"[gradient_boosting] Best params = {gs.best_params_}  "
        f"(CV macro-F1 = {gs.best_score_:.4f})"
    )
    return gs.best_estimator_


def build_logistic_regression(
    X_train: np.ndarray | sp.csr_matrix,
    y_train: np.ndarray,
    X_val: np.ndarray | sp.csr_matrix,
    y_val: np.ndarray,
    use_text: bool,
) -> Pipeline:
    """
    Fit a StandardScaler → LogisticRegression pipeline with CV-tuned C.

    When use_text=True, L1 regularisation + 'saga' solver is used because:
      - L1 performs built-in feature selection on the high-dimensional BoW space.
      - 'saga' is the only solver that supports L1 + sparse input efficiently.
    When use_text=False, L2 + 'lbfgs' (default) is used (better for small dense).

    The pipeline's StandardScaler is sparse-aware (with_mean=False when input
    is sparse, to avoid densifying the matrix).

    Parameters
    ----------
    X_train, X_val : dense ndarray or sparse matrix
    y_train, y_val : 1-D int ndarray
    use_text       : bool   Switches solver / penalty.
    """
    if sp.issparse(X_train):
        X_train_sparse = cast(sp.csr_matrix, X_train)
        X_val_sparse = cast(sp.csr_matrix, X_val)
        X_tv = sp.csr_matrix(sp.vstack([X_train_sparse, X_val_sparse]))
    else:
        X_train_dense = cast(np.ndarray, X_train)
        X_val_dense = cast(np.ndarray, X_val)
        X_tv = np.vstack([X_train_dense, X_val_dense])
    y_tv = np.concatenate([y_train, y_val])

    if use_text:
        # Sparse-safe scaler (with_mean=False avoids densifying the matrix)
        scaler = StandardScaler(with_mean=False)
        lr = LogisticRegression(
            penalty="l1", solver="saga", max_iter=2000, random_state=RANDOM_STATE
        )
        param_grid = {"lr__C": [0.001, 0.01, 0.1, 1, 10]}
    else:
        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        param_grid = {"lr__C": [0.001, 0.01, 0.1, 1, 10, 100]}

    pipe = Pipeline([("scaler", scaler), ("lr", lr)])
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_tv, y_tv)

    print(
        f"[logistic_regression] Best C = {gs.best_params_['lr__C']}  "
        f"(CV macro-F1 = {gs.best_score_:.4f})"
    )
    return gs.best_estimator_


# ── Threshold-based classifier ───────────────────────────────────────────
#
# Metric: macro-F1 on validation set (equal class weight, robust to imbalance).
#
# When use_text=True, the ThresholdClassifier operates ONLY on the numeric
# features stored in its thresholds_ dict.  It extracts just those columns
# from whichever matrix is passed in — BoW columns are simply ignored.
# This is the correct design: threshold rules on binary word-presence
# columns are semantically meaningless and computationally intractable.
class ThresholdClassifier:
    """
    Rule-based classifier: predict POSITIVE_CLASS when
        feature_1 < t_1  AND … AND  feature_N < t_N
    applied to NUMERIC features only.

    Attributes
    ----------
    thresholds_  : dict[str, float]   Per-feature optimal thresholds.
    best_val_f1_ : float              Val macro-F1 at chosen thresholds.
    """

    def __init__(self) -> None:
        self.thresholds_: dict[str, float] = {}
        self.best_val_f1_: float = -1.0

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        numeric_features: list[str],
    ) -> "ThresholdClassifier":
        """
        Learn one threshold per numeric feature by brute-force search.

        The method works in three steps:
        1. Build a small candidate list of threshold values for each numeric
           feature using percentiles from the training split.
        2. Try every combination of those candidate thresholds.
        3. Score each combination on the validation split with macro-F1 and keep
           the best one.

        The rule being tuned is:
            predict 1 if feature_1 < t_1 and feature_2 < t_2 and ...
            otherwise predict 0.

        Only numeric features are used here. Text / BoW columns are ignored.

        Parameters
        ----------
        train_df : pl.DataFrame
            Training split used to generate candidate thresholds.
        val_df : pl.DataFrame
            Validation split used to choose the best threshold combination.
        numeric_features : list[str]
            Numeric columns to threshold on.

        Returns
        -------
        ThresholdClassifier
            The fitted classifier, with thresholds_ and best_val_f1_ populated.
        """
        percentiles = np.linspace(5, 95, THRESHOLD_GRID_STEPS)
        candidates: dict[str, np.ndarray] = {
            feat: np.unique(np.percentile(train_df[feat].to_numpy(), percentiles))
            for feat in numeric_features
        }

        X_val = to_numpy_features(val_df, numeric_features)
        y_val = val_df[TARGET_COL].to_numpy()

        best_f1 = -1.0
        best_combo: tuple[float, ...] = tuple(
            candidates[f][0] for f in numeric_features
        )

        for combo in itertools.product(*[candidates[f] for f in numeric_features]):
            mask = np.ones(len(X_val), dtype=bool)
            for j, t in enumerate(combo):
                mask &= X_val[:, j] < t
            score = f1_score(y_val, mask.astype(int), average="macro", zero_division=0)
            if score > best_f1:
                best_f1, best_combo = score, combo

        self.thresholds_ = dict(zip(numeric_features, best_combo))
        self.best_val_f1_ = float(best_f1)
        print(
            "[threshold] Tuned thresholds (optimised by val macro-F1):\n"
            + "\n".join(f"  {f:30s} < {t:.4f}" for f, t in self.thresholds_.items())
            + f"\n  Val macro-F1 = {best_f1:.4f}"
        )
        return self

    def predict(
        self, X: np.ndarray | sp.csr_matrix, numeric_indices: list[int]
    ) -> np.ndarray:
        """
        Predict from a combined feature matrix.

        Parameters
        ----------
        X               : dense ndarray or sparse matrix   Full feature matrix.
        numeric_indices : list[int]   Column positions of the numeric features
                          within X.  Only those columns are used; BoW columns
                          are ignored.
        """
        if not self.thresholds_:
            raise RuntimeError("Call fit() before predict().")

        # Extract just the numeric columns (works for both dense and sparse)
        if sp.issparse(X):
            X_sparse = cast(sp.csr_matrix, X)
            X_num = X_sparse[:, numeric_indices].toarray()
        else:
            # Ensure X is a numpy array for type safety, then slice
            X_dense = cast(np.ndarray, X)
            X_num = X_dense[:, numeric_indices]

        mask = np.ones(X_num.shape[0], dtype=bool)
        for j in range(len(self.thresholds_)):
            t = list(self.thresholds_.values())[j]
            mask &= X_num[:, j] < t
        return mask.astype(int)

    def plot_threshold_search(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        numeric_features: list[str],
        save_dir: Path,
    ) -> None:
        """
        Plot how validation macro-F1 changes across threshold candidates.

        This visualizes the same brute-force search performed in `fit()`, but
        in a human-readable form so you can inspect how sensitive the rule is
        to each threshold choice.

        How to read the image:
        - 1 numeric feature: the x-axis is the candidate threshold and the
            y-axis is validation macro-F1. The red dashed line marks the chosen
            threshold.
        - 2 numeric features: the axes are the two thresholds. Brighter colors
            mean higher validation macro-F1. The blue marker shows the selected
            optimum.
        - 3+ numeric features: each subplot shows one feature on the x-axis
            while all other features are held fixed at their fitted thresholds.
            The red dashed line is the chosen threshold for that feature.

        In all cases, the plot only uses numeric features. Text / BoW columns
        are not included.

        The figure is saved as `threshold_search.png` inside `save_dir`.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        percentiles = np.linspace(5, 95, THRESHOLD_GRID_STEPS)
        candidates: dict[str, np.ndarray] = {
            f: np.unique(np.percentile(train_df[f].to_numpy(), percentiles))
            for f in numeric_features
        }
        X_val = to_numpy_features(val_df, numeric_features)
        y_val = val_df[TARGET_COL].to_numpy()

        if len(numeric_features) == 1:
            feat = numeric_features[0]
            ts = candidates[feat]
            f1s = [
                f1_score(
                    y_val,
                    (X_val[:, 0] < t).astype(int),
                    average="macro",
                    zero_division=0,
                )
                for t in ts
            ]
            fig, ax = plt.subplots(figsize=(6, 4))
            f1s_float = [float(np.asarray(f)) for f in f1s]
            ax.plot(ts, f1s_float, marker="o", color="steelblue")
            ax.axvline(
                self.thresholds_[feat],
                color="red",
                linestyle="--",
                label=f"best t = {self.thresholds_[feat]:.3f}",
            )
            ax.set_xlabel(feat)
            ax.set_ylabel("Val macro-F1")
            ax.set_title("Threshold search (1-D)")
            ax.legend()

        elif len(numeric_features) == 2:
            f0, f1n = numeric_features[0], numeric_features[1]
            ts0, ts1 = candidates[f0], candidates[f1n]
            grid = np.zeros((len(ts1), len(ts0)))
            for i, t1 in enumerate(ts1):
                for j, t0 in enumerate(ts0):
                    preds = ((X_val[:, 0] < t0) & (X_val[:, 1] < t1)).astype(int)
                    grid[i, j] = f1_score(
                        y_val, preds, average="macro", zero_division=0
                    )

            best_row, best_col = np.unravel_index(np.argmax(grid), grid.shape)
            best_t0 = ts0[best_col]
            best_t1 = ts1[best_row]
            best_score = grid[best_row, best_col]

            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                extent=(ts0.min(), ts0.max(), ts1.min(), ts1.max()),
                cmap="YlOrRd",
            )
            plt.colorbar(im, ax=ax, label="Val macro-F1")
            ax.scatter(
                best_t0,
                best_t1,
                color="white",
                edgecolors="black",
                marker="*",
                s=220,
                zorder=6,
                label=f"best grid cell ({best_t0:.2f}, {best_t1:.2f}) = {best_score:.3f}",
            )
            ax.scatter(
                self.thresholds_[f0],
                self.thresholds_[f1n],
                color="blue",
                s=120,
                zorder=5,
                label=f"fitted thresholds ({self.thresholds_[f0]:.2f}, {self.thresholds_[f1n]:.2f})",
            )
            ax.set_xlabel(f0)
            ax.set_ylabel(f1n)
            ax.set_title("Threshold search: 2-D heatmap (val macro-F1)")
            ax.legend(fontsize=8)

        else:
            n = len(numeric_features)
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
            axes = np.atleast_1d(axes)
            for idx, feat in enumerate(numeric_features):
                ts = candidates[feat]
                f1s = []
                for t in ts:
                    mask = X_val[:, idx] < t
                    for oidx, of in enumerate(numeric_features):
                        if of != feat:
                            mask &= X_val[:, oidx] < self.thresholds_[of]
                    f1s.append(
                        f1_score(
                            y_val, mask.astype(int), average="macro", zero_division=0
                        )
                    )
                axes[idx].plot(ts, f1s, marker="o", color="steelblue")
                axes[idx].axvline(
                    self.thresholds_[feat],
                    color="red",
                    linestyle="--",
                    label=f"best = {self.thresholds_[feat]:.3f}",
                )
                axes[idx].set_xlabel(feat)
                axes[idx].set_ylabel("Val macro-F1" if idx == 0 else "")
                axes[idx].set_title(f"Marginal: {feat}")
                axes[idx].legend(fontsize=7)
            fig.suptitle("Threshold search: marginal plots (others fixed at optimum)")

        fig.tight_layout()
        out = save_dir / "threshold_search.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[plot] Threshold search saved -> {out}")
