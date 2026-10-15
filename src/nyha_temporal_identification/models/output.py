from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from nyha_temporal_identification.models.constants import MI_DROP_THRESHOLD


def plot_feature_importance(
    mi_scores: dict[str, float],
    gb_model: GradientBoostingClassifier | None,
    selected_features: list[str],
    has_text: bool,
    save_dir: Path,
) -> None:
    """
    Save a bar chart of MI scores.

    GB feature importances are only plotted when use_text=False because with
    BoW features the importance indices no longer map to named columns cleanly.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    show_gb = (gb_model is not None) and (not has_text)
    ncols = 2 if show_gb else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols + 1, 4))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.barh(list(mi_scores.keys()), list(mi_scores.values()), color="steelblue")
    ax.axvline(
        MI_DROP_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"drop threshold = {MI_DROP_THRESHOLD}",
    )
    ax.set_title("Mutual Information (numeric features)")
    ax.set_xlabel("MI score")
    ax.legend(fontsize=8)

    if show_gb and gb_model is not None:
        axes[1].barh(
            selected_features, gb_model.feature_importances_, color="darkorange"
        )
        axes[1].set_title("Gradient Boosting Importances")
        axes[1].set_xlabel("Importance")

    fig.tight_layout()
    out = save_dir / "feature_importance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] Feature importance saved -> {out}")


def compare_results(results: dict[str, dict[str, float]]) -> None:
    """Print a tidy aligned comparison table across all evaluated models."""
    print(f"\n{'=' * 70}")
    print("  FINAL COMPARISON  (test set)")
    print(f"{'=' * 70}")
    print(f"  {'Model':<32} {'micro-F1':>10} {'wgt-F1':>10} {'#folds':>6}")
    print(f"  {'-' * 62}")
    for name, m in results.items():
        print(
            f"  {name:<32} {m['merged_micro_f1']:>10.4f}"
            f" {m['fold_micro_f1_weighted_mean']:>10.4f}"
            f" {int(m['n_folds']):>6}"
        )
    print(f"{'=' * 70}\n")
