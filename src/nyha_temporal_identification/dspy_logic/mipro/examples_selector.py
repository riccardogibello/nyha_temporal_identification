import logging
from collections import Counter

import dspy

logger = logging.getLogger(__name__)


def reorder_by_rarity(
    examples: list[dspy.Example],
) -> list[dspy.Example]:
    """
    Reorder normalised examples (one per mention) so those with the rarest
    NYHA class × temporality combination come first.

    DSPy's GroundedProposer reads the first view_data_batch_size examples
    for the dataset summary, so placing rare-combo examples first ensures
    the proposal stage sees the most informative cases without restricting
    the bootstrapping pool — the full reordered set remains available.

    Args:
        examples: normalised flat examples from documents_to_dspy_examples().
                  Each must have nyha_class and temporality as top-level fields.

    Raises:
        ValueError: if any example is missing nyha_class or temporality,
                    which indicates normalise_examples() was not called upstream.
    """
    if not examples:
        return []

    # ── Invariant check ───────────────────────────────────────────────────────
    missing = [
        i
        for i, ex in enumerate(examples)
        if getattr(ex, "nyha_class", None) is None
        or getattr(ex, "temporality", None) is None
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} examples are missing nyha_class or temporality "
            f"(indices: {missing[:10]}{'...' if len(missing) > 10 else ''}) — "
            "was normalise_examples() called before reorder_by_rarity()?"
        )

    # ── Global combination counts ─────────────────────────────────────────────
    combo_counts: Counter = Counter((ex.nyha_class, ex.temporality) for ex in examples)
    logger.info(
        "[MIPRO] Combination counts across full trainset:\n%s",
        "\n".join(
            f"  {cls}×{temp}: {cnt}"
            for (cls, temp), cnt in sorted(combo_counts.items())
        ),
    )

    # ── Dynamic smoothing anchored to rarest observed combo ───────────────────
    # smooth = 0.5 × min_count so dampening scales with the actual population:
    # a combo seen once gets weight 1/1.5, seen twice 1/2.5, etc.
    min_count = min(combo_counts.values())
    smooth = min_count * 0.5

    # ── Sort by descending weight ─────────────────────────────────────────────
    # Higher weight = rarer combo = appears earlier in the list.
    reordered = sorted(
        examples,
        key=lambda ex: 1.0 / (combo_counts[(ex.nyha_class, ex.temporality)] + smooth),
        reverse=True,
    )

    return reordered
