import logging
import random
from typing import List

import dspy

from nyha_temporal_identification.constants import (
    AUTHORED_DATETIME_COLUMN,
    DEIDENTIFIED_TEXT_COLUMN,
)
from nyha_temporal_identification.data_model.dspy_model import (
    NYHAMention,
    NYHANote,
)
from nyha_temporal_identification.dspy_logic.regex import CLASSIFICATION_CONTEXT_WINDOW
from nyha_temporal_identification.dspy_logic.utils import (
    SPAN_MARKER,
    build_classification_context,
    truncate_to_token_budget,
)
from nyha_temporal_identification.utils.general import Config

logger = logging.getLogger(__name__)


def documents_to_dspy_examples(
    documents: List[NYHANote],
    config: Config,
    tokenizer=None,
    text_token_budget: int = 0,
    explode: bool | None = None,
) -> List[dspy.Example]:
    """
    Convert NYHANote list to normalised DSPy Example objects.

    When tokenizer is provided and text_token_budget > 0, each example's text
    is capped to text_token_budget tokens. For the windowed path the cap is
    applied inside normalise_examples (after rebasing); for the full-note path
    it is applied here (tail truncation, no markers).

    ``explode`` controls whether each note is split into one example per mention
    (windowed single-span examples) or kept as a single document-level example
    carrying all its gold mentions:

    - ``explode=True``  → per-mention windowed examples (demo pool / trainset).
    - ``explode=False`` → document-level examples (optimiser valset + reporting,
      so candidate ranking and reports run on whole notes exactly as inference
      does — the extractor windows internally per span).
    - ``explode=None``  → defaults to ``config.use_windowed_classification_context``
      (preserves the historical behaviour).
    """
    if explode is None:
        explode = config.use_windowed_classification_context
    raw = []
    for doc in documents:
        text = doc.deidentified_text
        if tokenizer is not None and text_token_budget > 0 and not config.use_windowed_classification_context:
            text = truncate_to_token_budget(text, tokenizer, text_token_budget)
        raw.append(
            dspy.Example(
                deidentified_text=text,
                authored_datetime=doc.authored_datetime,
                mentions=doc.mentions,
            ).with_inputs(
                DEIDENTIFIED_TEXT_COLUMN,
                AUTHORED_DATETIME_COLUMN,
            )
        )

    if explode:
        return normalise_examples(raw, tokenizer=tokenizer, text_token_budget=text_token_budget)
    return raw


def subsample_examples(
    examples: List[dspy.Example],
    max_size: int,
    min_per_class: int = 0,
    seed: int = 0,
) -> List[dspy.Example]:
    """
    Cap examples to at most max_size while guaranteeing a per-class floor.

    No-op when max_size <= 0 or len(examples) <= max_size. When capping:
    1. Reserve up to min_per_class examples per nyha_class (rare-class floor).
    2. Fill remaining budget by proportional random sampling across classes.
    3. Return the union, order-shuffled under seed.

    Works on both windowed (per-mention, single nyha_class) and note-level
    examples (mentions list). For note-level examples the class is taken from
    the first mention; examples with no mentions go into an "other" bucket.
    """
    if max_size <= 0 or len(examples) <= max_size:
        return examples

    rng = random.Random(seed)

    def _class_key(ex: dspy.Example) -> str:
        if hasattr(ex, "nyha_class") and ex.nyha_class:
            return str(ex.nyha_class)
        mentions = getattr(ex, "mentions", [])
        if mentions:
            return str(getattr(mentions[0], "nyha_class", "other"))
        return "other"

    # Group by class
    by_class: dict[str, list[dspy.Example]] = {}
    for ex in examples:
        key = _class_key(ex)
        by_class.setdefault(key, []).append(ex)

    # Reserve floor
    reserved: list[dspy.Example] = []
    remainder_by_class: dict[str, list[dspy.Example]] = {}
    for cls, cls_examples in by_class.items():
        shuffled = cls_examples[:]
        rng.shuffle(shuffled)
        floor = min(min_per_class, len(shuffled))
        reserved.extend(shuffled[:floor])
        remainder_by_class[cls] = shuffled[floor:]

    remaining_budget = max_size - len(reserved)
    if remaining_budget <= 0:
        rng.shuffle(reserved)
        return reserved[:max_size]

    # Proportional sampling from remainder
    all_remaining = [ex for exs in remainder_by_class.values() for ex in exs]
    rng.shuffle(all_remaining)
    sampled = all_remaining[:remaining_budget]

    result = reserved + sampled
    rng.shuffle(result)
    return result


def normalise_examples(
    examples: List[dspy.Example],
    window_chars: int = CLASSIFICATION_CONTEXT_WINDOW,
    tokenizer=None,
    text_token_budget: int = 0,
) -> List[dspy.Example]:
    """
    Explode a list of dspy.Examples into one example per mention.

    Each output example's deidentified_text is the canonical classifier input
    produced by build_classification_context: a character window around the
    span, with >>> / <<< markers injected, then token-capped symmetrically
    around the marker pair. This is byte-identical to what NYHAExtractor.forward
    feeds the classifier at inference time, eliminating the previous train/serve
    mismatch where demos were unmarked.

    Span offsets in the rebased mention are located by searching for the marker
    in the context string rather than by arithmetic rebasing, so they remain
    valid after the symmetric token cap. linked_date is dropped (None) because
    its offsets are invalid in the marker-aware coordinate system and the classify
    signature does not consume it.
    """
    normalised: List[dspy.Example] = []
    n_exploded = 0
    n_skipped = 0

    for ex in examples:
        full_text: str = getattr(ex, DEIDENTIFIED_TEXT_COLUMN, "")
        authored_datetime: str = getattr(ex, AUTHORED_DATETIME_COLUMN, "")
        mentions: List[NYHAMention] = getattr(ex, "mentions", [])

        if not full_text:
            logger.warning(
                "Example has empty deidentified_text — skipping. "
                "(authored_datetime=%s)",
                authored_datetime,
            )
            n_skipped += 1
            continue

        if not mentions:
            logger.warning(
                "Example has no mentions — skipping. (authored_datetime=%s)",
                authored_datetime,
            )
            n_skipped += 1
            continue

        for mention in mentions:
            if not mention.validate_against_text(full_text):
                logger.warning(
                    "Mention '%s' (%d, %d) failed validation against full text — skipping.",
                    mention.span,
                    mention.char_start,
                    mention.char_end,
                )
                n_skipped += 1
                continue

            context = build_classification_context(
                full_text,
                mention.char_start,
                mention.char_end,
                window=window_chars,
                tokenizer=tokenizer,
                max_tokens=text_token_budget,
            )

            marker_pos = context.find(SPAN_MARKER)
            if marker_pos == -1:
                logger.warning(
                    "Marker absent in context for '%s' — skipping.", mention.span
                )
                n_skipped += 1
                continue
            marked_start = marker_pos + len(SPAN_MARKER)
            marked_end = marked_start + len(mention.span)
            if context[marked_start:marked_end].strip().lower() != mention.span.strip().lower():
                logger.warning(
                    "Context span mismatch for '%s' (got '%s') — skipping.",
                    mention.span,
                    context[marked_start:marked_end],
                )
                n_skipped += 1
                continue

            rebased = mention.model_copy(
                update={
                    "char_start": marked_start,
                    "char_end": marked_end,
                    "linked_date": None,
                }
            )

            normalised.append(
                dspy.Example(
                    deidentified_text=context,
                    authored_datetime=authored_datetime,
                    span=rebased.span,
                    nyha_class=rebased.nyha_class,
                    temporality=rebased.temporality,
                    evidence=rebased.evidence,
                    mentions=[rebased],
                ).with_inputs(
                    DEIDENTIFIED_TEXT_COLUMN,
                    AUTHORED_DATETIME_COLUMN,
                )
            )
            n_exploded += 1

    logger.info(
        "normalise_examples: %d mentions exploded, %d skipped. "
        "Total output: %d examples.",
        n_exploded,
        n_skipped,
        len(normalised),
    )
    return normalised
