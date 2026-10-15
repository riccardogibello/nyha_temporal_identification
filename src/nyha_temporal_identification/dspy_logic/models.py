"""
models.py — Core data models for NYHA extraction pipeline.

Extended to incorporate date-linkage from nyha_parsing.py:
- NYHAMention now carries linked_date and char_distance fields
- resolve_indices() accepts pre-extracted spaCy spans as an alternative
  to pure LLM text output
- Temporality inference incorporates date-distance heuristics
"""

from __future__ import annotations

from itertools import groupby
from typing import List, Optional

from nyha_temporal_identification.data_model.dspy_model import (
    DspyDate,
    NYHAMention,
    NYHAMentionRaw,
)
from nyha_temporal_identification.data_model.general import NyhaMention
from nyha_temporal_identification.evaluation.evaluation import OVERLAP_THRESHOLD

# ── Temporality inference ─────────────────────────────────────────────────────

# Heuristic: if the note's authored date is known and the linked date is more
# than this many days in the past, treat as historical regardless of anchor words.
HISTORICAL_DATE_LAG_DAYS = 30

# Character distance thresholds (from nyha_parsing.py logic)
# If char_distance < this value AND no historical anchor → lean current
CURRENT_DISTANCE_THRESHOLD = 50
# If char_distance > this → lean historical regardless of anchor words
HISTORICAL_DISTANCE_THRESHOLD = 200


def resolve_indices(
    deidentified_text: str,
    raw_mentions: List[NYHAMentionRaw],
) -> List[NYHAMention]:
    """
    Convert LLM span-text output into character-indexed NYHAMention objects.

    The LLM produces NYHAMentionRaw objects that identify mentions by their
    surface text (e.g. "NYHA III") and an occurrence index (1 = first occurrence,
    2 = second, etc.) rather than by character position. This function resolves
    those surface-text references into exact (char_start, char_end) indices by
    scanning the deidentified text.

    Resolution strategy
    -------------------
    A naive substring search (str.find) is ambiguous when one span is a prefix
    of another (e.g. "NYHA I" vs "NYHA II"): the shorter span would incorrectly
    match inside the longer one. This is avoided by processing spans in
    descending length order and maintaining a masked working copy of the text:

      1. Sort all mentions by descending span length so longer spans are
         resolved first. Within the same span text, sort by ascending occurrence
         so relative ordering is preserved.
      2. Group mentions that share the same normalised span text. For each
         group, collect ALL occurrence positions from the current masked text
         before blanking any of them. This ensures occurrence numbers remain
         consistent even when multiple mentions share the same span.
      3. For each mention in the group, look up its position by occurrence index
         directly from the pre-collected list.
      4. After resolving a mention, blank its region in the masked text with
         spaces (preserving indices). Subsequent shorter-span searches will
         therefore never match inside an already-resolved region.
      5. Construct an NYHAMention with the resolved indices. Skip with a
         warning if Pydantic validation fails.
      6. Call validate_against_text() against the original (unmasked) text as
         a final sanity check. Skip with a warning if it fails.
      7. Restore the original raw_mentions ordering in the returned list.

    Note: date linkage fields (linked_date, char_distance) are intentionally
    left empty here. They are populated separately by merge_with_spacy_spans()
    after spaCy pre-processing has identified date spans in the text.

    Args:
        deidentified_text: The full clinical note as a string, used both for
                           span search and for index validation.
        raw_mentions:      List of NYHAMentionRaw objects produced by the LLM,
                           each carrying a span string, occurrence index, NYHA
                           class, temporality label, and evidence phrase.

    Returns:
        List of successfully resolved NYHAMention objects, in the same order
        as raw_mentions. Mentions that cannot be found, fail construction, or
        fail validation are silently dropped (with a warning printed) rather
        than raising an exception, so a single bad LLM output does not abort
        the whole document.
    """
    resolved_map: dict[int, NYHAMention] = {}  # original index → mention

    # Sort by descending span length so longer spans are resolved first,
    # preventing shorter substrings from accidentally matching inside them.
    # Secondary sort by normalised span text groups identical spans together
    # (required by itertools.groupby). Tertiary sort by occurrence preserves
    # relative ordering within each group.
    sorted_mentions = sorted(
        enumerate(raw_mentions),
        key=lambda x: (
            -len(x[1].span.strip()),
            x[1].span.strip().lower(),
            x[1].occurrence,
        ),
    )

    # Maintain a masked lowercased copy of the text. As each span is resolved,
    # its region is blanked with spaces so that no subsequent shorter span can
    # match there. Blanking with spaces (rather than, e.g., null bytes) is safe
    # because it preserves all character indices relative to the original text.
    masked_text = deidentified_text.lower()

    for _, group in groupby(sorted_mentions, key=lambda x: x[1].span.strip().lower()):
        group_list = sorted(group, key=lambda x: x[1].occurrence)
        span_key = group_list[0][1].span.strip().lower()

        # Collect ALL positions of this span in the current masked text before
        # blanking any of them. This is essential for correctness when multiple
        # mentions share the same span text: if we blanked after each resolution,
        # the occurrence numbers for subsequent mentions in the group would shift.
        all_positions: List[int] = []
        search_from = 0
        while True:
            idx = masked_text.find(span_key, search_from)
            if idx == -1:
                break
            all_positions.append(idx)
            search_from = idx + 1

        for original_idx, m in group_list:
            target_occ = m.occurrence

            if target_occ > len(all_positions):
                print(
                    f"[WARN] Could not find span '{m.span}' "
                    f"(occurrence {m.occurrence}) in text — skipping."
                )
                continue

            # Occurrence is 1-based, so index directly into the pre-collected list.
            found_start = all_positions[target_occ - 1]
            found_end = found_start + len(span_key)

            # Blank the resolved region in the masked text so that no subsequent
            # shorter span (e.g. "NYHA I" after "NYHA II" is resolved) can match
            # inside it. Length is preserved so all other indices remain valid.
            masked_text = (
                masked_text[:found_start]
                + " " * (found_end - found_start)
                + masked_text[found_end:]
            )

            # Construct the validated mention — Pydantic may reject it if indices
            # are inconsistent (e.g. end <= start, out-of-bounds).
            try:
                mention = NYHAMention(
                    char_start=found_start,
                    char_end=found_end,
                    span=deidentified_text[found_start:found_end],
                    nyha_class=m.nyha_class,
                    temporality=m.temporality,
                    evidence=m.evidence,
                )
            except Exception as e:
                print(
                    f"[WARN] Failed to construct NYHAMention for span '{m.span}' "
                    f"at ({found_start}, {found_end}): {e} — skipping."
                )
                continue

            # Final sanity check: confirm the resolved indices reproduce the span
            # text exactly (guards against off-by-one errors or encoding issues).
            if mention.validate_against_text(deidentified_text):
                resolved_map[original_idx] = mention
            else:
                print(
                    f"[WARN] Index validation failed for '{mention.span}' "
                    f"at ({mention.char_start}, {mention.char_end}) — skipping."
                )

    # Restore the original ordering from raw_mentions before returning.
    return [resolved_map[i] for i in range(len(raw_mentions)) if i in resolved_map]


def merge_with_spacy_spans(
    llm_mentions: List[NYHAMention],
    spacy_mentions: List[NyhaMention],
    deidentified_text: str,
) -> List[NYHAMention]:
    """
    Enrich LLM-resolved mentions with date-linkage data from spaCy pre-processing.

    Responsibility: date-linkage only. This function copies linked_date and
    char_distance from the best-matching spaCy mention onto each LLM mention.
    It does NOT perform any temporality inference or override the LLM's labels.

    Matching: for each LLM mention, the spaCy mention with the highest span
    overlap above OVERLAP_THRESHOLD is selected. If no match is found, the
    mention is returned unchanged.

    Args:
        llm_mentions:      NYHAMention objects resolved from LLM output.
        spacy_mentions:    NyhaMention objects from spaCy pre-processing, each
                           potentially carrying a linked_date and char_distance.
        deidentified_text: full clinical note, used for overlap calculation.

    Returns:
        List of NYHAMention objects in the same order as llm_mentions, with
        linked_date and char_distance fields populated where a spaCy match
        was found. Temporality and evidence fields are never modified here.
    """
    enriched: List[NYHAMention] = []

    for llm_m in llm_mentions:
        # --- Step 1: find the best-overlapping spaCy mention ---
        best_spacy = None
        best_overlap = 0.0

        for sp in spacy_mentions:
            # Prefer original (pre-cleaning) indices; fall back to cleaned indices
            sp_start = sp.original_start if sp.original_start is not None else sp.start
            sp_end = sp.original_end if sp.original_end is not None else sp.end
            if sp_start < 0 or sp_end < 0:
                continue
            try:
                # Build a minimal temporary mention just for overlap calculation
                tmp = NYHAMention(
                    char_start=sp_start,
                    char_end=sp_end,
                    span=deidentified_text[sp_start:sp_end],
                    nyha_class=llm_m.nyha_class,  # placeholder — not used for overlap
                    evidence="",
                )
                ov = llm_m.overlap(tmp)
                if ov > best_overlap:
                    best_overlap = ov
                    best_spacy = sp
            except Exception:
                continue

        # --- Step 2: copy date-linkage fields if a match was found ---
        if best_spacy and best_overlap > OVERLAP_THRESHOLD:
            linked_date_obj: Optional[DspyDate] = None
            char_distance: Optional[int] = best_spacy.char_distance

            ld = best_spacy.linked_date
            if ld:
                try:
                    linked_date_obj = DspyDate(
                        text=ld.text,
                        parsed_date=ld.parsed_date,
                        original_start=ld.original_start,
                        original_end=ld.original_end,
                    )
                except Exception:
                    pass

            llm_m = llm_m.model_copy(
                update={
                    "linked_date": linked_date_obj,
                    "char_distance": char_distance,
                }
            )

        enriched.append(llm_m)

    return enriched
