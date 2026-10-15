"""
failsafe.py — Leakage-safe re-injection of dropped spaCy candidates.

Every spaCy candidate not covered by a surviving LLM mention is re-injected
as temporality="current"/source="failsafe" so the downstream masker hides it.
Coverage-based: masking correctness relies solely on the overlap test, not on
the drop-reason attribution (which is best-effort for audit only).

See docs/spec/leakage_safe_failsafe.md for full rationale and contract.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from nyha_temporal_identification.constants import OVERLAP_THRESHOLD
from nyha_temporal_identification.data_model.dspy_model import NYHAMention

if TYPE_CHECKING:
    from nyha_temporal_identification.data_model.general import NyhaMention


def _bare_failsafe_mention(
    candidate: NyhaMention,
    reason: str,
    err: Exception,
) -> NYHAMention | None:
    """
    Offsets-only fallback when a candidate's class/vector cannot be promoted.

    Under the canonical class/vector convention spaCy no longer produces
    endpoints-only / non-contiguous vectors, so this path is defensive-only: it
    is reached only for a genuinely malformed candidate (e.g. bad offsets, a
    hand-built non-contiguous vector). Class is irrelevant to masking, so we
    re-inject the bare span (temporality="current", source="failsafe") rather
    than dropping it: losing one current value to a validation error would defeat
    the leakage-safety guarantee and must never fail the whole note. Returns None
    only if even the bare span is unconstructable (e.g. invalid offsets), in
    which case the caller logs+skips.
    """
    try:
        return NYHAMention(
            char_start=getattr(candidate, "original_start", -1),
            char_end=getattr(candidate, "original_end", -1),
            span=getattr(candidate, "original_text", ""),
            temporality="current",
            source="failsafe",
            evidence=(
                f"failsafe re-injection ({reason}; "
                f"class dropped: {type(err).__name__})"
            ),
        )
    except Exception:
        return None


def reinject_dropped_mentions(
    spacy_mentions: list[NyhaMention],
    survivors: list[NYHAMention],
    *,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    drop_reasons: dict[int, str] | None = None,
) -> list[NYHAMention]:
    """
    Return failsafe mentions to append for every uncovered spaCy candidate.

    A candidate is "covered" if any survivor overlaps it at >= overlap_threshold
    (via NYHAMention.overlap, which is duck-typed for NyhaMention too).
    Uncovered candidates are promoted to NYHAMention with temporality="current"
    and source="failsafe" — the safe direction for masking regardless of the
    spaCy/rule temporality.

    Args:
        spacy_mentions:    Layer-0 spaCy candidates for the document.
        survivors:         Validated LLM mentions after Layer 3.
        overlap_threshold: IoU threshold from evaluation.evaluation.
        drop_reasons:      Optional map from id(candidate) to reason string
                           (e.g. "classify_failed"). Used only for evidence text.

    Returns:
        List of failsafe NYHAMentions to append (does not mutate survivors).
    """
    if drop_reasons is None:
        drop_reasons = {}

    failsafe_mentions: list[NYHAMention] = []
    for candidate in spacy_mentions:
        covered = any(
            s.overlap(candidate) >= overlap_threshold for s in survivors
        )
        if covered:
            continue

        reason = drop_reasons.get(id(candidate), "uncovered")
        try:
            promoted = NYHAMention.from_nyha_mention(candidate)
            promoted = promoted.model_copy(
                update={
                    "temporality": "current",
                    "source": "failsafe",
                    "evidence": f"failsafe re-injection ({reason})",
                }
            )
        except Exception as err:
            # Defensive safety net: spaCy candidates are valid under the canonical
            # convention (the endpoints-only range case is gone), so this only
            # trips on a genuinely malformed candidate. Re-inject the bare span
            # instead of letting the error propagate out of forward() and fail
            # (and un-cache) the whole note — which would leave this current value
            # unmasked and leak it downstream.
            promoted = _bare_failsafe_mention(candidate, reason, err)
            if promoted is None:
                logging.warning(
                    "Failsafe could not re-inject candidate %r (offsets %s-%s): %s",
                    getattr(candidate, "original_text", "?"),
                    getattr(candidate, "original_start", "?"),
                    getattr(candidate, "original_end", "?"),
                    err,
                )
                continue
        failsafe_mentions.append(promoted)

    return failsafe_mentions


def assess_current_consistency(
    mentions: list[NYHAMention],
    policy: str = "balanced",
) -> dict:
    """
    Note-level consistency check on a document's final mention list.

    Args:
        mentions: All mentions for a single note (survivors + failsafe).
        policy:   "balanced"     — conflict only among confident (source="llm")
                                   current mentions; failsafe currents reported
                                   but never trigger conflict.
                  "conservative" — conflict among all currents (llm + failsafe).
                  "off"          — always ok.

    Returns:
        {
            "policy": str,
            "note_status": "ok" | "conflict",
            "reinjected_count": int,
            "confident_current_classes": list[str],
            "failsafe_current_classes": list[str],
        }
    """
    confident_currents = [
        m for m in mentions if m.source == "llm" and m.temporality == "current"
    ]
    failsafe_currents = [m for m in mentions if m.source == "failsafe"]

    confident_classes = [
        c for m in confident_currents if (c := m.get_nyha_class()) is not None
    ]
    failsafe_classes = [
        c for m in failsafe_currents if (c := m.get_nyha_class()) is not None
    ]

    if policy == "off":
        note_status = "ok"
    elif policy == "conservative":
        all_classes = confident_classes + failsafe_classes
        note_status = "conflict" if len(set(all_classes)) > 1 else "ok"
    else:  # balanced (default)
        note_status = "conflict" if len(set(confident_classes)) > 1 else "ok"

    return {
        "policy": policy,
        "note_status": note_status,
        "reinjected_count": len(failsafe_currents),
        "confident_current_classes": confident_classes,
        "failsafe_current_classes": failsafe_classes,
    }


def reinject_uncovered_regex_spans(
    text: str,
    survivors: list[NYHAMention],
    existing_failsafe: list[NYHAMention],
    *,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> list[NYHAMention]:
    """
    Pass 2 failsafe: re-inject any NYHA regex match not covered by survivors
    or pass-1 failsafe mentions.

    Runs find_raw_nyha_spans on the raw text (no spaCy, no cleaning) and
    promotes each uncovered interval to a NYHAMention with
    temporality="current" and source="failsafe".  Upgrades the leakage
    guarantee from "given spaCy recall" to "given regex recall".

    Args:
        text:              Raw deidentified note text.
        survivors:         Validated LLM mentions after Layer 3.
        existing_failsafe: Pass-1 failsafe mentions (from reinject_dropped_mentions).
        overlap_threshold: IoU threshold for coverage test.

    Returns:
        New failsafe NYHAMentions to append (does not mutate survivors or
        existing_failsafe).
    """
    import re

    from nyha_temporal_identification.extraction.parsing import (
        _build_nyha_patterns,
        find_raw_nyha_spans,
        parse_pair_or_single,
    )

    already: list[NYHAMention] = survivors + existing_failsafe
    raw_spans = find_raw_nyha_spans(text)
    patterns = _build_nyha_patterns()

    result: list[NYHAMention] = []
    for s, e in raw_spans:
        span_text = text[s:e]
        candidate = NYHAMention(
            char_start=s,
            char_end=e,
            span=span_text,
            temporality="current",
            source="failsafe",
            evidence="failsafe re-injection (regex_uncovered)",
        )
        if any(candidate.overlap(m) >= overlap_threshold for m in already):
            continue

        # Best-effort class: re-run patterns on the span to extract group 1 (the
        # value part), then parse it.  parse_pair_or_single expects just the value
        # token, not the full "NYHA II" string.
        parsed: np.ndarray | None = None
        for pat in patterns:
            pm = re.search(pat, span_text, re.IGNORECASE)
            if pm:
                parsed = parse_pair_or_single(pm.group(1))
                if parsed is not None:
                    break

        # parse_pair_or_single returns only valid vectors (single class or
        # adjacent pair) or None, so the parsed vector is always storable. An
        # I-III surface parses to None and keeps the bare span, preserving
        # leakage safety.
        if parsed is not None:
            nyha_vec = [int(v) for v in parsed.tolist()]
        else:
            nyha_vec = []
        candidate = candidate.model_copy(update={"nyha_vector": nyha_vec})
        result.append(candidate)

    return result
