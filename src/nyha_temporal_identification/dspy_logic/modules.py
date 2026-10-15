"""
dspy_modules.py — DSPy extraction module with spaCy pre-processing.

Pipeline overview:
    Layer 0 — spaCy pre-processing (rule-based):
        Extracts candidate NYHA spans and links nearby dates. Guides Layer 1
        and enriches results in Layer 3.

    Layer 1 — Span identification (LLM, SpanIdentificationSignature):
        Identifies all NYHA mentions in the note, using spaCy candidates as
        high-recall seeds. May add mentions the rule-based step missed.

    Layer 2 — Span classification (LLM, SpanClassificationSignature):
        Classifies each span independently for nyha_class, temporality, and
        evidence, using a windowed context excerpt around the mention.

    Layer 3 — Post-processing (rule-based):
        a. resolve_indices(): maps span text + occurrence index to exact
           character offsets in the note.
        b. validate_against_text(): drops spans whose resolved indices do not
           reproduce the original span text.
        c. merge_with_spacy_spans(): enriches validated mentions with
           linked_date and char_distance from spaCy, then conditionally
           re-runs multi-signal temporality inference.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, List, Literal, Optional, cast

from nyha_temporal_identification.dspy_logic.utils import build_classification_context

if TYPE_CHECKING:
    pass

import dspy

from nyha_temporal_identification.data_model.dspy_model import (
    NYHAMention,
    NYHAMentionRaw,
)
from nyha_temporal_identification.data_model.general import NyhaMention
from nyha_temporal_identification.dspy_logic.models import (
    merge_with_spacy_spans,
    resolve_indices,
)
from nyha_temporal_identification.dspy_logic.regex import (
    CLASSIFICATION_CONTEXT_WINDOW,
    spacy_preprocess,
)
from nyha_temporal_identification.dspy_logic.signatures.nyha_label_translation import (
    normalise_nyha_class,
)
from nyha_temporal_identification.dspy_logic.signatures.span_classification_signature import (
    SpanClassificationSignature,
    SpanClassificationSignatureWindowed,
)
from nyha_temporal_identification.dspy_logic.signatures.span_identification_signature import (
    RawMentionOutput,
    SpanIdentificationSignature,
)

ExtractorVariant = Literal["with_identification", "spacy_only"]
ClassificationContextVariant = Literal["windowed_context", "full_context"]


def get_extractor_variant(use_identification: bool) -> ExtractorVariant:
    return "with_identification" if use_identification else "spacy_only"


def get_classification_context_variant(
    use_windowed_classification_context: bool,
) -> ClassificationContextVariant:
    if use_windowed_classification_context:
        return "windowed_context"
    return "full_context"


def get_extractor_run_variant(
    use_identification: bool,
    use_windowed_classification_context: bool,
) -> str:
    extractor_variant = get_extractor_variant(use_identification)
    context_variant = get_classification_context_variant(
        use_windowed_classification_context
    )
    return f"{extractor_variant}_{context_variant}"


class NYHAExtractor(dspy.Module):
    """
    Four-layer pipeline for extracting and classifying NYHA functional class
    mentions from deidentified Dutch clinical notes.

    Args:
        max_retries:               Maximum LLM retry attempts per layer. Default: 3.
        use_identification:        If True, run Layer 1 (LLM span identification).
                                   If False, promote all spaCy candidates directly
                                   to Layer 2, skipping LLM-based span verification.
                                   Default: True.
        failsafe_dropped_current:  If True, re-inject every dropped/uncovered spaCy
                                   candidate as current/source="failsafe" after Layer 3
                                   so it is masked by the downstream consumer. Only
                                   enabled on the annotation/inference path; all
                                   eval/optimisation steps pass False. Default: False.
    """

    def __init__(
        self,
        identify_lm=None,
        classify_lm=None,
        max_retries: int = 3,
        use_identification: bool = True,
        use_windowed_classification_context: bool = False,
        tokenizer=None,
        max_eval_tokens: int = 0,
        failsafe_dropped_current: bool = False,
    ):
        super().__init__()
        self.use_identification = use_identification
        self.use_windowed_classification_context = use_windowed_classification_context

        self.identify: Optional[dspy.ChainOfThought] = None
        if self.use_identification:
            self.identify = dspy.ChainOfThought(SpanIdentificationSignature)
        else:
            self.identify = None
        if self.use_windowed_classification_context:
            self.classify = dspy.ChainOfThought(SpanClassificationSignatureWindowed)
        else:
            self.classify = dspy.ChainOfThought(SpanClassificationSignature)

        if identify_lm and self.identify:
            self.identify.set_lm(identify_lm)
        if classify_lm:
            self.classify.set_lm(classify_lm)
        self.max_retries = max_retries
        self.tokenizer = tokenizer
        self.max_eval_tokens = max_eval_tokens
        self.failsafe_dropped_current = failsafe_dropped_current

    def _identify_with_retry(
        self,
        deidentified_text: str,
        preextracted_spans: str,
    ) -> list[RawMentionOutput]:
        """
        Run LLM span identification with retry logic.

        Args:
            deidentified_text:  Full clinical note text.
            preextracted_spans: JSON-serialised spaCy candidates injected as
                                seeds into the prompt.

        Returns:
            List of raw mention outputs from the LLM, or [] if all attempts fail.
        """
        if self.identify:
            for attempt in range(self.max_retries):
                try:
                    result = self.identify(
                        deidentified_text=deidentified_text,
                        preextracted_spans=preextracted_spans,
                    )
                    assert isinstance(result.raw_spans, list)
                    return result.raw_spans
                except Exception as exc:
                    logging.warning(
                        f"Span identification attempt {attempt + 1} failed: {exc}"
                    )

        return []

    def _classify_with_retry(
        self,
        deidentified_text: str,
        span: str,
        preidentified_nyha_class: str,
        authored_datetime: str,
    ) -> (
        tuple[
            Literal[
                "I", "I-II", "I-IV", "II", "II-III", "II-IV", "III", "III-IV", "IV"
            ],
            Literal["current", "historical"],
            str,
        ]
        | None
    ):
        """
        Run LLM span classification with retry logic.

        Args:
            deidentified_text:      Windowed context excerpt around the span.
            span:                   Exact NYHA mention text to classify.
            preidentified_nyha_class: NYHA class from spaCy, used as a prior.
            authored_datetime:      Note creation date (YYYY-MM-DD).

        Returns:
            (nyha_class, temporality, evidence) on success, None if all
            attempts fail.
        """
        for attempt in range(self.max_retries):
            try:
                result = self.classify(
                    deidentified_text=deidentified_text,
                    span=span,
                    preidentified_nyha_class=preidentified_nyha_class,
                    authored_datetime=authored_datetime,
                )
                return result.nyha_class, result.temporality, result.evidence
            except Exception as exc:
                logging.warning(f"Classification attempt {attempt + 1} failed: {exc}")

        logging.warning(
            f"Classification failed for span '{span}' after {self.max_retries} attempts. "
            "Skipping."
        )
        return None

    def _resolve_char_position(
        self,
        text: str,
        span: str,
        occurrence: int,
    ) -> tuple[int, int] | None:
        """
        Find the character offsets of the Nth occurrence of span in text.

        Args:
            text:       Full note text.
            span:       Span string to locate.
            occurrence: 1-based occurrence index.

        Returns:
            (char_start, char_end) or None if the occurrence is not found.
        """
        span_lower = span.strip().lower()
        search_from = 0
        char_start = -1

        for _ in range(occurrence):
            idx = text.lower().find(span_lower, search_from)
            if idx == -1:
                return None
            char_start = idx
            search_from = idx + 1
        return char_start, char_start + len(span)

    def _build_raw_spans_from_spacy(
        self,
        spacy_mentions: list[NyhaMention],
    ) -> list[RawMentionOutput]:
        """
        Promote spaCy candidates directly to Layer 2 (bypasses LLM identification).

        Assigns per-span occurrence indices so downstream resolution works
        correctly when the same surface form appears multiple times.
        """
        occurrence_counter: dict[str, int] = {}
        raw_spans: list[RawMentionOutput] = []
        for sp in sorted(spacy_mentions, key=lambda m: m.start):
            key = sp.original_text.lower()
            occurrence_counter[key] = occurrence_counter.get(key, 0) + 1
            raw_spans.append(
                RawMentionOutput(
                    span=sp.original_text,
                    nyha_class=cast(
                        Literal[
                            "I",
                            "I-II",
                            "I-IV",
                            "II",
                            "II-III",
                            "II-IV",
                            "III",
                            "III-IV",
                            "IV",
                        ],
                        normalise_nyha_class(sp.get_nyha_string_class()),
                    ),
                    occurrence=occurrence_counter[key],
                )
            )
        return raw_spans

    def forward(
        self,
        deidentified_text: str,
        authored_datetime: Optional[str] = None,
    ) -> List[NYHAMention]:
        """
        Run the full four-layer extraction pipeline for one clinical note.

        Args:
            deidentified_text:  Raw deidentified clinical note text.
            authored_datetime:  Note creation date in ISO format (optional).
                                Used for temporal distance calculations.

        Returns:
            List of NYHAMention objects. Empty if no mentions were found or
            all failed validation.
        """
        # ── Layer 0: spaCy pre-processing ─────────────────────────────────────
        spacy_mentions = spacy_preprocess(
            deidentified_text,
            authored_datetime,
        )

        # ── Layer 1: span identification ───────────────────────────────────────
        # drop_reasons tracks why each spaCy candidate was lost (audit only).
        drop_reasons: dict[int, str] = {}

        if self.use_identification:
            occurrence_counter: dict[str, int] = {}
            preextracted_list = []
            for sp in spacy_mentions:
                key = sp.original_text.lower()
                occurrence_counter[key] = occurrence_counter.get(key, 0) + 1
                preextracted_list.append(
                    {
                        "span": sp.original_text,
                        "nyha_class": normalise_nyha_class(sp.get_nyha_string_class()),
                        "occurrence": occurrence_counter[key],
                    }
                )
            preextracted_spans = json.dumps(preextracted_list)
            raw_spans = self._identify_with_retry(
                deidentified_text,
                preextracted_spans,
            )
            # Candidates absent from Layer-1 output are attribution "identification_dropped".
            # We can't perfectly reconstruct which spaCy span was dropped, so we mark all
            # spaCy spans as potentially identification-dropped; the coverage test in
            # reinject_dropped_mentions is the authoritative gating.
            if not raw_spans:
                for sp in spacy_mentions:
                    drop_reasons[id(sp)] = "identification_dropped"
        else:
            raw_spans = self._build_raw_spans_from_spacy(
                spacy_mentions,
            )

        # ── Layer 2: per-span classification ──────────────────────────────────
        # Track which spaCy candidates fail at classification (best-effort).
        classify_failed_spans: set[str] = set()
        raw_mentions: list[NYHAMentionRaw] = []
        for rs in raw_spans:
            position = self._resolve_char_position(
                deidentified_text,
                rs.span,
                rs.occurrence,
            )
            context = build_classification_context(
                deidentified_text,
                position[0] if position is not None else None,
                position[1] if position is not None else None,
                window=CLASSIFICATION_CONTEXT_WINDOW if self.use_windowed_classification_context else None,
                tokenizer=self.tokenizer,
                max_tokens=self.max_eval_tokens,
            )

            classification = self._classify_with_retry(
                deidentified_text=context,
                span=rs.span,
                preidentified_nyha_class=rs.nyha_class,
                authored_datetime=authored_datetime or "",
            )
            if classification is None:
                classify_failed_spans.add(rs.span.lower())
                continue

            nyha_class, temporality, evidence = classification
            raw_mentions.append(
                NYHAMentionRaw(
                    span=rs.span,
                    nyha_class=nyha_class,
                    temporality=temporality,
                    evidence=evidence,
                    occurrence=rs.occurrence,
                )
            )

        # Attribute classify_failed to spaCy candidates whose span text matched.
        for sp in spacy_mentions:
            if sp.original_text.lower() in classify_failed_spans:
                drop_reasons.setdefault(id(sp), "classify_failed")

        # ── Layer 3: post-processing ───────────────────────────────────────────
        resolved = resolve_indices(deidentified_text, raw_mentions)
        validated = [m for m in resolved if m.validate_against_text(deidentified_text)]
        validated = merge_with_spacy_spans(validated, spacy_mentions, deidentified_text)

        # Mark spaCy candidates that survived resolve/validate as resolve_failed if not
        # already attributed (they resolved in Layer 2 but didn't pass Layer 3).
        resolved_spans = {m.span.lower() for m in resolved}
        for sp in spacy_mentions:
            span_lower = sp.original_text.lower()
            if span_lower not in classify_failed_spans and span_lower not in resolved_spans:
                drop_reasons.setdefault(id(sp), "resolve_or_validate_failed")

        # ── Failsafe re-injection (Layer 3, optional) ─────────────────────────
        if self.failsafe_dropped_current:
            from nyha_temporal_identification.dspy_logic.failsafe import (
                reinject_dropped_mentions,
                reinject_uncovered_regex_spans,
            )

            failsafe = reinject_dropped_mentions(
                spacy_mentions, validated, drop_reasons=drop_reasons
            )
            regex_failsafe = reinject_uncovered_regex_spans(
                deidentified_text,
                survivors=validated,
                existing_failsafe=failsafe,
            )
            return validated + failsafe + regex_failsafe

        return validated


class NYHAExtractorWithSpanIdentification(NYHAExtractor):
    """Extractor variant that runs LLM span identification before classification."""

    def __init__(
        self,
        identify_lm=None,
        classify_lm=None,
        max_retries: int = 3,
        use_windowed_classification_context: bool = False,
        tokenizer=None,
        max_eval_tokens: int = 0,
        failsafe_dropped_current: bool = False,
    ):
        super().__init__(
            identify_lm=identify_lm,
            classify_lm=classify_lm,
            max_retries=max_retries,
            use_identification=True,
            use_windowed_classification_context=use_windowed_classification_context,
            tokenizer=tokenizer,
            max_eval_tokens=max_eval_tokens,
            failsafe_dropped_current=failsafe_dropped_current,
        )


class NYHAExtractorSpacyOnly(NYHAExtractor):
    """Extractor variant that skips LLM span identification and uses spaCy spans only."""

    def __init__(
        self,
        identify_lm=None,
        classify_lm=None,
        max_retries: int = 3,
        use_windowed_classification_context: bool = False,
        tokenizer=None,
        max_eval_tokens: int = 0,
        failsafe_dropped_current: bool = False,
    ):
        super().__init__(
            identify_lm=identify_lm,
            classify_lm=classify_lm,
            max_retries=max_retries,
            use_identification=False,
            use_windowed_classification_context=use_windowed_classification_context,
            tokenizer=tokenizer,
            max_eval_tokens=max_eval_tokens,
            failsafe_dropped_current=failsafe_dropped_current,
        )


def build_nyha_extractor(
    identify_lm=None,
    classify_lm=None,
    max_retries: int = 3,
    use_identification: bool = True,
    use_windowed_classification_context: bool = False,
    tokenizer=None,
    max_eval_tokens: int = 0,
    failsafe_dropped_current: bool = False,
) -> NYHAExtractor:
    if use_identification:
        return NYHAExtractorWithSpanIdentification(
            identify_lm=identify_lm,
            classify_lm=classify_lm,
            max_retries=max_retries,
            use_windowed_classification_context=use_windowed_classification_context,
            tokenizer=tokenizer,
            max_eval_tokens=max_eval_tokens,
            failsafe_dropped_current=failsafe_dropped_current,
        )

    return NYHAExtractorSpacyOnly(
        identify_lm=identify_lm,
        classify_lm=classify_lm,
        max_retries=max_retries,
        use_windowed_classification_context=use_windowed_classification_context,
        tokenizer=tokenizer,
        max_eval_tokens=max_eval_tokens,
        failsafe_dropped_current=failsafe_dropped_current,
    )
