from typing import Literal

import dspy

_NYHA_VALID_LABELS = (
    "'I', 'I-II', 'II', 'II-III', 'II-IV', 'III', 'III-IV', 'I-IV', 'IV'"
)

_EVIDENCE_FORMAT_B = (
    "'no cue; date <DATE> found in context, <N> years/months before note date, classified as historical' "
    "or 'no cue; date <DATE> found in context, within 3 months of note date, classified as current' "
    "or 'no cue; no relevant date in context; default current'"
)

_SHARED_DOCSTRING_PREFIX = """
    You are a clinical NLP expert specialised in heart failure documentation.
    Your task is to classify the NYHA functional class and temporality of a
    single NYHA mention extracted from a clinical note.

    --- TARGET SPAN ---
    The exact mention to classify is marked with >>> and <<< delimiters
    in the text, e.g.: >>>NYHA III<
    Classify ONLY this marked occurrence. If the same text appears elsewhere
    without markers, ignore those occurrences entirely. If the span field and
    the marked occurrence in the text diverge, always trust the >>> <<< markers.

    --- NYHA CLASS RULES ---
    Assign the NYHA functional class expressed by the MARKED mention only.
    Use a range label (e.g. 'II-III') ONLY when the note explicitly states
    a range (e.g. 'NYHA II-III', 'NYHA I–II') OR when the marked span
    contains a recognised modifier suffix or qualifier phrase. Apply the
    following deterministic mapping — the modifier must be explicitly present
    in the marked span:
      'I+'  or 'II-'                                          → 'I-II'
      'II+' or 'III-' or qualifier on II (e.g. 'II borderline',
        'II à la limite') or qualifier meaning below III
        (e.g. 'III borderline')                               → 'II-III'
      'III+' or 'IV-'                                         → 'III-IV'
    Do NOT infer a modifier or range from context, improvement, or
    deterioration. If no modifier is present, use a single class label.
    Class I is a fully valid designation; do not override it with a higher
    class unless the marked span clearly states otherwise.
    Clinical formatting patterns such as label-value layouts (e.g.
    '(NYHA): III') are valid; assign the class from the Roman numeral present.

    The preidentified_nyha_class is a useful starting point when the marked
    span is ambiguous or abbreviated and consistent with it. However:
    - ALWAYS prioritise the marked span and its immediate context over this
      prior.
    - Override the prior whenever the marked span or its sentence clearly
      states a different class. Examples of when to override: the span reads
      'NYHA II-III' but the prior is 'II'; the span reads 'NYHA II+' but the
      prior is 'II'; the sentence explicitly corrects the prior ('not NYHA II
      but NYHA III').
    - nyha_class must always reflect the class designation visible in the
      marked span — span content and nyha_class output must never contradict
      each other.
    - Do NOT infer improvement or deterioration unless explicitly stated.

    --- TEMPORALITY RULES ---
    'current'    — the MARKED mention describes the patient's functional status
                   at the time of writing (this clinical encounter).
    'historical' — the MARKED mention describes a past functional status.

    RESOLUTION ORDER (apply in order, stop at first match):
    1. Local linguistic cues in the same sentence as the MARKED span.
       - If an unambiguous current-tense cue is present (e.g. 'momenteel',
         'nu', 'currently') with no conflicting past date directly anchoring
         the MARKED mention in that clause, classify as 'current'.
       - If an unambiguous historical cue is present (e.g. 'previously',
         'voorheen', an explicit past date directly anchoring the MARKED
         mention in the same clause), classify as 'historical'.
       - When a current-tense cue and a past date co-occur in the same
         sentence, determine which one directly anchors the MARKED mention:
         if the past date contextualises a separate, incidental event (e.g.
         'momenteel, zoals in 2019, NYHA III' — the 2019 date records when
         the status also held, not when it was measured), the current-tense
         cue wins and the result is 'current'. If the past date is the primary
         temporal anchor for the MARKED mention and the current-tense cue is
         incidental, the past date wins and the result is 'historical'.
    2. A date found in the text that clearly and directly contextualises the
       MARKED mention (not a nearby unrelated event such as admission, lab
       result, or prior diagnosis). Linguistic cues from Rule 1 always take
       priority; apply date-based reasoning only when Rule 1 yields no clear
       signal. Use authored_datetime to compute temporal distance:
         - Same day as note date → 'current'.
         - Within the same clinical encounter (≤3 months before note date)
           → 'current'.
         - More than 3 months before note date → 'historical'.

       DATE ANCHORING — how to determine whether a date contextualises the
       MARKED mention:

       A date directly anchors the MARKED mention when it appears as the
       primary temporal label for the clinical event or measurement that
       contains the MARKED mention. The most common pattern in Dutch clinical
       notes is a year or month-year at the start of a bullet or line,
       followed by a clinical event description that includes the NYHA mention:

         '2019  Echo cor: EF 35%, >>>NYHA III<<<'
             → 2019 directly anchors the mention; authored_datetime is 2023
             → more than 3 months before note date → 'historical'

         '08-2021  Opname HFU: >>>NYHA II-III<<<, behandeling gestart'
             → 08-2021 directly anchors the mention; if authored_datetime
               is 2024 → more than 3 months before note date → 'historical'

       A date does NOT anchor the MARKED mention when it labels a different
       event on a nearby line, or when it refers to a procedure, lab result,
       or diagnosis that is incidental to the NYHA measurement:

         '2020  MRI hart: LVEF 40%
          >>>NYHA II<<<'
             → 2020 anchors the MRI, not the NYHA mention on the next line
             → do not use 2020 as the temporal anchor for >>>NYHA II<<<
             → proceed to Rule 3 or Rule 4

       When the date and the MARKED mention appear in the same bullet point,
       sentence, or structured history line with no intervening event label,
       treat the date as anchoring the mention. When they are on separate
       lines with different event labels, do not transfer the date anchor
       across lines.
"""

_RULE_3_FULL = """    3. If the MARKED mention appears in a final clinical assessment or summary
       section with no temporal marker, classify as 'current'. Recognise such
       sections by headers such as: 'Conclusie', 'Assessment', 'Plan',
       'Samenvatting', 'Indruk', 'Beleid', 'Conclusion', 'Summary',
       or equivalent closing paragraphs without a section header.
"""

_RULE_3_WINDOWED = """    3. If the MARKED mention appears in a final clinical assessment or summary
       section with no temporal marker, classify as 'current'. Recognise such
       sections by headers such as: 'Conclusie', 'Assessment', 'Plan',
       'Samenvatting', 'Indruk', 'Beleid', 'Conclusion', 'Summary',
       or equivalent closing paragraphs without a section header.
       IMPORTANT: Because the input is a window of the full note, the
       relevant section header may fall outside the provided excerpt. Apply
       this rule ONLY when a recognised section header or an equivalent
       closing paragraph is explicitly visible within the windowed text.
       If no such header or closing paragraph is visible, skip this rule
       and proceed to Rule 4.
"""

_SHARED_DOCSTRING_SUFFIX = """    4. Default → 'current'.

    Linguistic cues (non-exhaustive; apply the general principle to unseen
    cues: any present-tense verb with no explicit past temporal anchor in
    the same clause signals 'current'; any past-tense verb or explicit past
    temporal anchor signals 'historical'):
    - Current  (EN): 'currently', 'now', 'at this visit', 'remains',
      'is classified as', 'tolerates', 'presents with', 'has'
    - Current  (NL): 'momenteel', 'nu', 'vandaag', 'huidig', 'is',
      'heeft', 'bevindt zich in', 'classificeert als'
    - Historical (EN): 'previously', 'was', 'had', 'prior to', 'before',
      'last year', 'last month', 'at the time of admission', 'used to be',
      'historically', 'at that time'
    - Historical (NL): 'voorheen', 'vroeger', 'was', 'had', 'eerder',
      'bij opname', 'destijds', 'in het verleden', 'vorig jaar',
      'vorige maand', explicit past dates

    --- USING DATE MENTIONS ---
    Any date in the text may serve as evidence, but:
    - Verify that it directly contextualises the MARKED mention, not a
      different event (admission, lab result, prior diagnosis, etc.).
    - If it does not clearly refer to the MARKED mention, ignore it.
    - Linguistic cues (Rule 1) always take priority over date-based reasoning
      (Rule 2).
    - If authored_datetime is missing or unknown, omit date-based reasoning
      entirely and proceed to the next applicable rule in the resolution order.

    --- EVIDENCE ---
    Always return one of:
    a) The exact word or phrase verbatim from the text near the MARKED span
       that drives the temporality decision. Do not paraphrase or normalise.
    b) If no explicit cue exists, a short reasoning string in one of these
       formats:
       {evidence_format_b}
    Never fabricate text that does not appear in the note. Never leave empty.
    Note: fallback reasoning strings in format (b) are constructed summaries,
    not verbatim quotations — this is expected when no explicit cue exists.
    """

_SHARED_DOCSTRING_FULL = (
    _SHARED_DOCSTRING_PREFIX
    + _RULE_3_FULL
    + _SHARED_DOCSTRING_SUFFIX.format(evidence_format_b=_EVIDENCE_FORMAT_B)
)

_SHARED_DOCSTRING_WINDOWED = (
    _SHARED_DOCSTRING_PREFIX
    + _RULE_3_WINDOWED
    + _SHARED_DOCSTRING_SUFFIX.format(evidence_format_b=_EVIDENCE_FORMAT_B)
)

_SHARED_FIELDS = dict(
    span=dspy.InputField(
        desc=(
            "The exact NYHA mention text to classify, without markers "
            "(e.g. 'NYHA III', 'NYHA class II-III', 'NYHA II+'). Provided "
            "for reference; if it diverges from the marked occurrence in the "
            "text, trust the >>> <<< markers."
        )
    ),
    preidentified_nyha_class=dspy.InputField(
        desc=(
            "NYHA class from span detection. Follows the same label schema "
            f"as the nyha_class output field ({_NYHA_VALID_LABELS}). "
            "Useful when the marked span is ambiguous or abbreviated and "
            "consistent with this value. Always override with what the marked "
            "span or its immediate sentence clearly states — span content and "
            "the nyha_class output must never contradict each other."
        )
    ),
    authored_datetime=dspy.InputField(
        desc=(
            "The date this note was authored (YYYY-MM-DD). Used only after "
            "linguistic cues (Rule 1) have been evaluated and found "
            "inconclusive, to compute temporal distance from any date found "
            "in the text that directly contextualises the marked mention. "
            "If missing or unknown, omit date-based reasoning and proceed "
            "with document-section defaults or the fallback."
        )
    ),
    nyha_class=dspy.OutputField(
        desc=(
            f"Final NYHA class label for the marked mention ({_NYHA_VALID_LABELS}). "
            "Must reflect the class designation visible in the marked span, "
            "including any modifier mapping (e.g. 'NYHA II+' → 'II-III'). "
            "Must never contradict the span content."
        )
    ),
    temporality=dspy.OutputField(
        desc=(
            "Temporal classification of the marked mention: 'current' or "
            "'historical'. Determined by the resolution order defined in the "
            "prompt: (1) local linguistic cues; (2) semantically relevant "
            "date in context evaluated against authored_datetime, with ≤3 "
            "months classified as 'current' and >3 months as 'historical'; "
            "(3) document-section default for final assessment or summary "
            "sections; (4) default 'current'."
        )
    ),
    evidence=dspy.OutputField(
        desc=(
            "Verbatim phrase from the text near the marked span driving the "
            "temporality decision (format a), OR a short constructed reasoning "
            "string when no explicit cue exists (format b), as defined in the "
            "prompt under EVIDENCE. Never paraphrase a real cue that exists "
            "in the note. Never fabricate text. Never leave empty."
        )
    ),
)


class SpanClassificationSignature(dspy.Signature):
    __doc__ = _SHARED_DOCSTRING_FULL

    deidentified_text: str = dspy.InputField(
        desc=(
            "The full deidentified clinical note text. The target span is "
            "marked with >>> and <<< delimiters. Classify ONLY the marked "
            "occurrence; ignore any unmarked repetitions of the same text "
            "elsewhere in the note. Use local sentence-level context around "
            "the marker first, but attend to the full note if needed — "
            "particularly to identify whether the marked mention appears in "
            "a final clinical assessment or summary section."
        )
    )

    span: str = _SHARED_FIELDS["span"]
    preidentified_nyha_class: str = _SHARED_FIELDS["preidentified_nyha_class"]
    authored_datetime: str = _SHARED_FIELDS["authored_datetime"]

    nyha_class: Literal[
        "I", "I-IV", "I-II", "II", "II-III", "II-IV", "III", "III-IV", "IV"
    ] = _SHARED_FIELDS["nyha_class"]
    temporality: Literal["current", "historical"] = _SHARED_FIELDS["temporality"]
    evidence: str = _SHARED_FIELDS["evidence"]


class SpanClassificationSignatureWindowed(dspy.Signature):
    __doc__ = _SHARED_DOCSTRING_WINDOWED

    deidentified_text: str = dspy.InputField(
        desc=(
            "A character-level window of the deidentified clinical note, "
            "centred on the NYHA mention. The window contains the marked "
            "span (delimited by >>> and <<<) plus surrounding context, but "
            "does NOT represent the full note. Section-header detection "
            "(Rule 3) applies only when a recognised header or closing "
            "paragraph is explicitly visible within this excerpt."
        )
    )

    span: str = _SHARED_FIELDS["span"]
    preidentified_nyha_class: str = _SHARED_FIELDS["preidentified_nyha_class"]
    authored_datetime: str = _SHARED_FIELDS["authored_datetime"]

    nyha_class: Literal[
        "I", "I-IV", "I-II", "II", "II-III", "II-IV", "III", "III-IV", "IV"
    ] = _SHARED_FIELDS["nyha_class"]
    temporality: Literal["current", "historical"] = _SHARED_FIELDS["temporality"]
    evidence: str = _SHARED_FIELDS["evidence"]


def extract_windowed_text(
    text: str,
    span_start: int,
    span_end: int,
    window: int,
    open_marker: str = ">>>",
    close_marker: str = "<<<",
) -> str:
    """
    Extract a character window around a span and insert >>> / <<< markers.

    Parameters
    ----------
    text        : Full deidentified clinical note.
    span_start  : Inclusive start character offset of the NYHA mention.
    span_end    : Exclusive end character offset of the NYHA mention.
    window      : Number of characters to include on each side of the span.
    open_marker : Delimiter inserted immediately before the span (default '>>>').
    close_marker: Delimiter inserted immediately after the span (default '<<<').

    Returns
    -------
    Windowed string with the span delimited by open_marker / close_marker.
    """
    if span_start < 0 or span_end > len(text) or span_start >= span_end:
        raise ValueError(
            f"Invalid span offsets: span_start={span_start}, span_end={span_end}, "
            f"text length={len(text)}"
        )

    window_start = max(0, span_start - window)
    window_end = min(len(text), span_end + window)

    before = text[window_start:span_start]
    span_text = text[span_start:span_end]
    after = text[span_end:window_end]

    return f"{before}{open_marker}{span_text}{close_marker}{after}"
