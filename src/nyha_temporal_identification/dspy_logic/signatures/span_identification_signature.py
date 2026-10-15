from typing import List, Literal

import dspy
from pydantic import BaseModel, Field


class RawMentionOutput(BaseModel):
    span: str = Field(
        description=(
            "The exact substring as it appears in the clinical note, preserving "
            "original casing, punctuation and whitespace. Do not normalise, "
            "paraphrase or expand abbreviations. Example: 'NYHA III' not "
            "'NYHA class III'. If the pre-extracted span has imprecise boundaries, "
            "correct it to the verbatim substring from the note that best captures "
            "the full NYHA mention including its class designation — the corrected "
            "value replaces the pre-extracted one. Boundary correction MAY extend "
            "beyond the original span boundaries into the immediately surrounding "
            "text to recover a clearly associated Roman numeral or class designation. "
            "When a modifier suffix (+, -) or qualifier phrase (e.g. 'borderline', "
            "'à la limite') is present immediately after the class designation, include "
            "it in the span (e.g. 'NYHA II+', 'NYHA III borderline'). "
            "Clinical formatting patterns such as label-value layouts "
            "(e.g. '(NYHA): III') are valid spans."
        )
    )
    nyha_class: Literal[
        "I", "I-II", "I-IV", "II", "II-III", "II-IV", "III", "III-IV", "IV"
    ] = Field(
        description=(
            "The NYHA functional class expressed by this mention. Use a range "
            "label (e.g. 'II-III') only when the note explicitly describes a "
            "range OR when the span contains a recognised modifier suffix or "
            "qualifier phrase. Apply the following deterministic mapping — the "
            "modifier must be explicitly present in the span:\n"
            "  'I+'  or 'II-'  → 'I-II'\n"
            "  'II+' or 'III-' or qualifier on II (e.g. 'II borderline', "
            "'II à la limite') or qualifier meaning below III "
            "(e.g. 'III borderline') → 'II-III'\n"
            "  'III+' or 'IV-' → 'III-IV'\n"
            "Single-class mentions without any modifier must use a single label. "
            "Roman numerals only (e.g. 'III' not '3'). Must reflect the class "
            "designation present in the final span value."
        )
    )
    occurrence: int = Field(
        default=1,
        description=(
            "1-based index of this occurrence of the span text in document "
            "order, counted independently per unique span text (case-insensitive). "
            "Always count against the final span value — if a pre-extracted span "
            "was corrected, count occurrences of the corrected text, not the original. "
            "If 'NYHA II' appears three times, the occurrences are 1, 2, 3. "
            "If the span appears only once, set occurrence=1."
        ),
    )


class SpanIdentificationSignature(dspy.Signature):
    """
    You are a clinical NLP expert specialised in cardiology. Your task is to
    identify every valid mention of NYHA (New York Heart Association) functional
    class in the clinical note below.

    Pre-extracted candidate spans are provided as a starting point. Verify each
    one: include it only if it genuinely refers to a NYHA functional class with
    its class specification present. Remove invalid spans. You MAY add spans
    missed by pre-extraction. If the pre-extracted list is empty, scan the full
    note independently and extract all valid NYHA mentions from scratch.

    Valid NYHA mentions include the class designation (Roman numeral or range):
      'NYHA I', 'NYHA II', 'NYHA class III', 'Klasse NYHA II', 'NYHA I-II', 'in NYHA III'
    Note: class I is a fully valid designation and must not be discarded.
    Modifier suffixes and qualifier phrases attached to a class designation are
    also valid and must be included in the span:
      'NYHA II+', 'NYHA III-', 'NYHA II borderline', 'NYHA III à la limite'
    Clinical formatting patterns such as label-value layouts are valid:
      '(NYHA): III', 'Dyspnoe klasse (NYHA): III'
    Invalid mentions (do NOT include):
      - A standalone Roman numeral ('II') with no NYHA reference
      - References that do not identify a functional class

    Boundary correction examples:
      - Over-capture: pre-extracted 'in NYHA III severity' → corrected 'NYHA III'
      - Under-capture within span: pre-extracted 'NYHA' → corrected 'NYHA III'
        (Roman numeral immediately follows in the note)
      - Under-capture beyond span: pre-extracted '(NYHA)' where note reads
        '(NYHA): III' → corrected '(NYHA): III' (Roman numeral recovered from
        immediately surrounding context outside original span boundaries)
      - Already correct: pre-extracted 'NYHA class II' → keep as-is
      - Modifier present: pre-extracted 'NYHA II' where note reads 'NYHA II+'
        → corrected 'NYHA II+' (modifier is part of the span)

    Modifier-to-class mapping (apply deterministically when modifier is present):
      'I+'  or 'II-'                                          → 'I-II'
      'II+' or 'III-' or qualifier on II (e.g. 'II borderline',
        'II à la limite') or qualifier meaning below III
        (e.g. 'III borderline')                               → 'II-III'
      'III+' or 'IV-'                                         → 'III-IV'

    Rules:
    - Copy spans character-for-character from the note; do not paraphrase or
      expand abbreviations.
    - Boundary correction MAY extend beyond the original span boundaries into
      the immediately surrounding text in the note if doing so recovers a
      clearly associated Roman numeral or class designation. Do not absorb
      unrelated surrounding words or clauses — the corrected span must remain
      a tight expression of the NYHA mention and its class.
    - Include modifier suffixes (+, -) and qualifier phrases (e.g. 'borderline',
      'à la limite') in the span when they are explicitly present immediately
      after the class designation.
    - Discard a span only when no Roman numeral or class designation is
      recoverable from the local context at all, or when the span does not
      refer to NYHA functional class in any reading. Discard is a last resort,
      not the default response to a pre-extracted span missing its class
      designation.
    - If two spans overlap or are nested, keep the one with more precise or
      complete boundaries and discard the other.
    - nyha_class must always reflect the class designation present in the final
      span value; span and nyha_class must never contradict each other.
    - Ensure the Roman numeral or class specification is present in the final span.
    - Do not apply modifier mapping by inference — the modifier must be explicitly
      present in the span text. If absent, use a single class label.
    - Occurrence indices are always counted against the final span value; if a
      span was corrected, count occurrences of the corrected text in the note.
    - If the same span text (case-insensitive) appears multiple times, assign
      each a separate RawMentionOutput with the correct occurrence index,
      counted independently per unique span text.
    - Do not assign temporality or classify context — that is handled downstream.
    - Do not hallucinate spans not present in the note.
    - Return an empty list if no valid NYHA mentions are present.
    """

    deidentified_text: str = dspy.InputField(
        desc="The full deidentified clinical note. All NYHA mentions must be extracted verbatim."
    )
    preextracted_spans: str = dspy.InputField(
        desc=(
            "A JSON list of candidate spans from rule-based pre-processing. Each entry "
            'has the form {"span": "NYHA II", "nyha_class": "II", "occurrence": 1}. '
            "Verify each: include only genuine NYHA functional class mentions. Spans with "
            "imprecise boundaries should be corrected to the verbatim substring from the "
            "note that best captures the full NYHA mention including its class designation "
            "and any modifier suffix or qualifier phrase. Boundary correction MAY extend "
            "beyond the original span boundaries to recover a clearly associated Roman "
            "numeral from the immediately surrounding context. The corrected value replaces "
            "the pre-extracted one in the output. Discard only when no Roman numeral or "
            "class designation is recoverable from local context at all. "
            "If this list is empty, scan the full note independently."
        )
    )
    raw_spans: List[RawMentionOutput] = dspy.OutputField(
        desc=(
            "All valid NYHA functional class mentions found in the note. Each entry "
            "corresponds to exactly one occurrence of one span. Pre-extracted spans may "
            "have their boundaries corrected; the final span value is always a verbatim "
            "substring of the note. nyha_class must always match the class designation "
            "(including any modifier mapping) in the final span. Occurrence is always "
            "counted against the final span value."
        )
    )
