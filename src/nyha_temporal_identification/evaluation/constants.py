# ── Error type registry ───────────────────────────────────────────────────────

import re

ERROR_TYPES = {
    "false_negative": "Gold mention not found in predictions",
    "false_positive": "Predicted mention not in gold",
    "span_boundary_mismatch": "Model extracted the vicinity but span boundaries were off "
    "(e.g. label-value layout, class I dropped by pre-extractor)",
    "wrong_temporality": "Span + class matched but temporality wrong",
    "wrong_class": "Span matched but class wrong",
    "wrong_both": "Span matched but both class and temporality wrong",
    "implicit_no_date": "No temporal anchor AND no linked date available",
    "implicit_date_missed": "No anchor words but a linked date was present — "
    "date-linkage inference did not resolve temporality",
    "improvement_narrative": "Improvement narrative (from X to Y) ambiguity",
    "telegraphic_prose": "Abbreviated / telegraphic clinical style",
}

# wrong_both intentionally present in both — it is a joint failure.
CLASS_ERROR_TYPES = {
    "wrong_class",
    "wrong_both",
    "false_negative",
    "false_positive",
    "span_boundary_mismatch",
    "improvement_narrative",
    "telegraphic_prose",
}
TEMP_ERROR_TYPES = {
    "wrong_temporality",
    "wrong_both",
    "implicit_no_date",
    "implicit_date_missed",
}

TELEGRAPHIC_DENSITY: float = 0.07

IMPROVEMENT_RE = re.compile(
    r"("
    r"\bfrom\b.{0,20}(class\s*)?(I{1,3}V?|IV).{0,20}\bto\b"
    r"|"
    r"\bvan\b.{0,20}(klasse\s*)?(I{1,3}V?|IV).{0,20}\bnaar\b"
    r"|"
    r"\b(verbeterd|verslechterd)\b.{0,30}(I{1,3}V?|IV)"
    r")",
    re.IGNORECASE,
)
