# Mapping from Arabic/mixed notation to canonical Roman numeral labels.
# Only the 7 digit strings spaCy can actually produce (single classes and
# adjacent pairs) are listed; broad ranges (1-4 / 2-4) collapse to a single
# class upstream, and 1-3 is never produced (see
# docs/spec/nyha_class_vector_convention.md).
NYHA_CLASS_NORMALISATION = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "1-2": "I-II",
    "2-3": "II-III",
    "3-4": "III-IV",
}


def normalise_nyha_class(raw: str) -> str:
    """Normalise Arabic or mixed NYHA class notation to canonical Roman numerals."""
    return NYHA_CLASS_NORMALISATION.get(raw.strip(), raw.strip())
