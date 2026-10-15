from typing import Literal

# Forward map: NYHA_MAP key (vector_key) -> canonical roman label.
# The set of keys is exactly the set of *valid* vectors: one single-class vector
# per class, plus one per adjacent pair. There is no key for broad ranges
# (I-IV / II-IV) — those collapse to a single class and reuse "1"/"2".
NYHA_MAP: dict[
    str, Literal["I", "I-II", "I-IV", "II", "II-III", "II-IV", "III", "III-IV", "IV"]
] = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "1_2": "I-II",
    "2_3": "II-III",
    "3_4": "III-IV",
}

# Backward map: label -> multi-hot vector. Total over the 9-value label taxonomy.
# This is the authority for "derive vector from label": it must NOT be
# reconstructed as {v: k for k, v in NYHA_MAP.items()} because that would lose
# the broad-range labels I-IV / II-IV, which collapse to class I / II.
LABEL_TO_VECTOR: dict[str, list[int]] = {
    "I": [1, 0, 0, 0],
    "II": [0, 1, 0, 0],
    "III": [0, 0, 1, 0],
    "IV": [0, 0, 0, 1],
    "I-II": [1, 1, 0, 0],
    "II-III": [0, 1, 1, 0],
    "III-IV": [0, 0, 1, 1],
    "I-IV": [1, 0, 0, 0],  # collapses to class I
    "II-IV": [0, 1, 0, 0],  # collapses to class II
}


def vector_key(vec: list[int]) -> str:
    """Return the NYHA_MAP key for a multi-hot vector ("_"-joined 1-based indices).

    E.g. [0, 1, 1, 0] -> "2_3", [0, 1, 0, 0] -> "2", [0, 0, 0, 0] -> "".
    The result is a NYHA_MAP key iff the vector is valid (a single class or an
    adjacent pair).
    """
    return "_".join(str(i + 1) for i, v in enumerate(vec) if v)
