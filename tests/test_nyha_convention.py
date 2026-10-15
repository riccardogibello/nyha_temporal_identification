"""Tests for the canonical NYHA class/vector convention.

See docs/spec/nyha_class_vector_convention.md. Covers the two canonical maps
(NYHA_MAP, LABEL_TO_VECTOR), the vector_key helper, the round-trip invariants of
spec §3, and the NYHAMention derive/validate behaviour for broad ranges and
invalid vectors.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from nyha_temporal_identification.data_model.constants import (
    LABEL_TO_VECTOR,
    NYHA_MAP,
    vector_key,
)
from nyha_temporal_identification.data_model.dspy_model import NYHAMention

# The 9-value label taxonomy (matches the NYHAMention.nyha_class Literal).
ALL_LABELS = [
    "I",
    "II",
    "III",
    "IV",
    "I-II",
    "II-III",
    "III-IV",
    "I-IV",
    "II-IV",
]

# The 7 valid vectors (single classes + adjacent pairs).
VALID_VECTORS = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
]

# canonical(L): broad ranges collapse to their lower class; others are identity.
CANONICAL = {
    "I-IV": "I",
    "II-IV": "II",
}


def _canonical(label: str) -> str:
    return CANONICAL.get(label, label)


def _mention(nyha_class: str = "I", nyha_vector=None) -> NYHAMention:
    kwargs = {
        "char_start": 0,
        "char_end": 10,
        "span": "NYHA xx",
        "nyha_class": nyha_class,
    }
    if nyha_vector is not None:
        kwargs["nyha_vector"] = nyha_vector
    return NYHAMention(**kwargs)


# ── Canonical constants ───────────────────────────────────────────────────────


def test_nyha_map_has_exactly_seven_keys():
    assert set(NYHA_MAP) == {"1", "2", "3", "4", "1_2", "2_3", "3_4"}
    assert "1_1" not in NYHA_MAP
    assert "2_2" not in NYHA_MAP


def test_nyha_map_values_are_seven_primary_labels():
    assert set(NYHA_MAP.values()) == {
        "I",
        "II",
        "III",
        "IV",
        "I-II",
        "II-III",
        "III-IV",
    }


def test_label_to_vector_total_over_nine_labels():
    assert set(LABEL_TO_VECTOR) == set(ALL_LABELS)


def test_broad_range_labels_collapse():
    assert LABEL_TO_VECTOR["I-IV"] == [1, 0, 0, 0]
    assert LABEL_TO_VECTOR["II-IV"] == [0, 1, 0, 0]


@pytest.mark.parametrize(
    "vec,key",
    [
        ([1, 0, 0, 0], "1"),
        ([0, 1, 0, 0], "2"),
        ([0, 1, 1, 0], "2_3"),
        ([0, 0, 1, 1], "3_4"),
        ([0, 0, 0, 0], ""),
        ([1, 0, 1, 0], "1_3"),
    ],
)
def test_vector_key(vec, key):
    assert vector_key(vec) == key


# ── Round-trip invariants (spec §3) ───────────────────────────────────────────


def test_invariant_1_vector_validity():
    # key(v) in NYHA_MAP iff v is valid.
    for v in VALID_VECTORS:
        assert vector_key(v) in NYHA_MAP
    for bad in ([0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1]):
        assert vector_key(bad) not in NYHA_MAP


def test_invariant_2_forward_is_primary_label():
    for v in VALID_VECTORS:
        label = NYHA_MAP[vector_key(v)]
        # Forward always yields one of the 7 primary labels (never I-IV / II-IV).
        assert label not in ("I-IV", "II-IV")
        assert label == _mention(nyha_class=label, nyha_vector=v).get_nyha_class()


def test_invariant_3_backward_total():
    for label in ALL_LABELS:
        assert LABEL_TO_VECTOR[label] == LABEL_TO_VECTOR[label]  # present for all 9


def test_invariant_4_vector_label_vector_identity():
    for v in VALID_VECTORS:
        assert LABEL_TO_VECTOR[NYHA_MAP[vector_key(v)]] == v


def test_invariant_5_label_vector_label_canonicalises():
    for label in ALL_LABELS:
        assert NYHA_MAP[vector_key(LABEL_TO_VECTOR[label])] == _canonical(label)


# ── NYHAMention derive + validate ─────────────────────────────────────────────


def test_broad_range_label_derives_collapsed_vector():
    m = _mention(nyha_class="II-IV")
    assert m.nyha_vector == [0, 1, 0, 0]
    assert m.get_nyha_class() == "II"


def test_i_iv_label_derives_collapsed_vector():
    m = _mention(nyha_class="I-IV")
    assert m.nyha_vector == [1, 0, 0, 0]
    assert m.get_nyha_class() == "I"


def test_adjacent_pair_label_derives_range_vector():
    m = _mention(nyha_class="II-III")
    assert m.nyha_vector == [0, 1, 1, 0]
    assert m.get_nyha_class() == "II-III"


def test_validator_rejects_noncontiguous_vector():
    with pytest.raises(ValidationError):
        _mention(nyha_class="II", nyha_vector=[1, 0, 1, 0])


def test_validator_rejects_three_class_vector():
    with pytest.raises(ValidationError):
        _mention(nyha_class="II", nyha_vector=[0, 1, 1, 1])


def test_validator_rejects_empty_set_vector():
    with pytest.raises(ValidationError):
        _mention(nyha_class="II", nyha_vector=[0, 0, 0, 0])


def test_single_and_adjacent_vectors_accepted():
    for v in VALID_VECTORS:
        m = _mention(nyha_class=NYHA_MAP[vector_key(v)], nyha_vector=v)
        assert m.nyha_vector == v
