"""Tests for data_handling/data_loading.py — patient_grouped_split."""

import pytest

from nyha_temporal_identification.data_handling.data_loading import patient_grouped_split
from nyha_temporal_identification.data_model.dspy_model import NYHANote


def _note(note_id: str) -> NYHANote:
    return NYHANote(old_note_id=note_id, deidentified_text="text", mentions=[])


def _make_docs(note_ids: list[str]) -> list[NYHANote]:
    return [_note(nid) for nid in note_ids]


def _group_map(assignments: dict[str, list[str]]) -> dict[str, str]:
    """Build note_to_group from {group_id: [note_id, ...]}."""
    mapping = {}
    for group_id, note_ids in assignments.items():
        for nid in note_ids:
            mapping[nid] = group_id
    return mapping


# ── basic correctness ──────────────────────────────────────────────────────────

def test_all_docs_present():
    docs = _make_docs([str(i) for i in range(30)])
    note_to_group = {d.old_note_id: d.old_note_id for d in docs}
    train, val, test = patient_grouped_split(docs, note_to_group, seed=0)
    assert len(train) + len(val) + len(test) == 30


def test_no_patient_leakage():
    # patient A has notes 0..9, patient B has notes 10..19
    assignments = {
        "patient_A": [str(i) for i in range(10)],
        "patient_B": [str(i) for i in range(10, 20)],
        "patient_C": [str(i) for i in range(20, 30)],
    }
    docs = _make_docs([nid for ids in assignments.values() for nid in ids])
    note_to_group = _group_map(assignments)

    train, val, test = patient_grouped_split(docs, note_to_group, seed=42)

    def groups_in(split):
        return {note_to_group[d.old_note_id] for d in split}

    g_train, g_val, g_test = groups_in(train), groups_in(val), groups_in(test)
    assert g_train.isdisjoint(g_test), "patient group leaked between train and test"
    assert g_train.isdisjoint(g_val), "patient group leaked between train and val"
    assert g_val.isdisjoint(g_test), "patient group leaked between val and test"


def test_deterministic_same_seed():
    docs = _make_docs([str(i) for i in range(40)])
    note_to_group = {d.old_note_id: str(int(d.old_note_id) // 4) for d in docs}

    r1 = patient_grouped_split(docs, note_to_group, seed=7)
    r2 = patient_grouped_split(docs, note_to_group, seed=7)
    for s1, s2 in zip(r1, r2):
        assert [d.old_note_id for d in s1] == [d.old_note_id for d in s2]


def test_different_seeds_differ():
    docs = _make_docs([str(i) for i in range(40)])
    note_to_group = {d.old_note_id: str(int(d.old_note_id) // 4) for d in docs}

    train_a, _, _ = patient_grouped_split(docs, note_to_group, seed=1)
    train_b, _, _ = patient_grouped_split(docs, note_to_group, seed=2)
    # With 10 groups and two different seeds, trains are very likely to differ.
    ids_a = [d.old_note_id for d in train_a]
    ids_b = [d.old_note_id for d in train_b]
    assert ids_a != ids_b


def test_approximate_split_sizes():
    # 100 singleton groups → exact arithmetic applies
    docs = _make_docs([str(i) for i in range(100)])
    note_to_group = {d.old_note_id: d.old_note_id for d in docs}

    train, val, test = patient_grouped_split(docs, note_to_group, seed=0)
    # test target = 30, train target = 14 (20% of 70)
    assert 20 <= len(test) <= 40, f"test size {len(test)} out of expected range"
    assert 8 <= len(train) <= 20, f"train size {len(train)} out of expected range"


# ── edge cases ─────────────────────────────────────────────────────────────────

def test_three_groups_all_splits_non_empty():
    assignments = {"g1": ["0", "1"], "g2": ["2", "3"], "g3": ["4", "5"]}
    docs = _make_docs(["0", "1", "2", "3", "4", "5"])
    note_to_group = _group_map(assignments)

    train, val, test = patient_grouped_split(docs, note_to_group, seed=0)
    assert len(test) > 0
    assert len(train) > 0
    assert len(val) > 0


def test_single_group_no_crash():
    docs = _make_docs(["0", "1", "2"])
    note_to_group = {"0": "g1", "1": "g1", "2": "g1"}

    train, val, test = patient_grouped_split(docs, note_to_group, seed=0)
    total = len(train) + len(val) + len(test)
    assert total == 3
    # Only one group exists — it goes to test; train and val must be empty
    non_empty = [s for s in (train, val, test) if s]
    assert len(non_empty) == 1


def test_singleton_groups_preserve_all_docs():
    note_ids = [str(i) for i in range(20)]
    docs = _make_docs(note_ids)
    note_to_group = {nid: nid for nid in note_ids}

    train, val, test = patient_grouped_split(docs, note_to_group, seed=0)
    all_ids = {d.old_note_id for d in train + val + test}
    assert all_ids == set(note_ids)
