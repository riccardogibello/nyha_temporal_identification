"""Tests for data_handling/data_loading.py — patient_grouped_split,
load_note_patient_groups (incl. missing-table fallbacks), load_whitelisted_note_ids."""

import logging

import polars as pl
import pytest

from nyha_temporal_identification.constants import (
    ALL_NOTES_METADATA_PARQUET_FILE_NAME,
    CLEANED_TABLES_FOLDER_NAME,
    CONTACT_NOTE_TABLE_NAME,
    FREE_TEXT_FOLDER_NAME,
    ID_COLUMN,
    NOTES_DF_FIRST_FILTER_FILE_NAME,
    OLD_ID_COLUMN,
    OLD_TABLES_FOLDER_NAME,
    PATIENT_CONTACT_FOLDER_NAME,
    PATIENT_CONTACT_ID_COLUMN,
    PSEUDO_PATIENT_ID_COLUMN,
)
from nyha_temporal_identification.data_handling.data_loading import (
    load_note_patient_groups,
    load_whitelisted_note_ids,
    patient_grouped_split,
)
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


# ── load_note_patient_groups / load_whitelisted_note_ids ──────────────────────


def _write_contact_mapping(tmp_path, rows: dict[int, int]) -> None:
    old_tables = tmp_path / OLD_TABLES_FOLDER_NAME
    old_tables.mkdir(exist_ok=True)
    pl.DataFrame(
        {
            "PatientNotitieId": list(rows.keys()),
            "PatientContactId": list(rows.values()),
        }
    ).write_parquet(old_tables / CONTACT_NOTE_TABLE_NAME)


def _write_pseudo_patient_inputs(tmp_path) -> None:
    free_text = tmp_path / OLD_TABLES_FOLDER_NAME / FREE_TEXT_FOLDER_NAME
    free_text.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {OLD_ID_COLUMN: ["1", "2", "3"], PATIENT_CONTACT_ID_COLUMN: [100, 200, 300]}
    ).write_parquet(free_text / ALL_NOTES_METADATA_PARQUET_FILE_NAME)
    patient_contact = tmp_path / CLEANED_TABLES_FOLDER_NAME / PATIENT_CONTACT_FOLDER_NAME
    patient_contact.mkdir(parents=True)
    # Contacts 100 and 200 belong to the same patient
    pl.DataFrame(
        {ID_COLUMN: [100, 200, 300], PSEUDO_PATIENT_ID_COLUMN: [7, 7, 8]}
    ).write_parquet(patient_contact / "part-0.parquet")


def test_patient_groups_use_pseudo_patient_when_available(tmp_path):
    _write_pseudo_patient_inputs(tmp_path)
    groups = load_note_patient_groups(str(tmp_path), ["1", "2", "3", "4"])
    assert groups == {"1": "7", "2": "7", "3": "8", "4": "4"}


def test_patient_groups_fall_back_to_raw_contact(tmp_path, caplog):
    # Note 2 maps to two contacts: the smallest one wins
    old_tables = tmp_path / OLD_TABLES_FOLDER_NAME
    old_tables.mkdir()
    pl.DataFrame(
        {
            "PatientNotitieId": [1, 2, 2, 3],
            "PatientContactId": [10, 30, 10, 20],
        }
    ).write_parquet(old_tables / CONTACT_NOTE_TABLE_NAME)
    with caplog.at_level(logging.WARNING):
        groups = load_note_patient_groups(str(tmp_path), ["1", "2", "3", "4"])
    assert groups == {
        "1": "contact:10",
        "2": "contact:10",
        "3": "contact:20",
        "4": "4",
    }
    assert "grouping notes by raw patient contact" in caplog.text


def test_patient_groups_fall_back_when_patient_contact_dir_empty(tmp_path):
    _write_contact_mapping(tmp_path, {1: 10, 2: 10})
    (tmp_path / CLEANED_TABLES_FOLDER_NAME / PATIENT_CONTACT_FOLDER_NAME).mkdir(
        parents=True
    )
    groups = load_note_patient_groups(str(tmp_path), ["1", "2"])
    assert groups == {"1": "contact:10", "2": "contact:10"}


def test_patient_groups_all_singletons_without_any_mapping(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        groups = load_note_patient_groups(str(tmp_path), ["1", "2"])
    assert groups == {"1": "1", "2": "2"}
    assert "every note becomes its own split group" in caplog.text


def test_whitelisted_note_ids_loaded(tmp_path):
    cleaned = tmp_path / CLEANED_TABLES_FOLDER_NAME
    cleaned.mkdir()
    pl.DataFrame({OLD_ID_COLUMN: ["1", "5"], "other": [0, 0]}).write_parquet(
        cleaned / NOTES_DF_FIRST_FILTER_FILE_NAME
    )
    assert load_whitelisted_note_ids(str(tmp_path)) == ["1", "5"]


def test_whitelisted_note_ids_missing_returns_none(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert load_whitelisted_note_ids(str(tmp_path)) is None
    assert NOTES_DF_FIRST_FILTER_FILE_NAME in caplog.text
