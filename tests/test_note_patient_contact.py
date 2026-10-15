"""Tests for extraction/note.py — integrate_patient_contact_information when the
optional cleaned_tables/* inputs are present, partially present, or missing."""

import logging
import os

import polars as pl
import pytest

from nyha_temporal_identification.constants import (
    CLEANED_TABLES_FOLDER_NAME,
    CONTACT_NOTE_TABLE_NAME,
    HAS_SPECIALTY_FOLDER_NAME,
    HAS_SPECIALTY_ID_COLUMN,
    ID_COLUMN,
    OLD_PATIENT_CONTACT_ID_COLUMN,
    OLD_TABLES_FOLDER_NAME,
    PATIENT_CONTACT_FOLDER_NAME,
    PATIENT_CONTACT_ID_COLUMN,
    PSEUDO_PATIENT_ID_COLUMN,
    SPECIALTY_FOLDER_NAME,
    SPECIALTY_ID_COLUMN,
    SPECIALTY_NAME_COLUMN,
)
from nyha_temporal_identification.extraction import note as note_module
from nyha_temporal_identification.extraction.note import (
    integrate_patient_contact_information,
    load_note_contact_mapping,
)


@pytest.fixture(autouse=True)
def _reset_warned_tables(monkeypatch):
    monkeypatch.setattr(note_module, "_warned_missing_tables", set())


def _write_table(data_folder, table_name, df: pl.DataFrame) -> None:
    folder = os.path.join(data_folder, CLEANED_TABLES_FOLDER_NAME, table_name)
    os.makedirs(folder, exist_ok=True)
    df.write_parquet(os.path.join(folder, "part-0.parquet"))


def _build_data_folder(tmp_path, patient_contact=True, specialty=True) -> str:
    data_folder = str(tmp_path)
    old_tables = tmp_path / OLD_TABLES_FOLDER_NAME
    old_tables.mkdir()
    pl.DataFrame(
        {
            "PatientNotitieId": [1, 2, 3, None],
            "PatientContactId": [10, 10, 20, 30],
        }
    ).write_parquet(old_tables / CONTACT_NOTE_TABLE_NAME)

    if patient_contact:
        _write_table(
            data_folder,
            PATIENT_CONTACT_FOLDER_NAME,
            pl.DataFrame(
                {
                    ID_COLUMN: [100, 200],
                    OLD_PATIENT_CONTACT_ID_COLUMN: ["10", "20"],
                    HAS_SPECIALTY_ID_COLUMN: [1000, 2000],
                    PSEUDO_PATIENT_ID_COLUMN: [7, 8],
                }
            ),
        )
    if specialty:
        _write_table(
            data_folder,
            HAS_SPECIALTY_FOLDER_NAME,
            pl.DataFrame({ID_COLUMN: [1000, 2000], SPECIALTY_ID_COLUMN: [5, 6]}),
        )
        _write_table(
            data_folder,
            SPECIALTY_FOLDER_NAME,
            pl.DataFrame(
                {ID_COLUMN: [5, 6], SPECIALTY_NAME_COLUMN: ["Cardiologie", "Interne"]}
            ),
        )
    return data_folder


def _notes() -> pl.DataFrame:
    # Note 4 has no contact mapping and must be dropped by the inner join
    return pl.DataFrame({ID_COLUMN: [1, 2, 3, 4], "text": ["a", "b", "c", "d"]})


def _by_id(df: pl.DataFrame) -> dict:
    return {row[ID_COLUMN]: row for row in df.iter_rows(named=True)}


def test_load_note_contact_mapping_cleans_ids(tmp_path):
    mapping = load_note_contact_mapping(_build_data_folder(tmp_path))
    assert mapping.schema["PatientNotitieId"] == pl.Utf8
    assert mapping.schema["PatientContactId"] == pl.Utf8
    assert mapping["PatientNotitieId"].to_list() == ["1", "2", "3"]


def test_load_note_contact_mapping_missing_returns_empty(tmp_path):
    mapping = load_note_contact_mapping(str(tmp_path))
    assert mapping.is_empty()
    assert mapping.columns == ["PatientNotitieId", "PatientContactId"]


def test_all_tables_present_joins_specialty(tmp_path):
    out = integrate_patient_contact_information(_build_data_folder(tmp_path), _notes())
    rows = _by_id(out)
    assert set(rows) == {"1", "2", "3"}
    assert rows["1"][PATIENT_CONTACT_ID_COLUMN] == 100
    assert rows["3"][PATIENT_CONTACT_ID_COLUMN] == 200
    assert rows["1"][SPECIALTY_NAME_COLUMN] == "Cardiologie"
    assert rows["3"][SPECIALTY_NAME_COLUMN] == "Interne"
    assert "PatientContactId" not in out.columns


def test_patient_contact_missing_yields_null_columns(tmp_path, caplog):
    data_folder = _build_data_folder(tmp_path, patient_contact=False, specialty=False)
    with caplog.at_level(logging.WARNING):
        out = integrate_patient_contact_information(data_folder, _notes())
    assert set(_by_id(out)) == {"1", "2", "3"}
    for col in (
        PATIENT_CONTACT_ID_COLUMN,
        HAS_SPECIALTY_ID_COLUMN,
        SPECIALTY_ID_COLUMN,
        SPECIALTY_NAME_COLUMN,
    ):
        assert out[col].null_count() == out.height
    assert "PatientContactId" not in out.columns
    assert PATIENT_CONTACT_FOLDER_NAME in caplog.text


def test_specialty_tables_missing_keeps_patient_contact(tmp_path, caplog):
    data_folder = _build_data_folder(tmp_path, specialty=False)
    with caplog.at_level(logging.WARNING):
        out = integrate_patient_contact_information(data_folder, _notes())
    rows = _by_id(out)
    assert rows["1"][PATIENT_CONTACT_ID_COLUMN] == 100
    assert out[SPECIALTY_ID_COLUMN].null_count() == out.height
    assert out[SPECIALTY_NAME_COLUMN].null_count() == out.height
    assert HAS_SPECIALTY_FOLDER_NAME in caplog.text
    assert SPECIALTY_FOLDER_NAME in caplog.text


def test_missing_table_warning_logged_once(tmp_path, caplog):
    # Called once per DT4H source file: the warning must not repeat every time
    data_folder = _build_data_folder(tmp_path, patient_contact=False, specialty=False)
    with caplog.at_level(logging.WARNING):
        integrate_patient_contact_information(data_folder, _notes())
        integrate_patient_contact_information(data_folder, _notes())
    assert caplog.text.count("Optional table not found") == 1
