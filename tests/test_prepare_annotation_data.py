"""Tests for scripts/prepare_annotation_data.py — annotation JSON construction,
ID splitting, new-annotator assignment, note-map I/O, and the resume-mode
end-to-end flow. No model calls; synthetic notes and tmp data folders.

The critical regression test is ``test_resume_same_folder_preserves_existing``:
it guards against the resume mode wiping its own input when
``--old-annotations-folder`` points at the (default) output folder.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd

from nyha_temporal_identification.constants import (
    ALL_NOTES_METADATA_PARQUET_FILE_NAME,
    AUTHORED_COLUMN,
    FREE_TEXT_FOLDER_NAME,
    HAS_INVALID_NYHA_MENTION_COLUMN,
    HAS_NYHA_MENTION_COLUMN,
    INPUT_ANNOTATIONS_FOLDER_NAME,
    OLD_ID_COLUMN,
    OLD_TABLES_FOLDER_NAME,
    PATIENT_CONTACT_ID_COLUMN,
    SPECIALTY_NAME_COLUMN,
    TYPE_COLUMN,
)

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_annotation_data.py"
)
_spec = importlib.util.spec_from_file_location("prepare_annotation_data", _SCRIPT)
pad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pad)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _date(text, start, end):
    return {"text": text, "original_start": start, "original_end": end}


def _mention(classes, text, start, end, linked_date=None):
    return {
        "classes": classes,
        "original_text": text,
        "original_start": start,
        "original_end": end,
        "linked_date": linked_date,
    }


def _note(text="patient text", dates=None, mentions=None, contact_id=7):
    return {
        "deidentified_text": text,
        PATIENT_CONTACT_ID_COLUMN: contact_id,
        TYPE_COLUMN: "Brief",
        AUTHORED_COLUMN: "2024-01-01",
        "extracted_dates": dates or [],
        "extracted_nyha_mentions": mentions or [],
    }


def _entities_by_id(result):
    return {e["id"]: e for e in result if e["type"] == "labels"}


def _relations(result):
    return [e for e in result if e["type"] == "relation"]


# ── create_annotation_json ────────────────────────────────────────────────────


class TestCreateAnnotationJson:
    def test_sentinel_span_offsets_index_into_text(self):
        note = _note(text="line one. NYHA II.")
        out = pad.create_annotation_json(note)
        text = out["data"]["text"]
        ents = _entities_by_id(out["predictions"][0]["result"])

        for eid, expected in [
            ("date_0", "NO_LINKED_DATE"),
            ("nyha_type_0", "HISTORICAL_DATE"),
            ("nyha_type_1", "CURRENT_DATE"),
        ]:
            ent = ents[eid]["value"]
            assert text[ent["start"] : ent["end"]] == expected
            assert ent["text"] == expected

    def test_text_starts_with_deidentified_text(self):
        note = _note(text="hello world")
        out = pad.create_annotation_json(note)
        assert out["data"]["text"].startswith("hello world")

    def test_extracted_date_becomes_entity(self):
        note = _note(dates=[_date("01-01-2020", 0, 10)])
        out = pad.create_annotation_json(note)
        ents = _entities_by_id(out["predictions"][0]["result"])
        # date_0 is the sentinel; the extracted one is date_3.
        date_entities = [
            e for k, e in ents.items() if k.startswith("date_") and k != "date_0"
        ]
        assert len(date_entities) == 1
        assert date_entities[0]["value"]["text"] == "01-01-2020"

    def test_mention_label_uses_classes(self):
        note = _note(mentions=[_mention([2, 3], "NYHA II-III", 0, 11)])
        out = pad.create_annotation_json(note)
        ents = _entities_by_id(out["predictions"][0]["result"])
        nyha = [e for k, e in ents.items() if k.startswith("nyha_") and "type" not in k]
        assert nyha[0]["value"]["labels"] == ["nyha class 2-3"]

    def test_unlinked_mention_links_to_date_0(self):
        note = _note(mentions=[_mention([2], "NYHA II", 0, 7, linked_date=None)])
        out = pad.create_annotation_json(note)
        rels = _relations(out["predictions"][0]["result"])
        assert len(rels) == 1
        assert rels[0]["to_id"] == "date_0"
        assert rels[0]["labels"] == ["does not have date"]

    def test_linked_mention_reuses_existing_date_entity(self):
        # The same date span appears both in extracted_dates and as the mention's
        # linked_date; it must NOT be duplicated, and the relation must target it.
        span = _date("01-01-2020", 100, 110)
        note = _note(
            dates=[span],
            mentions=[_mention([3], "NYHA III", 0, 8, linked_date=dict(span))],
        )
        out = pad.create_annotation_json(note)
        result = out["predictions"][0]["result"]
        ents = _entities_by_id(result)
        # Exactly one non-sentinel date entity for span (100, 110).
        span_dates = [
            e
            for k, e in ents.items()
            if k.startswith("date_")
            and k != "date_0"
            and (e["value"]["start"], e["value"]["end"]) == (100, 110)
        ]
        assert len(span_dates) == 1
        rels = _relations(result)
        assert rels[0]["labels"] == ["has date"]
        assert rels[0]["to_id"] == span_dates[0]["id"]


# ── split_ids_with_overlap ────────────────────────────────────────────────────


class TestSplitIdsWithOverlap:
    def test_overlap_shared_and_chunks_disjoint(self):
        pad.random.seed(1)
        ids = list(range(100))
        names = ["a", "b", "c"]
        assignments = pad.split_ids_with_overlap(
            ids, overlap_ratio=0.1, annotator_names=names
        )
        assert set(assignments) == set(names)
        overlap = set.intersection(*[set(v) for v in assignments.values()])
        assert len(overlap) == 10  # int(100 * 0.1)
        # Exclusive parts (everything beyond the shared overlap) are disjoint.
        exclusives = [set(v) - overlap for v in assignments.values()]
        for i in range(len(exclusives)):
            for j in range(i + 1, len(exclusives)):
                assert exclusives[i].isdisjoint(exclusives[j])
        # Union covers the whole pool.
        assert set().union(*[set(v) for v in assignments.values()]) == set(ids)

    def test_seeded_split_is_deterministic(self):
        ids = list(range(50))
        pad.random.seed(7)
        first = pad.split_ids_with_overlap(ids, annotator_names=["a", "b"])
        pad.random.seed(7)
        second = pad.split_ids_with_overlap(ids, annotator_names=["a", "b"])
        assert first == second


# ── assign_new_annotators ─────────────────────────────────────────────────────


class TestAssignNewAnnotators:
    def test_new_annotator_gets_overlap_plus_disjoint_extra(self):
        pad.random.seed(3)
        existing = {"alice": ["1", "2", "3"], "carol": ["2", "3", "4"]}
        selected = [str(i) for i in range(1, 21)]
        result = pad.assign_new_annotators(["bob"], existing, selected, target=6)
        bob = result["bob"]
        overlap = {"2", "3"}  # intersection of alice & carol
        assert overlap.issubset(set(bob))
        assert len(bob) == 6
        # Extras are drawn only from ids not already assigned to anyone.
        assigned = set(existing["alice"]) | set(existing["carol"])
        extras = set(bob) - overlap
        assert extras.isdisjoint(assigned)

    def test_multiple_new_annotators_draw_disjoint_extras(self):
        pad.random.seed(4)
        existing = {"alice": ["1", "2"]}
        selected = [str(i) for i in range(1, 31)]
        result = pad.assign_new_annotators(
            ["bob", "dan"], existing, selected, target=5
        )
        overlap = {"1", "2"}
        bob_extra = set(result["bob"]) - overlap
        dan_extra = set(result["dan"]) - overlap
        assert bob_extra.isdisjoint(dan_extra)


# ── note-map I/O ──────────────────────────────────────────────────────────────


class TestNoteMapIO:
    def test_load_map_and_build_data(self, tmp_path):
        f1 = tmp_path / "DT4H_batch1.json"
        f2 = tmp_path / "DT4H_batch2.json"
        f1.write_text(json.dumps({"1": _note(text="a"), "2": _note(text="b")}))
        f2.write_text(json.dumps({"3": _note(text="c")}))
        # A non-matching file must be ignored.
        (tmp_path / "other.json").write_text(json.dumps({"9": _note()}))

        id_to_file = pad.load_note_identifier_json_map(str(tmp_path))
        assert set(id_to_file) == {"1", "2", "3"}

        data = pad.build_note_data_map(["1", "3"], id_to_file)
        assert set(data) == {"1", "3"}
        assert data["1"]["deidentified_text"] == "a"

    def test_load_old_assignments_skips_missing(self, tmp_path):
        alice_dir = tmp_path / "alice" / INPUT_ANNOTATIONS_FOLDER_NAME
        alice_dir.mkdir(parents=True)
        (alice_dir / "10.json").write_text("{}")
        (alice_dir / "11.json").write_text("{}")
        result = pad.load_old_annotator_assignments(
            str(tmp_path), ["alice", "bob"]
        )
        assert set(result) == {"alice"}  # bob has no folder
        assert set(result["alice"]) == {"10", "11"}


# ── End-to-end resume mode (regression for the clear-before-read bug) ─────────


def _build_data_folder(tmp_path, n_notes=10, specialty="Cardiologie"):
    """Creates a minimal data folder: metadata parquet + DT4H note map."""
    free_text = tmp_path / OLD_TABLES_FOLDER_NAME / FREE_TEXT_FOLDER_NAME
    free_text.mkdir(parents=True)

    rows = []
    note_map = {}
    for i in range(1, n_notes + 1):
        rows.append(
            {
                OLD_ID_COLUMN: str(i),
                HAS_NYHA_MENTION_COLUMN: True,
                HAS_INVALID_NYHA_MENTION_COLUMN: False,
                SPECIALTY_NAME_COLUMN: specialty,
                TYPE_COLUMN: "Brief",
            }
        )
        note_map[str(i)] = _note(
            text=f"note {i} NYHA II",
            mentions=[_mention([2], "NYHA II", 7, 14)],
        )
    pd.DataFrame(rows).to_parquet(
        free_text / ALL_NOTES_METADATA_PARQUET_FILE_NAME
    )
    (free_text / "DT4H_batch.json").write_text(json.dumps(note_map))
    return str(tmp_path), str(free_text)


def _annotator_ids(output_folder, name):
    folder = os.path.join(output_folder, name, INPUT_ANNOTATIONS_FOLDER_NAME)
    return {f[:-5] for f in os.listdir(folder) if f.endswith(".json")}


def test_resume_same_folder_preserves_existing(tmp_path):
    data_folder, free_text = _build_data_folder(tmp_path, n_notes=10)
    output_folder = os.path.join(free_text, INPUT_ANNOTATIONS_FOLDER_NAME)

    # Pre-seed an existing assignment for alice (ids 1,2,3) at the output path.
    alice_dir = os.path.join(output_folder, "alice", INPUT_ANNOTATIONS_FOLDER_NAME)
    os.makedirs(alice_dir)
    for i in ("1", "2", "3"):
        with open(os.path.join(alice_dir, f"{i}.json"), "w") as f:
            f.write("{}")

    # Resume with --old-annotations-folder pointing at the SAME output folder.
    args = argparse.Namespace(
        data_folder=data_folder,
        annotator_names=["alice", "bob"],
        old_annotations_folder=output_folder,
        samples_per_annotator=5,
        overlap_ratio=0.10,
        seed=42,
    )
    pad.main(args)

    # alice's original three notes survived (were not wiped before being read).
    alice_ids = _annotator_ids(output_folder, "alice")
    assert alice_ids == {"1", "2", "3"}

    # bob is a new annotator: inherits alice's overlap (1,2,3) + extra to reach 5.
    bob_ids = _annotator_ids(output_folder, "bob")
    assert {"1", "2", "3"}.issubset(bob_ids)
    assert len(bob_ids) == 5
    assert (bob_ids - {"1", "2", "3"}).issubset({str(i) for i in range(4, 11)})


def test_fresh_split_runs_end_to_end(tmp_path):
    data_folder, free_text = _build_data_folder(tmp_path, n_notes=10)
    output_folder = os.path.join(free_text, INPUT_ANNOTATIONS_FOLDER_NAME)
    args = argparse.Namespace(
        data_folder=data_folder,
        annotator_names=["a", "b"],
        old_annotations_folder=None,
        samples_per_annotator=3,
        overlap_ratio=0.10,
        seed=42,
    )
    pad.main(args)
    a_ids = _annotator_ids(output_folder, "a")
    b_ids = _annotator_ids(output_folder, "b")
    assert a_ids and b_ids
    # Each annotator's zip archive was produced.
    assert os.path.exists(os.path.join(output_folder, "a_annotations.zip"))
    assert os.path.exists(os.path.join(output_folder, "b_annotations.zip"))


def test_missing_specialty_skips_specialty_filter(tmp_path, capsys):
    # Without the optional specialty tables, specialty_name is all-null; the
    # cardiology filter must be skipped rather than select nothing.
    data_folder, free_text = _build_data_folder(tmp_path, n_notes=10, specialty=None)
    output_folder = os.path.join(free_text, INPUT_ANNOTATIONS_FOLDER_NAME)
    args = argparse.Namespace(
        data_folder=data_folder,
        annotator_names=["a", "b"],
        old_annotations_folder=None,
        samples_per_annotator=3,
        overlap_ratio=0.10,
        seed=42,
    )
    pad.main(args)
    assert "skipping the Cardiologie specialty filter" in capsys.readouterr().out
    assert _annotator_ids(output_folder, "a")
    assert _annotator_ids(output_folder, "b")
