import json
import logging
import os
import random
from typing import List, Optional

import polars as pl

from nyha_temporal_identification.constants import (
    ALL_NOTES_METADATA_PARQUET_FILE_NAME,
    AUTHORED_COLUMN,
    CLEANED_TABLES_FOLDER_NAME,
    DEBUG,
    DEIDENTIFIED_TEXT_COLUMN,
    EXPORTED_ANNOTATIONS_FOLDER_NAME,
    EXTRACTED_NYHA_MENTIONS_KEY,
    FORMATTED_FOLDER_NAME,
    FREE_TEXT_FOLDER_NAME,
    ID_COLUMN,
    NOTES_DF_FIRST_FILTER_FILE_NAME,
    OLD_ID_COLUMN,
    OLD_TABLES_FOLDER_NAME,
    PATIENT_CONTACT_FOLDER_NAME,
    PATIENT_CONTACT_ID_COLUMN,
    PSEUDO_PATIENT_ID_COLUMN,
)
from nyha_temporal_identification.data_handling.annotation_formatting import (
    format_annotated_dataset,
)
from nyha_temporal_identification.data_handling.utils import remove_extra_text
from nyha_temporal_identification.data_model.dspy_model import (
    NYHAMention,
    NYHANote,
    nyha_mention_from_json_dict,
)
from nyha_temporal_identification.extraction.note import load_note_contact_mapping


def load_dataset(
    data_folder_path: str,
    whitelisted_note_ids: Optional[list[str]] = None,
    is_inference: bool = False,
) -> List[NYHANote]:
    print(f"\n[INFO] Loading dataset from: {data_folder_path}")
    # Firstly format all the annotated data into a consistent format to be fed to the pipeline
    format_annotated_dataset(
        data_folder_path,
    )
    if is_inference:
        base_json_folder_path = os.path.join(
            data_folder_path,
            OLD_TABLES_FOLDER_NAME,
            FREE_TEXT_FOLDER_NAME,
        )
    else:
        base_json_folder_path = os.path.join(
            data_folder_path,
            EXPORTED_ANNOTATIONS_FOLDER_NAME,
            FORMATTED_FOLDER_NAME,
        )
    # Verify the folder exists
    assert os.path.exists(base_json_folder_path), (
        f"Folder not found: {base_json_folder_path}"
    )
    json_file_names = [
        f
        for f in os.listdir(base_json_folder_path)
        if f.endswith(".json")
        and (FORMATTED_FOLDER_NAME in base_json_folder_path or f.startswith("DT4H_"))
    ]
    assert json_file_names, f"No JSON files found in folder: {base_json_folder_path}"

    whitelisted_note_ids_set = (
        set(whitelisted_note_ids) if whitelisted_note_ids is not None else None
    )

    documents: List[NYHANote] = []
    for file_name in json_file_names:
        file_path = os.path.join(base_json_folder_path, file_name)
        with open(file_path) as f:
            raw = json.load(f)

        for old_note_id, note_map in raw.items():
            if (
                whitelisted_note_ids_set is not None
                and old_note_id not in whitelisted_note_ids_set
            ):
                continue
            note_text = remove_extra_text(
                note_map.get(
                    DEIDENTIFIED_TEXT_COLUMN,
                    "",
                ),
            )
            mentions: List[NYHAMention] = []
            for m in note_map.get(EXTRACTED_NYHA_MENTIONS_KEY, []):
                try:
                    mention = nyha_mention_from_json_dict(m)
                    if not mention.validate_against_text(
                        deidentified_text=note_text,
                    ):
                        print(
                            f"[WARN] doc={old_note_id}: span mismatch — skipping mention"
                        )
                        continue
                    mentions.append(mention)
                except Exception as e:
                    print(f"[WARN] doc={old_note_id}: invalid mention ({e}) — skipping")

            documents.append(
                NYHANote(
                    old_note_id=old_note_id,
                    deidentified_text=note_text,
                    mentions=mentions,
                    authored_datetime=note_map.get(AUTHORED_COLUMN, None),
                )
            )

    if DEBUG:
        # Clamp to a smaller number of documents for faster debugging
        documents = documents[:10]
        logging.warning(
            f"DEBUG mode is ON: only the first {len(documents)} documents will be loaded"
        )

    print(
        f"\n  Loaded {len(documents)} documents "
        f"({sum(len(d.mentions) for d in documents)} total gold mentions)"
    )
    n_dated = sum(1 for d in documents if d.authored_datetime)
    print(f"  Notes with authored_datetime: {n_dated}/{len(documents)}")
    return documents


def load_whitelisted_note_ids(data_folder_path: str) -> Optional[list[str]]:
    """Load the note-ID whitelist used by ``annotate_training_nyha_mentions``.

    Returns ``None`` (no whitelist: every extracted note is used) when the
    whitelist table produced by the internal upstream cleaning step is missing.
    """
    whitelist_path = os.path.join(
        data_folder_path,
        CLEANED_TABLES_FOLDER_NAME,
        NOTES_DF_FIRST_FILTER_FILE_NAME,
    )
    if not os.path.exists(whitelist_path):
        logging.warning(
            f"Note whitelist not found: {whitelist_path} — "
            "running inference on all extracted notes"
        )
        return None
    return pl.read_parquet(whitelist_path, columns=[OLD_ID_COLUMN])[
        OLD_ID_COLUMN
    ].to_list()


def _groups_from_pseudo_patient(
    metadata_path: str,
    patient_contact_folder_path: str,
    note_ids: list[str],
) -> dict[str, str]:
    """Resolve ``old_note_id → patient_contact_id → pseudo_patient_id``."""
    metadata = (
        pl.scan_parquet(metadata_path)
        .select([OLD_ID_COLUMN, PATIENT_CONTACT_ID_COLUMN])
        .with_columns(pl.col(OLD_ID_COLUMN).cast(pl.Utf8))
        .filter(pl.col(OLD_ID_COLUMN).is_in(note_ids))
        .collect()
    )

    contact_ids = (
        metadata[PATIENT_CONTACT_ID_COLUMN].drop_nulls().unique().to_list()
    )

    patient_contact = (
        pl.scan_parquet(os.path.join(patient_contact_folder_path, "*.parquet"))
        .select([ID_COLUMN, PSEUDO_PATIENT_ID_COLUMN])
        .filter(pl.col(ID_COLUMN).is_in(contact_ids))
        .collect()
    )

    joined = metadata.join(
        patient_contact,
        left_on=PATIENT_CONTACT_ID_COLUMN,
        right_on=ID_COLUMN,
        how="left",
    )

    note_to_group: dict[str, str] = {}
    for row in joined.iter_rows(named=True):
        note_id = row[OLD_ID_COLUMN]
        pseudo_id = row.get(PSEUDO_PATIENT_ID_COLUMN)
        if pseudo_id is not None:
            note_to_group[note_id] = str(pseudo_id)
    return note_to_group


def _groups_from_raw_contact(
    data_folder_path: str,
    note_ids: list[str],
) -> dict[str, str]:
    """Group notes by the raw ``PatientContactId`` of the old_tables mapping.

    A note mapped to several contacts takes the smallest contact ID, so the
    grouping is deterministic. Keys are prefixed with ``contact:`` so they never
    collide with the note-ID singleton keys.
    """
    mapping = load_note_contact_mapping(data_folder_path)
    if mapping.is_empty():
        logging.warning(
            "Note → patient-contact mapping not found or empty — "
            "every note becomes its own split group"
        )
        return {}
    mapping = (
        mapping.filter(pl.col("PatientNotitieId").is_in(note_ids))
        .group_by("PatientNotitieId")
        .agg(pl.col("PatientContactId").min())
    )
    return {
        row["PatientNotitieId"]: f"contact:{row['PatientContactId']}"
        for row in mapping.iter_rows(named=True)
    }


def load_note_patient_groups(
    data_folder_path: str,
    note_ids: list[str],
) -> dict[str, str]:
    """Map each note ID to a split group ID.

    Groups by ``pseudo_patient_id`` when the patient-contact table from the
    internal cleaning step is available; otherwise falls back to the raw
    ``PatientContactId`` of the old_tables note → contact mapping. Unmapped
    notes fall back to their own note_id as a singleton group so they are never
    dropped from the split.
    """
    metadata_path = os.path.join(
        data_folder_path,
        OLD_TABLES_FOLDER_NAME,
        FREE_TEXT_FOLDER_NAME,
        ALL_NOTES_METADATA_PARQUET_FILE_NAME,
    )
    patient_contact_folder_path = os.path.join(
        data_folder_path,
        CLEANED_TABLES_FOLDER_NAME,
        PATIENT_CONTACT_FOLDER_NAME,
    )
    has_patient_contact = os.path.isdir(patient_contact_folder_path) and any(
        f.endswith(".parquet") for f in os.listdir(patient_contact_folder_path)
    )

    if os.path.exists(metadata_path) and has_patient_contact:
        note_to_group = _groups_from_pseudo_patient(
            metadata_path, patient_contact_folder_path, note_ids
        )
    else:
        missing = [
            path
            for path, present in (
                (metadata_path, os.path.exists(metadata_path)),
                (patient_contact_folder_path, has_patient_contact),
            )
            if not present
        ]
        logging.warning(
            f"Patient grouping input not found: {', '.join(missing)} — "
            "grouping notes by raw patient contact instead of by patient"
        )
        note_to_group = _groups_from_raw_contact(data_folder_path, note_ids)

    for note_id in note_ids:
        if note_id not in note_to_group:
            note_to_group[note_id] = note_id

    return note_to_group


def patient_grouped_split(
    documents: List[NYHANote],
    note_to_group: dict[str, str],
    *,
    seed: int,
    test_frac: float = 0.30,
    train_frac_of_remainder: float = 0.20,
) -> tuple[List[NYHANote], List[NYHANote], List[NYHANote]]:
    """Patient-grouped deterministic train/val/test split.

    Whole patient groups are kept together; fractions are document-count
    targets approached greedily.
    """
    from collections import defaultdict

    groups: dict[str, List[NYHANote]] = defaultdict(list)
    for doc in documents:
        groups[note_to_group[doc.old_note_id]].append(doc)

    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)

    total = len(documents)
    n_test = round(total * test_frac)
    n_train = round((total - n_test) * train_frac_of_remainder)

    def _fill(keys, target):
        bucket = []
        acc = 0
        n_consumed = 0
        for key in keys:
            size = len(groups[key])
            if not bucket or abs(acc + size - target) <= abs(acc - target):
                bucket.extend(groups[key])
                acc += size
                n_consumed += 1
            else:
                break
        return bucket, n_consumed

    test_docs, n_test_groups = _fill(group_keys, n_test)
    remaining_keys = group_keys[n_test_groups:]

    train_docs, n_train_groups = _fill(remaining_keys, n_train)
    val_keys = remaining_keys[n_train_groups:]
    val_docs = [doc for key in val_keys for doc in groups[key]]

    return train_docs, val_docs, test_docs
