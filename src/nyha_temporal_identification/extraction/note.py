import json
import os
import re
from collections import defaultdict
from typing import Tuple

import numpy as np
import polars as pl
from datasets import Dataset

from nyha_temporal_identification.constants import (
    ACTUAL_ID_COLUMN,
    AUTHORED_COLUMN,
    CLEANED_TABLES_FOLDER_NAME,
    CONTACT_NOTE_TABLE_NAME,
    DEIDENTIFIED_TEXT_COLUMN,
    EXTRACTED_DATES_KEY,
    EXTRACTED_INVALID_NYHA_MENTIONS_KEY,
    EXTRACTED_NYHA_MENTIONS_KEY,
    HAS_INVALID_NYHA_MENTION_COLUMN,
    HAS_NYHA_MENTION_COLUMN,
    HAS_SPECIALTY_FOLDER_NAME,
    HAS_SPECIALTY_ID_COLUMN,
    ID_COLUMN,
    NBSP,
    OLD_ID_COLUMN,
    OLD_PATIENT_CONTACT_ID_COLUMN,
    OLD_TABLES_FOLDER_NAME,
    PART_COLUMN,
    PATIENT_CONTACT_FOLDER_NAME,
    PATIENT_CONTACT_ID_COLUMN,
    SPECIALTY_FOLDER_NAME,
    SPECIALTY_ID_COLUMN,
    SPECIALTY_NAME_COLUMN,
    TYPE_COLUMN,
)
from nyha_temporal_identification.extraction.date_parsing import parse_flexible_date
from nyha_temporal_identification.utils.io import load_parquet_dataset_to_polars


def integrate_patient_contact_information(
    data_folder_path: str,
    note_df: pl.DataFrame,
) -> pl.DataFrame:
    note_id_patient_contact_df = load_parquet_dataset_to_polars(
        os.path.join(
            data_folder_path,
            OLD_TABLES_FOLDER_NAME,
            CONTACT_NOTE_TABLE_NAME,
        ),
        columns=["PatientNotitieId", "PatientContactId"],
    )
    note_id_patient_contact_df = note_id_patient_contact_df.drop_nulls(
        subset=["PatientNotitieId", "PatientContactId"]
    )
    # Cast PatientNotitieId and PatientContactId to string
    note_id_patient_contact_df = note_id_patient_contact_df.with_columns(
        pl.col("PatientNotitieId").cast(pl.Utf8),
        pl.col("PatientContactId").cast(pl.Utf8),
    )
    # Drop any row containing N/A strings in the PatientNotitieId or PatientContactId columns
    note_id_patient_contact_df = note_id_patient_contact_df.filter(
        ~(pl.col("PatientNotitieId").str.contains("N/A", literal=True))
        & ~(pl.col("PatientContactId").str.contains("N/A", literal=True))
    )
    # Cast the ID_COLUMN to string to match the PatientNotitieId column
    note_df = note_df.with_columns(pl.col(ID_COLUMN).cast(pl.Utf8))
    # Perform an inner join, so that we only keep the notes that have a corresponding
    # patient contact information
    note_df = note_df.join(
        note_id_patient_contact_df,
        left_on=ID_COLUMN,
        right_on="PatientNotitieId",
        how="inner",
    )
    # Get a list of the unique patient contact identifiers from the note dataframe
    unique_patient_contact_ids = note_df["PatientContactId"].unique().to_list()
    # Use this to load the patient contact information only for the relevant patient contacts,
    # to avoid memory issues
    patient_contact_df = load_parquet_dataset_to_polars(
        os.path.join(
            data_folder_path,
            CLEANED_TABLES_FOLDER_NAME,
            PATIENT_CONTACT_FOLDER_NAME,
        ),
        filters=[
            pl.col(OLD_PATIENT_CONTACT_ID_COLUMN).is_in(unique_patient_contact_ids)
        ],
    )
    has_specialty_df = load_parquet_dataset_to_polars(
        os.path.join(
            data_folder_path,
            CLEANED_TABLES_FOLDER_NAME,
            HAS_SPECIALTY_FOLDER_NAME,
        ),
    )
    specialty_df = load_parquet_dataset_to_polars(
        os.path.join(
            data_folder_path,
            CLEANED_TABLES_FOLDER_NAME,
            SPECIALTY_FOLDER_NAME,
        ),
    )

    # Join the table on the old patient contact and get the new patient contact ID
    note_df = note_df.join(
        patient_contact_df.select(
            [
                ID_COLUMN,
                OLD_PATIENT_CONTACT_ID_COLUMN,
                HAS_SPECIALTY_ID_COLUMN,
            ]
        ).rename(
            {
                ID_COLUMN: PATIENT_CONTACT_ID_COLUMN,
            }
        ),
        left_on="PatientContactId",
        right_on=OLD_PATIENT_CONTACT_ID_COLUMN,
        how="left",
    ).drop("PatientContactId")
    note_df = note_df.join(
        has_specialty_df.select(
            [
                ID_COLUMN,
                SPECIALTY_ID_COLUMN,
            ]
        ),
        left_on=HAS_SPECIALTY_ID_COLUMN,
        right_on=ID_COLUMN,
        how="left",
    ).join(
        specialty_df.select(
            [
                ID_COLUMN,
                SPECIALTY_NAME_COLUMN,
            ]
        ),
        left_on=SPECIALTY_ID_COLUMN,
        right_on=ID_COLUMN,
        how="left",
    )

    return note_df


def preprocess_notes(
    file_path: str,
    data_folder_path: str,
) -> pl.DataFrame:
    """
    This method loads all the clinical notes in the given parquet file, extracts the relevant columns,
    and aggregates them by the actual note ID, as they are divided into parts.

    Args:
        file_path (str): Path to the parquet file containing the clinical notes.
        data_folder_path (str): Path to the folder containing the data.
    Returns:
        pl.DataFrame: A DataFrame with the preprocessed notes.
    """
    note_df = load_parquet_dataset_to_polars(file_path)
    kept_columns = [
        OLD_ID_COLUMN,
        TYPE_COLUMN,
        AUTHORED_COLUMN,
        DEIDENTIFIED_TEXT_COLUMN,
    ]

    # NOTE: .unique() call does not preserve the order of the rows
    note_df = note_df.select(kept_columns).unique()
    note_df = note_df.with_columns(pl.col(OLD_ID_COLUMN).cast(pl.String))

    note_df = (
        note_df.with_columns(pl.col(OLD_ID_COLUMN).str.split("-").alias("id_split"))
        .with_columns(
            pl.col("id_split")
            .list.get(0, null_on_oob=True)
            .cast(pl.Int64, strict=False)
            .alias(ACTUAL_ID_COLUMN),
            pl.col("id_split")
            .list.get(1, null_on_oob=True)
            .cast(pl.Int64, strict=False)
            .alias(PART_COLUMN),
        )
        .drop("id_split")
    )

    note_df = note_df.drop_nulls(subset=[ACTUAL_ID_COLUMN, PART_COLUMN])
    note_df = note_df.sort([ACTUAL_ID_COLUMN, PART_COLUMN], descending=[False, False])

    note_df = note_df.group_by(ACTUAL_ID_COLUMN).agg(
        [
            pl.col(TYPE_COLUMN).first(),
            pl.col(AUTHORED_COLUMN).first(),
            pl.col(DEIDENTIFIED_TEXT_COLUMN).str.join(" "),
        ]
    )

    note_df = note_df.rename({ACTUAL_ID_COLUMN: ID_COLUMN})
    note_df = integrate_patient_contact_information(
        data_folder_path,
        note_df,
    )

    return note_df


def build_final_note_map(
    dataset: Dataset,
) -> Tuple[dict, list]:
    """
    Build a JSON map between note identifiers and all the information related to the note,
    including the extracted NYHA mentions and dates. Also return lightweight metadata rows
    for ALL notes (both with and without NYHA mentions) to populate the main notes dataset.

    Args:
        dataset (Dataset): A Hugging Face Dataset containing the clinical notes and their associated information.

    Returns:
        Tuple[dict, list]: A tuple containing:
            - json_map: dict keyed by note ID, containing text + mentions + dates for NYHA-positive notes only.
            - metadata_rows: list of dicts with (old_id, patient_contact_id, type, authored, has_nyha_mention)
              for every note in the dataset.
    """
    json_map = {}
    metadata_rows = []

    for row in dataset:
        row = dict(row)

        mentions = json.loads(row.get(EXTRACTED_NYHA_MENTIONS_KEY) or "[]")
        invalid_mentions = json.loads(
            row.get(EXTRACTED_INVALID_NYHA_MENTIONS_KEY) or "[]"
        )
        dates = json.loads(row.get(EXTRACTED_DATES_KEY) or "[]")
        original_id = row[ID_COLUMN]
        authored_raw = row.get(AUTHORED_COLUMN)
        authored_dt = parse_flexible_date(authored_raw) if authored_raw else None
        authored_iso = authored_dt.isoformat() if authored_dt else None

        has_invalid_mention = bool(invalid_mentions)
        has_mention = bool(mentions)
        metadata_rows.append(
            {
                OLD_ID_COLUMN: original_id,
                PATIENT_CONTACT_ID_COLUMN: row.get(PATIENT_CONTACT_ID_COLUMN),
                TYPE_COLUMN: row.get(TYPE_COLUMN),
                AUTHORED_COLUMN: authored_iso,
                SPECIALTY_NAME_COLUMN: row.get(SPECIALTY_NAME_COLUMN),
                HAS_NYHA_MENTION_COLUMN: has_mention,
                HAS_INVALID_NYHA_MENTION_COLUMN: has_invalid_mention,
                DEIDENTIFIED_TEXT_COLUMN: row.get(DEIDENTIFIED_TEXT_COLUMN),
            }
        )

        if has_mention or has_invalid_mention:
            json_map[original_id] = {
                ID_COLUMN: original_id,
                PATIENT_CONTACT_ID_COLUMN: row.get(PATIENT_CONTACT_ID_COLUMN),
                TYPE_COLUMN: row.get(TYPE_COLUMN),
                AUTHORED_COLUMN: authored_iso,
                DEIDENTIFIED_TEXT_COLUMN: row.get(DEIDENTIFIED_TEXT_COLUMN),
                EXTRACTED_NYHA_MENTIONS_KEY: mentions,
                EXTRACTED_INVALID_NYHA_MENTIONS_KEY: invalid_mentions,
                HAS_INVALID_NYHA_MENTION_COLUMN: has_invalid_mention,
                EXTRACTED_DATES_KEY: dates,
                SPECIALTY_NAME_COLUMN: row.get(SPECIALTY_NAME_COLUMN),
            }

    return json_map, metadata_rows


def structure_note(
    clinical_note_raw_text: str,
) -> str:
    """
    Structure the clinical note text by identifying segments based on sequences of multiple
    spaces and applying indentation to reflect the hierarchy of the segments. The method processes
    the text in multiple passes, starting with the longest sequences of spaces to determine the highest
    level of indentation, and then applying additional indentation for shorter sequences within those segments.
    Special characters (NBSP) are used to ensure that the indentation is preserved when rendered in environments
    that may collapse regular spaces. The final output is a structured version of the clinical note with appropriate
    line breaks and indentation to enhance readability.

    Args:
        clinical_note_raw_text (str): The raw text of the clinical note that needs to be structured.

    Returns:
        str: A structured version of the clinical note with segments separated by line breaks and indented
        according to the hierarchy defined by sequences of spaces.
    """
    # Split the text by sequences of 2 or more spaces, capturing the delimiters
    parts = re.split(r"( {2,})", clinical_note_raw_text)

    # Separate segments and separators
    # parts format: [segment, separator, segment, separator, ...]
    segments = parts[::2]
    separators = parts[1::2]

    # Store lengths of separators and map them to their indices
    sep_lengths = [len(s) for s in separators]
    length_to_indices = defaultdict(list)
    for idx, length in enumerate(sep_lengths):
        length_to_indices[length].append(idx)

    unique_lengths = sorted(length_to_indices.keys(), reverse=True)

    # Initial indentations (-4)
    segment_indents = np.full(len(segments), -4, dtype=int)

    # Boundaries initialized to surround the whole text
    # Indices correspond to separator indices: -1 (before 0) and len(seps) (after last)
    # This tracks the "groups" of segments that haven't been split by larger spaces yet
    boundaries = [-1, len(separators)]

    for length in unique_lengths:
        current_sep_indices = length_to_indices[length]
        num_indices = len(current_sep_indices)
        current_sep_idx_ptr = 0

        # Iterate over current active groups defined by boundaries
        for i in range(len(boundaries) - 1):
            group_start = boundaries[i]
            group_end = boundaries[i + 1]

            # Advance pointer to first index > group_start
            while (
                current_sep_idx_ptr < num_indices
                and current_sep_indices[current_sep_idx_ptr] <= group_start
            ):
                current_sep_idx_ptr += 1

            # Check if we have a separator in the current group
            if current_sep_idx_ptr < num_indices:
                sep_idx = current_sep_indices[current_sep_idx_ptr]

                # Check if it is within the current group bounds
                if sep_idx < group_end:
                    # Found first valid separator of this length in the group
                    # Apply indentation +8 to all subsequent segments in this group
                    # The segments affected are those to the right of this separator
                    segment_indents[sep_idx + 1 : group_end + 1] += 8

        # Update boundaries with processed separators and re-sort to maintain group structure
        boundaries.extend(current_sep_indices)
        boundaries.sort()

    final_segments = []
    for i, text in enumerate(segments):
        # Strip spaces
        text = text.strip()

        # Determine prefix based on indentation
        indent = segment_indents[i]
        INDENT_STEP = 8

        level = max(indent, 0) // INDENT_STEP
        prefix = (NBSP * (INDENT_STEP * level)) + " " if level > 0 else ("\n" * 2)

        final_segments.append(prefix + text)

    return "\n".join(final_segments)
