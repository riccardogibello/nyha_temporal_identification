import csv
import gc
import json
import logging
import os
import random
import shutil
from collections import defaultdict

import polars as pl
import tqdm

from nyha_temporal_identification.constants import (
    ALL_NOTES_METADATA_PARQUET_FILE_NAME,
    ALL_NOTES_TEXTS_PARQUET_FILE_NAME,
    DEBUG,
    DEFAULT_NYHA_CLASS_SEPARATOR,
    DEIDENTIFIED_TEXT_COLUMN,
    EXTRACTED_NYHA_MENTIONS_KEY,
    FREE_TEXT_FOLDER_NAME,
    NON_EMPTY_SOURCE_FILES_FILE_NAME,
    NYHA_COUNTS_FILE_NAME,
    NYHA_NOTES_PARQUET_DATASET_FILE_NAME,
    OLD_ID_COLUMN,
    OLD_TABLES_FOLDER_NAME,
    PROCESSED_FILE_NAME_PREFIX,
    PROCESSED_META_FILE_NAME_SUFFIX,
)
from nyha_temporal_identification.extraction.parsing import build_nyha_mentions_map
from nyha_temporal_identification.utils.io import find_file_in_subfolders


def aggregate_nyha_counts(
    fname_type_nyha_counts: dict[str, dict[str, dict[str, int]]],
) -> dict[str, int]:
    """
    This method receives a dictionary that, for each file and type of visit, stores
    the counts of each NYHA class found in that file.

    Args:
        fname_type_nyha_counts (dict[str, dict[str, dict[str, int]]]): A dictionary where
            the first key is the filename, the second key is the type of clinical note,
            and the value is another dictionary with NYHA class counts.

    Returns:
        dict[str, int]: A dictionary with the aggregated counts of each NYHA class
            across all files and types.
    """
    agg_counts = defaultdict(int)
    for type_nyha_counts in fname_type_nyha_counts.values():
        for nyha_counts in type_nyha_counts.values():
            for nyha, count in nyha_counts.items():
                agg_counts[nyha] += count
    return dict(agg_counts)


def load_counts_table(
    counts_csv_path,
) -> dict[str, dict[str, dict[str, int]]]:
    """
    Loads from a file a table containing, for each filename and type of clinical note,
    the counts of each NYHA class found in that file (even for multiple classes per sample, like 2-3).

    Args:
        counts_csv_path (_type_): Path to the CSV file containing the counts table.

    Returns:
        dict[str, dict[str, dict[str, int]]]: A dictionary where the first key is the filename,
            the second key is the type of clinical note, and the value is another dictionary
            with NYHA class counts.
    """
    if not os.path.exists(counts_csv_path):
        return {}
    counts_table: dict[str, dict[str, dict[str, int]]] = {}
    with open(counts_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            type_clinical_note = row["type"]
            counts = {
                k: int(v)
                for k, v in row.items()
                if k != "filename" and v and k != "type"
            }
            if fname not in counts_table:
                counts_table[fname] = {}
            if type_clinical_note not in counts_table[fname]:
                counts_table[fname][type_clinical_note] = {}
            counts_table[fname][type_clinical_note] = counts
    return counts_table


def save_counts_table(counts_csv_path, counts_table):
    # Collect all possible NYHA classes
    all_classes = set()
    types = [type_counts.keys() for type_counts in counts_table.values()]
    unique_types = set()
    for type_list in types:
        unique_types.update(type_list)
    unique_types = list(sorted(unique_types))
    for type_counts in counts_table.values():
        for counts in type_counts.values():
            all_classes.update(counts.keys())
    fieldnames = ["filename", "type"] + [str(k) for k in sorted(all_classes)]
    with open(counts_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fname, type_counts in counts_table.items():
            for type_, counts in type_counts.items():
                row = {"filename": fname, "type": type_}
                for k in all_classes:
                    row[str(k)] = counts.get(k, 0)
                writer.writerow(row)


def aggregate_type_nyha(
    df: pl.DataFrame,
) -> dict[str, dict[str, int]]:
    """
    Aggregates the NYHA counts from the dataframe, presenting them per-file, into a single dictionary
    where the first key is the type of clinical note, and the value is another dictionary with NYHA
    class counts (even multiple classes per sample, like 2-3).

    Args:
        df (pl.DataFrame): A polars dataframe containing the clinical notes with NYHA class columns presenting
        0 if the class was not found, and 1 if it was found.
        Expects columns 'nyha_class_1', 'nyha_class_2', 'nyha_class_3', 'nyha_class_4'.

    Returns:
        dict[str, dict[str, int]]: A dictionary where the first key is the type of clinical note,
            and the value is another dictionary with NYHA class counts.
    """
    type_nyha_counts = {}

    # Define expressions for base columns
    c1 = pl.col("nyha_class_1")
    c2 = pl.col("nyha_class_2")
    c3 = pl.col("nyha_class_3")
    c4 = pl.col("nyha_class_4")

    # Calculate row sum of NYHA classes to identify multihot vectors
    row_sum = c1 + c2 + c3 + c4

    # Logic for pairs: sum==2, and specific contiguous classes
    is_pair_1_2 = (c1 == 1) & (c2 == 1) & (row_sum == 2)
    is_pair_2_3 = (c2 == 1) & (c3 == 1) & (row_sum == 2)
    is_pair_3_4 = (c3 == 1) & (c4 == 1) & (row_sum == 2)

    # Logic for singles: class is present AND it's NOT part of a pair
    # Note: 1-2 consumes 1 and 2. 1-3 (if existed) would be two singles.
    # so single_1 is only if 1 is present and 1-2 is NOT present.
    is_single_1 = (c1 == 1) & (~is_pair_1_2)
    is_single_2 = (c2 == 1) & (~is_pair_1_2) & (~is_pair_2_3)
    is_single_3 = (c3 == 1) & (~is_pair_2_3) & (~is_pair_3_4)
    is_single_4 = (c4 == 1) & (~is_pair_3_4)

    # Perform aggregation by group "type"
    agg_df = df.group_by("type").agg(
        [
            is_single_1.sum().cast(pl.Int64).alias("nyha_class_1"),
            is_single_2.sum().cast(pl.Int64).alias("nyha_class_2"),
            is_single_3.sum().cast(pl.Int64).alias("nyha_class_3"),
            is_single_4.sum().cast(pl.Int64).alias("nyha_class_4"),
            is_pair_1_2.sum()
            .cast(pl.Int64)
            .alias(f"nyha_class_1{DEFAULT_NYHA_CLASS_SEPARATOR}2"),
            is_pair_2_3.sum()
            .cast(pl.Int64)
            .alias(f"nyha_class_2{DEFAULT_NYHA_CLASS_SEPARATOR}3"),
            is_pair_3_4.sum()
            .cast(pl.Int64)
            .alias(f"nyha_class_3{DEFAULT_NYHA_CLASS_SEPARATOR}4"),
        ]
    )

    # Convert results to dictionary
    for row in agg_df.iter_rows(named=True):
        t = row.pop("type")
        # Ensure that only integer counts are in the dict
        counts = {k: int(v) for k, v in row.items()}
        type_nyha_counts[t] = counts

    return type_nyha_counts


def extract_nyha_labels(
    data_folder_path: str,
    reset_previous_data: bool = False,
    pseudonymise: bool = False,
    seed: int = 42,
    pseudonymisation_data_dir: str | None = None,
) -> None:
    """
    This method processes every parquet file containing clinical notes in the specified folder
    by extracting the contained NYHA class mentions from the notes, as well as the dates
    and the nearest date that can be linked to each of them.

    Per-file checkpoints are written incrementally so that interrupted runs can resume without
    reprocessing already-completed files. After all files are processed, the per-file checkpoints
    are merged into two unified datasets:
      - all_notes_metadata.parquet: lightweight metadata for every note (12M+ rows).
      - nyha_mentions_unified.parquet: full text + mentions + dates for NYHA-positive notes only.

    Args:
        data_folder_path (str): Path to the main data folder containing the clinical notes files.
        reset_previous_data (bool, optional): Whether to reset previous data. Defaults to False.
    """
    free_text_data_folder_path = os.path.join(
        data_folder_path,
        OLD_TABLES_FOLDER_NAME,
        FREE_TEXT_FOLDER_NAME,
    )
    # If the previous data must be reset, archive any existing processed files and counts
    if reset_previous_data:
        tmp_folder_path = os.path.join(free_text_data_folder_path, "tmp")
        os.makedirs(tmp_folder_path, exist_ok=True)

        def move_to_tmp(file_path: str) -> None:
            if not os.path.exists(file_path):
                return
            basename = os.path.basename(file_path)
            destination_path = os.path.join(tmp_folder_path, basename)
            if os.path.exists(destination_path):
                name, ext = os.path.splitext(basename)
                suffix = 1
                while True:
                    candidate_path = os.path.join(
                        tmp_folder_path,
                        f"{name}_{suffix}{ext}",
                    )
                    if not os.path.exists(candidate_path):
                        destination_path = candidate_path
                        break
                    suffix += 1
            shutil.move(file_path, destination_path)

        for fp in (
            find_file_in_subfolders(
                base_folder=free_text_data_folder_path,
                target_file=None,
            )
            or []
        ):
            basename = os.path.basename(fp)
            if (
                PROCESSED_FILE_NAME_PREFIX in basename
                or PROCESSED_META_FILE_NAME_SUFFIX in basename
            ):
                move_to_tmp(fp)
        # Move counts file if it exists
        counts_csv_path = os.path.join(
            free_text_data_folder_path,
            NYHA_COUNTS_FILE_NAME,
        )
        if os.path.exists(counts_csv_path):
            move_to_tmp(counts_csv_path)
        # Move unified output datasets if they exist
        for unified_file in (
            ALL_NOTES_METADATA_PARQUET_FILE_NAME,
            ALL_NOTES_TEXTS_PARQUET_FILE_NAME,
            NYHA_NOTES_PARQUET_DATASET_FILE_NAME,
            NON_EMPTY_SOURCE_FILES_FILE_NAME,
        ):
            path = os.path.join(free_text_data_folder_path, unified_file)
            if os.path.exists(path):
                move_to_tmp(path)

    # Extract all the source files containing clinical notes to be processed
    file_paths = [
        fp
        for fp in (
            find_file_in_subfolders(
                base_folder=free_text_data_folder_path,
                target_file=None,
            )
            or []
        )
        if PROCESSED_FILE_NAME_PREFIX not in os.path.basename(fp)
        and PROCESSED_META_FILE_NAME_SUFFIX not in os.path.basename(fp)
        and fp.split(os.path.sep)[-1].startswith("DT4H")
    ]
    # Set the seed and randomly shuffle the file paths to ensure a random processing order
    # (useful for debugging with a subset of files)
    random.seed(42)
    random.shuffle(file_paths)
    if not file_paths:
        logging.warning(
            f"No files found in {free_text_data_folder_path} for NYHA extraction. Please check the path and ensure it contains the expected clinical notes files."
        )
        return
    if DEBUG:
        logging.debug("DEBUG mode is ON: only processing the first files")
        file_paths = file_paths[:5]

    # Build the Pseudonymiser once for this run (loads CSVs once, reused across all files).
    _pseudonymiser = None
    if pseudonymise:
        from nyha_temporal_identification.utils.pseudonymisation import Pseudonymiser
        _pseudonymiser = Pseudonymiser(data_dir=pseudonymisation_data_dir)
        logging.info("Pseudonymisation enabled (seed=%d)", seed)

    counts_csv_path = os.path.join(
        free_text_data_folder_path,
        NYHA_COUNTS_FILE_NAME,
    )
    fname_type_nyha_counts = load_counts_table(counts_csv_path)
    non_empty_source_files: list[str] = []

    for fp in tqdm.tqdm(file_paths, desc="Processing files"):
        fname = os.path.basename(fp)
        fname_no_ext, _ = os.path.splitext(fname)
        processed_json_path = os.path.join(
            free_text_data_folder_path,
            f"{fname_no_ext}_{PROCESSED_FILE_NAME_PREFIX}.json",
        )
        processed_meta_path = os.path.join(
            free_text_data_folder_path,
            f"{fname_no_ext}_{PROCESSED_META_FILE_NAME_SUFFIX}.parquet",
        )

        # Both checkpoints must exist to consider this file already processed
        if not os.path.exists(processed_json_path) or not os.path.exists(
            processed_meta_path
        ):
            json_map, metadata_rows = build_nyha_mentions_map(
                file_path=fp,
                data_folder_path=data_folder_path,
                pseudonymiser=_pseudonymiser,
                seed=seed,
            )
            # Save the per-file metadata parquet (all notes, lightweight)
            pl.DataFrame(metadata_rows).write_parquet(processed_meta_path)
            # Save the per-file JSON with text + mentions + dates (NYHA-positive notes only)
            with open(processed_json_path, "w", encoding="utf-8") as f:
                json.dump(json_map, f, ensure_ascii=False, indent=4)
            # Update NYHA counts from the json_map (contains one_hot_vector per mention)
            if fname not in fname_type_nyha_counts and json_map:
                rows = []
                for note_data in json_map.values():
                    note_type = note_data.get("type", "unknown")
                    for mention in note_data.get(EXTRACTED_NYHA_MENTIONS_KEY, []):
                        vec = mention.get("one_hot_vector", [0, 0, 0, 0])
                        rows.append(
                            {
                                "type": note_type,
                                "nyha_class_1": vec[0],
                                "nyha_class_2": vec[1],
                                "nyha_class_3": vec[2],
                                "nyha_class_4": vec[3],
                            }
                        )
                if rows:
                    fname_type_nyha_counts[fname] = aggregate_type_nyha(
                        pl.DataFrame(rows)
                    )
                    save_counts_table(counts_csv_path, fname_type_nyha_counts)
            if metadata_rows:
                non_empty_source_files.append(fp)
        else:
            # Already processed — determine non-emptiness from the cached meta parquet
            if pl.read_parquet(processed_meta_path).height > 0:
                non_empty_source_files.append(fp)

    # Save the list of source files that produced at least one note after the join
    non_empty_source_files_path = os.path.join(
        free_text_data_folder_path,
        NON_EMPTY_SOURCE_FILES_FILE_NAME,
    )
    with open(non_empty_source_files_path, "w", encoding="utf-8") as f:
        json.dump(non_empty_source_files, f, indent=2)

    # --- Merge step: build unified datasets from all per-file checkpoints ---
    all_files = (
        find_file_in_subfolders(
            base_folder=free_text_data_folder_path,
            target_file=None,
        )
        or []
    )

    # Collect and merge all per-file metadata parquets into the unified main dataset
    meta_parquet_paths = [
        fp
        for fp in all_files
        if PROCESSED_META_FILE_NAME_SUFFIX in os.path.basename(fp)
        and fp.endswith(".parquet")
    ]
    if meta_parquet_paths:
        logging.info(
            f"Merging {len(meta_parquet_paths)} metadata parquets into unified main dataset..."
        )
        # Release whatever memory Python can recover before the merge — the processing
        # loop above may have left large allocations from HuggingFace multiprocessing.
        gc.collect()
        # Use scan_parquet + sink_parquet to stream each file rather than loading all
        # 66 DataFrames into memory at once (avoids OOM when processing memory is high).
        pl.scan_parquet(meta_parquet_paths).drop(DEIDENTIFIED_TEXT_COLUMN).sink_parquet(
            os.path.join(
                free_text_data_folder_path,
                ALL_NOTES_METADATA_PARQUET_FILE_NAME,
            )
        )
        pl.scan_parquet(meta_parquet_paths).select(
            [OLD_ID_COLUMN, DEIDENTIFIED_TEXT_COLUMN]
        ).sink_parquet(
            os.path.join(
                free_text_data_folder_path,
                ALL_NOTES_TEXTS_PARQUET_FILE_NAME,
            )
        )
