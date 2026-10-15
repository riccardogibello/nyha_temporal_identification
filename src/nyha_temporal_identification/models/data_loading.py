# =============================================================================
# 1. DATA LOADING
# =============================================================================
import json
import logging
import sys
from pathlib import Path

import polars as pl
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from nyha_temporal_identification.constants import OVERLAP_THRESHOLD
from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.data_model.general import NyhaMention
from nyha_temporal_identification.dspy_logic.regex import spacy_preprocess
from nyha_temporal_identification.extraction.date_parsing import parse_flexible_date
from nyha_temporal_identification.models.constants import (
    CHAR_DISTANCE_COL,
    DAYS_DISTANCE_COL,
    NOT_SET_STRING,
    NOTE_ID_COL,
    NYHA_GOLD_END_INDEX,
    NYHA_GOLD_START_INDEX,
    NYHA_GOLD_VEC_COL,
    NYHA_PRED_END_INDEX,
    NYHA_PRED_START_INDEX,
    NYHA_PRED_VEC_COL,
    POSITIVE_CLASS,
    PRED_COL,
    RANDOM_STATE,
    TARGET_COL,
    TEXT_COL,
)

# =============================================================================
# SECTION 1: ID SPLITTING AND PERSISTENCE
# =============================================================================


def split_document_ids(
    document_ids: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Split document IDs into train, validation, and test sets using an 80/10/10 split
    with a fixed random seed for reproducibility.

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    train_ids, temp_ids = train_test_split(
        document_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )
    return train_ids, val_ids, test_ids


def save_split_ids(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    output_path: str | Path,
) -> None:
    """
    Persist train/val/test ID splits to a JSON file for reproducibility and auditing.

    Args:
        train_ids: List of training document IDs.
        val_ids: List of validation document IDs.
        test_ids: List of test document IDs.
        output_path: Path to save the JSON file. Parent directory will be created if needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_data = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": len(train_ids) + len(val_ids) + len(test_ids),
        },
    }

    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=2)

    logging.info(f"Split IDs saved to {output_path}")


def save_dataframes_to_parquet(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_valid_df: pl.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Persist dataframes to parquet format for efficient storage and retrieval.

    Saves three parquet files to the specified directory:
    - train.parquet
    - val.parquet
    - test_valid.parquet

    Args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        test_valid_df: Valid test predictions dataframe.
        output_dir: Directory where parquet files will be saved. Created if needed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframes = {
        "train.parquet": train_df,
        "val.parquet": val_df,
        "test_valid.parquet": test_valid_df,
    }

    for filename, df in dataframes.items():
        filepath = output_dir / filename
        df.write_parquet(filepath)
        logging.info(f"Saved {filename} ({len(df)} rows) to {filepath}")

    logging.info(f"All dataframes saved to {output_dir}")


# =============================================================================
# SECTION 2: HELPER FUNCTIONS FOR DISTANCE COMPUTATION
# =============================================================================


def compute_distances(
    mention: NyhaMention | NYHAMention,
    authored_datetime: str | None,
    old_note_id: str,
) -> tuple[int, int]:
    """
    Compute ``char_distance`` and ``days_distance`` for either mention schema.

    The function accepts both ``NYHAMention`` and ``NyhaMention`` objects.
    It normalizes access to mention/date fields across both types.

    ``char_distance``:
    - Uses the mention's existing ``char_distance`` if available.
    - Otherwise, if a linked date exists, computes center-to-center character
        distance from normalized mention/date spans.
    - Falls back to ``sys.maxsize`` when no linked date exists or indices are
        invalid/missing.

    ``days_distance``:
    - Computed as ``(authored_date - linked_date).days`` when both are parsable.
    - Falls back to ``sys.maxsize`` when no linked date exists or date parsing
        is unavailable/invalid.
    """
    linked_date = getattr(mention, "linked_date", None)
    has_linked_date = linked_date is not None

    raw_char_distance = getattr(mention, "char_distance", None)
    if has_linked_date and raw_char_distance is not None:
        char_distance = int(raw_char_distance)
    elif has_linked_date:
        mention_start = getattr(
            mention,
            "char_start",
            getattr(mention, "original_start", getattr(mention, "start", -1)),
        )
        mention_end = getattr(
            mention,
            "char_end",
            getattr(mention, "original_end", getattr(mention, "end", -1)),
        )
        date_start = getattr(
            linked_date, "original_start", getattr(linked_date, "start", -1)
        )
        date_end = getattr(linked_date, "original_end", getattr(linked_date, "end", -1))

        if (
            isinstance(mention_start, int)
            and isinstance(mention_end, int)
            and isinstance(date_start, int)
            and isinstance(date_end, int)
            and mention_start >= 0
            and date_start >= 0
            and mention_end > mention_start
            and date_end > date_start
        ):
            avg_mention_char = (mention_start + mention_end) // 2
            avg_date_char = (date_start + date_end) // 2
            char_distance = abs(avg_mention_char - avg_date_char)
        else:
            char_distance = sys.maxsize
    else:
        char_distance = sys.maxsize

    if has_linked_date and linked_date is not None:
        try:
            authored_date_string = authored_datetime
            authored = (
                parse_flexible_date(authored_date_string)
                if authored_date_string
                else None
            )
            linked_date_string = getattr(linked_date, "parsed_date", None)
            linked = (
                parse_flexible_date(linked_date_string) if linked_date_string else None
            )
            days_distance = (
                (authored - linked).days if authored and linked else sys.maxsize
            )
        except Exception as exc:
            logging.warning(
                f"Could not compute days_distance for note '{old_note_id}': {exc}"
            )
            days_distance = sys.maxsize
    else:
        days_distance = sys.maxsize

    return char_distance, days_distance


# =============================================================================
# SECTION 3: DATAFRAME BUILDERS (TRAIN/VAL/TEST)
# =============================================================================


def build_train_val_dfs(
    documents: list[NYHANote],
    train_ids: list[str],
    val_ids: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build separate DataFrames for the train and validation sets using gold mentions.

    Only gold mentions are used as the training focuses on CURRENT/HISTORICAL identification.

    Args:
        documents: List of NYHANote objects containing gold mentions.
        train_ids: List of document IDs designated for training.
        val_ids: List of document IDs designated for validation.

    Returns:
        Tuple of (train_df, val_df).
    """
    id_document_map = {note.old_note_id: note for note in documents}

    train_df_rows = []
    val_df_rows = []
    for id_list, phase_name in zip([train_ids, val_ids], ["train", "val"]):
        for document_id in id_list:
            note = id_document_map[document_id]
            gold_mentions: list[NYHAMention] = note.mentions
            for gold_mention in gold_mentions:
                temporality = gold_mention.temporality
                nyha_char_start = gold_mention.char_start
                nyha_char_end = gold_mention.char_end
                avg_nyha_char = (nyha_char_start + nyha_char_end) // 2
                linked_date = gold_mention.linked_date
                if linked_date is not None:
                    date_char_start = linked_date.original_start
                    date_char_end = linked_date.original_end
                    avg_date_char = (date_char_start + date_char_end) // 2
                    char_distance = abs(avg_nyha_char - avg_date_char)
                    try:
                        authored_date_string = note.authored_datetime
                        authored = (
                            parse_flexible_date(authored_date_string)
                            if authored_date_string
                            else None
                        )
                        linked_date_string = linked_date.parsed_date
                        linked = (
                            parse_flexible_date(linked_date_string)
                            if linked_date_string
                            else None
                        )
                        days_distance = (
                            (authored - linked).days
                            if authored and linked
                            else sys.maxsize
                        )
                    except Exception as exc:
                        logging.warning(
                            f"Could not compute days_distance for note "
                            f"'{note.old_note_id}': {exc}"
                        )
                        days_distance = sys.maxsize
                else:
                    char_distance = sys.maxsize
                    days_distance = sys.maxsize  # Sentinel value for no linked date

                row = {
                    NOTE_ID_COL: note.old_note_id,
                    NYHA_GOLD_START_INDEX: nyha_char_start,
                    NYHA_GOLD_END_INDEX: nyha_char_end,
                    NYHA_GOLD_VEC_COL: NOT_SET_STRING,
                    NYHA_PRED_START_INDEX: nyha_char_start,
                    NYHA_PRED_END_INDEX: nyha_char_end,
                    NYHA_PRED_VEC_COL: NOT_SET_STRING,
                    CHAR_DISTANCE_COL: char_distance,
                    DAYS_DISTANCE_COL: days_distance,
                    TARGET_COL: 1 if temporality == POSITIVE_CLASS else 0,
                    TEXT_COL: note.deidentified_text,
                }
                if phase_name == "train":
                    train_df_rows.append(row)
                else:
                    val_df_rows.append(row)

    train_df = pl.DataFrame(train_df_rows)
    val_df = pl.DataFrame(val_df_rows)
    return train_df, val_df


def build_test_dfs(
    documents: list[NYHANote],
    test_ids: list[str],
) -> tuple[pl.DataFrame, list[NYHANote], dict[str, list[NYHAMention]]]:
    """
    Build test DataFrames by comparing predicted vs gold mentions with overlap assessment.

    Predicted mentions are extracted using spaCy and matched against gold mentions
    based on character index overlap. Distances are computed for valid and invalid
    predictions separately.

    Args:
        documents: List of NYHANote objects containing gold mentions.
        test_ids: List of document IDs designated for testing.

    Returns:
        Tuple of (valid_test_df, invalid_test_df, report_documents, invalid_doc_id_report_predictions).
    """
    valid_test_df_rows = []
    id_document_map = {note.old_note_id: note for note in documents}
    report_documents: list[NYHANote] = []
    invalid_doc_id_report_predictions: dict[str, list[NYHAMention]] = {}
    for test_document_id in test_ids:
        note = id_document_map[test_document_id]
        gold_mentions: list[NYHAMention] = note.mentions
        # Add the current note to the list of documents for testing
        report_documents.append(note)
        predicted_mentions: list[NYHAMention] = []
        for mention in spacy_preprocess(
            deidentified_text=note.deidentified_text,
            authored_datetime=note.authored_datetime,
        ):
            try:
                predicted_mentions.append(
                    NYHAMention.from_nyha_mention(other_nyha_mention=mention)
                )
            except (ValidationError, ValueError) as exc:
                # Defensive-only: under the canonical class/vector convention
                # spaCy predictions are always valid (broad ranges collapse to a
                # single class; I-III yields no classed span). This guards only a
                # genuinely malformed prediction; drop it rather than aborting the
                # whole run.
                logging.warning(
                    "[build_test_dfs] Skipping invalid spaCy prediction in "
                    "note %s at (%s, %s): %s",
                    note.old_note_id,
                    mention.original_start,
                    mention.original_end,
                    exc,
                )

        matched_gm_ids = set()
        for predicted_mention in predicted_mentions:
            # Find the best overlapping gold mention
            best_gold_mention, best_ov = max(
                (
                    (j, g.overlap(predicted_mention))
                    for j, g in enumerate(gold_mentions)
                ),
                key=lambda x: x[1],
                default=(None, 0.0),
            )
            if best_gold_mention is not None and best_ov > OVERLAP_THRESHOLD:
                if best_gold_mention in matched_gm_ids:
                    # This gold mention was already claimed by an earlier
                    # overlapping prediction. A second overlapping span is a
                    # duplicate detection, not a distinct mention: route it to
                    # the invalid predictions so it counts as a false positive
                    # in the report rather than emitting a duplicate valid row
                    # (same gold span + label) that would over-count this gold
                    # mention in the temporality metrics.
                    invalid_doc_id_report_predictions.setdefault(
                        note.old_note_id, []
                    ).append(predicted_mention)
                    continue
                gold_mention_entity: NYHAMention = gold_mentions[best_gold_mention]
                # Compute the char_distance and days_distance for this predicted mention
                # using only the information extracted
                char_distance, days_distance = compute_distances(
                    predicted_mention,
                    authored_datetime=note.authored_datetime,
                    old_note_id=note.old_note_id,
                )
                # Get the gold mention NYHA class and the predicted NYHA class, given that
                # if they do not match, the prediction will be considered wrong
                gm_nyha_vector = gold_mention_entity.nyha_vector
                predicted_nyha_vector = predicted_mention.nyha_vector
                valid_test_df_rows.append(
                    {
                        NOTE_ID_COL: note.old_note_id,
                        NYHA_GOLD_START_INDEX: gold_mention_entity.char_start,
                        NYHA_GOLD_END_INDEX: gold_mention_entity.char_end,
                        NYHA_GOLD_VEC_COL: gm_nyha_vector,
                        NYHA_PRED_START_INDEX: predicted_mention.char_start,
                        NYHA_PRED_END_INDEX: predicted_mention.char_end,
                        NYHA_PRED_VEC_COL: predicted_nyha_vector,
                        CHAR_DISTANCE_COL: char_distance,
                        DAYS_DISTANCE_COL: days_distance,
                        TARGET_COL: (
                            1
                            if gold_mention_entity.temporality == POSITIVE_CLASS
                            else 0
                        ),
                        PRED_COL: 0,
                        TEXT_COL: note.deidentified_text,
                    }
                )
                matched_gm_ids.add(best_gold_mention)
            else:
                # If the current extracted mention does not match any gold standard,
                # I should add as is to the invalid predictions to account for
                # false positive rate in the evaluation
                invalid_doc_id_report_predictions.setdefault(
                    note.old_note_id, []
                ).append(predicted_mention)

    valid_test_df = pl.DataFrame(valid_test_df_rows)
    return (
        valid_test_df,
        report_documents,
        invalid_doc_id_report_predictions,
    )
