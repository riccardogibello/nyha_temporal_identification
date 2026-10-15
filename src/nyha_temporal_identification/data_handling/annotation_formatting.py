import json
import logging
import os

from nyha_temporal_identification.constants import (
    ANNOTATION_RESULTS_FOLDER_NAMES,
    AUTHORED_COLUMN,
    CLASSES_KEY,
    CURRENT_DATE_ID,
    DATA_KEY,
    DEBUG,
    DEBUG_NOTE_ID,
    DEIDENTIFIED_TEXT_COLUMN,
    END_KEY,
    ENTITY_VALUE,
    EXPORTED_ANNOTATIONS_FOLDER_NAME,
    EXTRACTED_NYHA_MENTIONS_KEY,
    FORMATTED_DATE_KEY,
    FORMATTED_FOLDER_NAME,
    FROM_NAME_KEY,
    HISTORICAL_DATE_ID,
    ID_COLUMN,
    LABELS_KEY,
    LINKED_DATE_KEY,
    MERGED_FOLDER_NAME,
    NO_LINKED_DATE_ID,
    NYHA_CURRENT_STATUS,
    NYHA_HISTORICAL_STATUS,
    ONE_HOT_KEY,
    ORIGIN_KEY,
    ORIGINAL_END_KEY,
    ORIGINAL_START_KEY,
    ORIGINAL_TEXT_KEY,
    PREDICTION_VALUE,
    PREDICTIONS_KEY,
    RELATION_VALUE,
    RESULT_KEY,
    RESULTS_FOLDER_NAME,
    SPECIAL_ID_LABELS_MAP,
    SPECIAL_ID_NAME_MAP,
    SPECIAL_IDS,
    START_KEY,
    TEMPORAL_STATUS_KEY,
    TEXT_KEY,
    TO_NAME_KEY,
    TYPE_COLUMN,
    VALUE_KEY,
)
from nyha_temporal_identification.data_handling.utils import (
    extract_authored_date,
    is_special_entity,
    translate_nyha_name,
)
from nyha_temporal_identification.extraction.date_parsing import parse_flexible_date


def add_missing_special_entities(
    id_entities_map: dict[str, dict],
    note_text: str,
) -> dict[str, dict]:
    """
    This method updates the given map of entities by adding any missing special
    entities (e.g. NO_LINKED_DATE, CURRENT_DATE, HISTORICAL_DATE) that are present
    in the note text but not in the map, with their corresponding start and end
    indices in the text and predefined labels.

    Args:
        id_entities_map (dict[str, dict]): A map where the keys are entity IDs and
        the values are dictionaries containing entity information such as text,
        start and end indices, and labels.
        note_text (str): The text of the note containing the entities.

    Returns:
        dict[str, dict]: An updated map of entities that includes any missing
        special entities found in the note text.
    """
    copy_id_entities_map = id_entities_map.copy()
    # For each special identifier
    for special_id in SPECIAL_IDS:
        # Get the name of the special ID variable and verify if it is present
        # in the note text
        special_id_name = SPECIAL_ID_NAME_MAP.get(special_id, None)
        if special_id_name is None or special_id_name not in note_text:
            logging.warning(
                f"Special ID {special_id} not found in existing_start_end_map, but the corresponding name {special_id_name} is not present in the note text, so added to the map without a name and without a warning about missing name in the text"
            )
            continue
        # Find the start and end indices of the special ID in the note text,
        # if present, and add a corresponding entity to the map if not already
        # present
        start_index = note_text.index(special_id_name)
        end_index = start_index + len(special_id_name)
        if special_id not in copy_id_entities_map:
            special_entity = {
                ID_COLUMN: special_id,
                VALUE_KEY: {
                    TEXT_KEY: special_id_name,
                    START_KEY: start_index,
                    END_KEY: end_index,
                    LABELS_KEY: SPECIAL_ID_LABELS_MAP[special_id],
                },
                FROM_NAME_KEY: FROM_NAME_KEY,
                TO_NAME_KEY: TO_NAME_KEY,
                TYPE_COLUMN: ENTITY_VALUE,
                ORIGIN_KEY: PREDICTION_VALUE,
            }
            copy_id_entities_map[special_id] = special_entity

    return copy_id_entities_map


def format_annotated_dataset(
    data_folder_path: str,
) -> None:
    """
    This method transforms the annotated files as they are provided by the LabelStudio tool
    into the format required by the downstram pipeline.

    Args:
        data_folder_path (str): _description_

    Raises:
        ValueError: _description_
    """
    # Prepare the paths to all the annotators' folders
    # and the path to the output folder
    input_folder_paths = [
        os.path.join(
            data_folder_path,
            EXPORTED_ANNOTATIONS_FOLDER_NAME,
            RESULTS_FOLDER_NAME,
            folder_name,
        )
        for folder_name in ANNOTATION_RESULTS_FOLDER_NAMES
        if folder_name != MERGED_FOLDER_NAME
    ]
    output_folder_path = os.path.join(
        data_folder_path,
        EXPORTED_ANNOTATIONS_FOLDER_NAME,
        FORMATTED_FOLDER_NAME,
    )
    # If the output folder already exists, and it is not empty, return
    if os.path.exists(output_folder_path) and os.listdir(output_folder_path):
        logging.info(
            f"[INFO] Output folder {output_folder_path} already exists and is not empty, skipping formatting of annotated dataset"
        )
        return
    os.makedirs(output_folder_path, exist_ok=True)

    # For each input folder
    for input_folder_path in input_folder_paths:
        json_file_names = [
            f for f in os.listdir(input_folder_path) if f.endswith(".json")
        ]
        assert json_file_names, f"No JSON files found in folder: {input_folder_path}"

        # For each JSON file containing the note data
        for file_name in json_file_names:
            # Prepare the input and output file paths, and if the output file already
            # exists, skip the formatting for that file
            input_file_path = os.path.join(input_folder_path, file_name)
            output_file_path = os.path.join(output_folder_path, file_name)
            if os.path.exists(output_file_path):
                logging.info(
                    f"[INFO] Skipping already formatted file: {output_file_path}"
                )
                continue
            # Get all the annotated data from the input file, extract the relevant
            # information (note text and note identifier)
            with open(input_file_path) as f_in:
                data = json.load(f_in)
            note_original_text = data[DATA_KEY][TEXT_KEY]
            old_note_id = int(file_name.split(".")[0])
            if old_note_id != DEBUG_NOTE_ID and DEBUG:
                logging.warning(
                    f"[DEBUG] Skipping document with old_note_id={old_note_id} (not matching DEBUG_NOTE_ID={DEBUG_NOTE_ID})"
                )
                continue
            # Get all the annotations and divide them between entities and relations
            # and build the map of entities
            results = data[PREDICTIONS_KEY][0][RESULT_KEY]
            entities = [
                tmp_map for tmp_map in results if tmp_map[TYPE_COLUMN] == ENTITY_VALUE
            ]
            relations = [
                tmp_map for tmp_map in results if tmp_map[TYPE_COLUMN] == RELATION_VALUE
            ]
            id_entities_map = {e[ID_COLUMN]: e for e in entities}
            id_entities_map = add_missing_special_entities(
                id_entities_map,
                note_original_text,
            )
            # For each relation reported
            for relation_map in relations:
                # Get the name of the relation and skip it if it is the "same region"
                # relation, as it is not handled by annotators but is inserted when
                # merging the results (it should never happen as the conflicts were solved)
                relation_name = relation_map[LABELS_KEY][0]
                if relation_name == "same region":
                    raise ValueError(
                        f"Found a relation with name 'same region' in {input_file_path}, but this relation should have been removed during the merging of the annotations, so there is likely an issue in the preprocessing pipeline that needs to be fixed: {relation_map}"
                    )
                from_id: str = relation_map["from_id"]
                to_id = relation_map["to_id"]
                any_is_missing = (
                    from_id not in id_entities_map or to_id not in id_entities_map
                )
                if any_is_missing:
                    logging.warning(
                        f"[WARN] doc={old_note_id}: relation with missing entity — skipping relation"
                    )
                    continue
                is_from_entity_special = is_special_entity(
                    from_id,
                    id_entities_map[from_id][VALUE_KEY][TEXT_KEY],
                )

                is_to_entity_special = is_special_entity(
                    to_id,
                    id_entities_map[to_id][VALUE_KEY][TEXT_KEY],
                )
                if is_from_entity_special and is_to_entity_special:
                    logging.warning(
                        f"[WARN] doc={old_note_id}: relation with special entities — skipping relation"
                    )

                    continue

                nyha_id = (
                    from_id
                    if not is_from_entity_special and from_id.startswith("nyha_")
                    else to_id
                )
                nyha_map = id_entities_map[nyha_id]
                # If not present, add a "temporal_status" key to the map, set to the "current" status
                if not nyha_map.get(TEMPORAL_STATUS_KEY):
                    nyha_map[TEMPORAL_STATUS_KEY] = NYHA_CURRENT_STATUS
                # If not present, add an empty map for the "linked_date" key
                if not nyha_map.get(LINKED_DATE_KEY):
                    nyha_map[LINKED_DATE_KEY] = None
                other_id = to_id if nyha_id == from_id else from_id
                if other_id in [CURRENT_DATE_ID, HISTORICAL_DATE_ID]:
                    if other_id == CURRENT_DATE_ID:
                        nyha_map[TEMPORAL_STATUS_KEY] = NYHA_CURRENT_STATUS
                    else:
                        nyha_map[TEMPORAL_STATUS_KEY] = NYHA_HISTORICAL_STATUS
                else:
                    if other_id == NO_LINKED_DATE_ID:
                        continue
                    else:
                        # Get the date entity from the map and set it as the "linked_date" for the NYHA mention
                        date_entity = id_entities_map[other_id]
                        nyha_map[LINKED_DATE_KEY] = date_entity

            extracted_nyha_mentions = []
            authored_date = extract_authored_date(note_original_text)
            authored_date_string = (
                authored_date.isoformat() if authored_date is not None else ""
            )
            for id, tmp_nyha_map in id_entities_map.items():
                is_entity_special = is_special_entity(
                    id,
                    tmp_nyha_map[VALUE_KEY][TEXT_KEY],
                )
                if not id.startswith("nyha_") or is_entity_special:
                    continue

                tmp_linked_date_map = tmp_nyha_map.get(LINKED_DATE_KEY, None)
                linked_date_map = tmp_linked_date_map
                if tmp_linked_date_map:
                    date = parse_flexible_date(tmp_linked_date_map[VALUE_KEY][TEXT_KEY])
                    iso_formatted_string = date.isoformat() if date is not None else ""

                    linked_date_map = {
                        ORIGINAL_TEXT_KEY: tmp_linked_date_map[VALUE_KEY][TEXT_KEY],
                        FORMATTED_DATE_KEY: iso_formatted_string,
                        ORIGINAL_START_KEY: tmp_linked_date_map[VALUE_KEY][START_KEY],
                        ORIGINAL_END_KEY: tmp_linked_date_map[VALUE_KEY][END_KEY],
                        TEMPORAL_STATUS_KEY: tmp_nyha_map[TEMPORAL_STATUS_KEY],
                    }
                # NOTE: not all the fields are added, but only the needed ones
                classes, one_hot = translate_nyha_name(
                    tmp_nyha_map[VALUE_KEY][LABELS_KEY][0]
                )
                new_nyha_map = {
                    ORIGINAL_TEXT_KEY: tmp_nyha_map[VALUE_KEY][TEXT_KEY],
                    CLASSES_KEY: classes,
                    ONE_HOT_KEY: one_hot,
                    ORIGINAL_START_KEY: tmp_nyha_map[VALUE_KEY][START_KEY],
                    ORIGINAL_END_KEY: tmp_nyha_map[VALUE_KEY][END_KEY],
                    LINKED_DATE_KEY: linked_date_map,
                    TEMPORAL_STATUS_KEY: tmp_nyha_map[TEMPORAL_STATUS_KEY],
                }
                extracted_nyha_mentions.append(new_nyha_map)

            formatted_map = {
                old_note_id: {
                    ID_COLUMN: old_note_id,
                    AUTHORED_COLUMN: authored_date_string,
                    DEIDENTIFIED_TEXT_COLUMN: note_original_text,
                    EXTRACTED_NYHA_MENTIONS_KEY: extracted_nyha_mentions,
                }
            }
            
            with open(output_file_path, "w") as f_out:
                json.dump(formatted_map, f_out, indent=2)
