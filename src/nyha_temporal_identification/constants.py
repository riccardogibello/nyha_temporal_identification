from enum import Enum
from typing import Any

import polars as pl

DEBUG = False
DEBUG_NOTE_ID = 374890481
PROCESSED_FILE_NAME_PREFIX = "processed"

# NYHA EXTRACTION RELATED CONSTANTS
CURRENT_NYHA_DATE_CHAR_DISTANCE = 100000
CHAR_DISTANCE_KEY = "char_distance"
ORIGINAL_TEXT_KEY = "original_text"
ORIGINAL_START_KEY = "original_start"
START_KEY = "start"
END_KEY = "end"
ORIGINAL_END_KEY = "original_end"
LINKED_DATE_KEY = "linked_date"
ONE_HOT_KEY = "one_hot"
CLASSES_KEY = "classes"
EXTRACTED_NYHA_MENTIONS_KEY = "extracted_nyha_mentions"
EXTRACTED_INVALID_NYHA_MENTIONS_KEY = "extracted_invalid_nyha_mentions"
EXTRACTED_DATES_KEY = "extracted_dates"
TEMPORAL_STATUS_KEY = "temporal_status"
FORMATTED_DATE_KEY = "formatted_date"
NYHA_CLASSES_KEY = "nyha_classes"

# CONFIG PARAMETER NAMES
N_CTX_KEY = "n_ctx"
SEED_KEY = "seed"
TEMPERATURE_KEY = "temperature"
CHUNK_SIZE_KEY = "chunk_size"
MAX_TOKENS_KEY = "max_tokens"
TEMPLATE_KEY = "template"
FORCE_CPU_KEY = "force_cpu"
REPO_ID_KEY = "repo_id"
FILE_NAME_KEY = "file_name"

# DEFAULT VALUES
DEFAULT_FORCE_CPU = False
DEFAULT_CPUS_PER_TASK = 128
DEFAULT_NYHA_CLASS_SEPARATOR = "_"
DEFAULT_NO_TEXT = "None"
TEMPERATURE_KEY_DEFAULT = 0.7
CHUNK_SIZE_KEY_DEFAULT = 250
MAX_TOKENS_KEY_DEFAULT = 10240
TEMPLATE_KEY_DEFAULT = "start_of_turn"
REPO_ID_KEY_DEFAULT = "unsloth/medgemma-4b-it-GGUF"
FILE_NAME_KEY_DEFAULT = "medgemma-4b-it-BF16.gguf"
MOCK_SENTENCE = (
    "NO_LINKED_DATE\n\nHISTORICAL_DATE\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0CURRENT_DATE\n\n"
)
NO_LINKED_DATE_TAG = "NO_LINKED_DATE"
DEFAULT_NO_INTEGER_VALUE = -1
NBSP = "\u00a0"
HISTORICAL_DAYS_THRESHOLD = 180
OVERLAP_THRESHOLD = 0.1


class ColumnType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return super().__eq__(other) or (isinstance(other, str) and self.value == other)

    def __hash__(self):
        return hash(self.value)


DEFAULT_TYPE = ColumnType.STRING
TYPE_DEFAULTS: dict[ColumnType, Any] = {
    ColumnType.INTEGER: -1,
    ColumnType.FLOAT: -1.0,
    ColumnType.BOOLEAN: False,
    ColumnType.STRING: "N/A",
    ColumnType.DATE: "1900-01-01",
    ColumnType.DATETIME: "1900-01-01T00:00:00",
    ColumnType.TIME: "00:00:00",
}


def map_polars_type_to_column_type(polars_dtype: pl.DataType) -> ColumnType:
    if polars_dtype.is_integer():
        return ColumnType.INTEGER
    elif polars_dtype.is_float():
        return ColumnType.FLOAT
    elif polars_dtype == pl.Boolean:
        return ColumnType.BOOLEAN
    elif polars_dtype == pl.Date:
        return ColumnType.DATE
    elif isinstance(polars_dtype, pl.Datetime):
        return ColumnType.DATETIME
    elif isinstance(polars_dtype, pl.Time):
        return ColumnType.TIME
    else:
        return ColumnType.STRING


MONTH_NAME = (
    r"(?:jan(uary|uari)?|"
    r"feb(ruary|ruari)?|"
    r"mar(ch|t)?|"
    r"apr(il)?|"
    r"may|mei|"
    r"jun(e|i)?|"
    r"jul(y|i)?|"
    r"aug(ust|ustus)?|"
    r"sep(tember)?|"
    r"oct(ober)?|okt(ober)?|"
    r"nov(ember)?|"
    r"dec(ember)?)"
)

DATE_PATTERNS = [
    # Day-Month-Year (numeric)
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    # Day-Month-Year (month name)
    rf"\b\d{{1,2}}[-/\s]{MONTH_NAME}[-/\s]\d{{2,4}}\b",
    # Year-Month-Day
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
    # Year-Month-Day (month name)
    rf"\b\d{{4}}[-/\s]{MONTH_NAME}[-/\s]\d{{1,2}}\b",
    # Month-Year
    rf"\b(?:\d{{1,2}}|{MONTH_NAME})[-/\s]\d{{4}}\b",
    # Year-Month
    rf"\b\d{{4}}[-/\s](?:\d{{1,2}}|{MONTH_NAME})\b",
    # Month-Day-Year
    rf"\b{MONTH_NAME}[-/\s]\d{{1,2}}[-/\s]\d{{4}}\b",
    # Year only
    r"\b(19|20)\d{2}\b",
]

# FILE COLUMN MAPPING FILE-RELATED CONSTANTS
REGISTERED_PREFIXES_KEY = "registered_prefixes"
ADDITIONAL_TABLES_KEY = "additional_tables"
TABLE_FOREIGN_TABLES_KEY = "foreign_tables"
OLD_PK_KEY = "old_pk"
COLUMNS_KEY = "columns"
IS_ENTITY_KEY = "type"


# FOLDER NAMES
SNAPSHOT_FOLDER_SUBNAME = "snapshot"
CONFIG_DATA_FOLDER_NAME = "config_data"
CLEANED_TABLES_FOLDER_NAME = "cleaned_tables"
PATIENT_CONTACT_FOLDER_NAME = "patient_contact"
HAS_SPECIALTY_FOLDER_NAME = "has_specialty"
SPECIALTY_FOLDER_NAME = "specialty"
OUTPUT_IMAGES_FOLDER_NAME = "output_images"
ANNOTATION_IMAGES_FOLDER_NAME = "annotation_images"
TEMPORAL_STATUS_PREDICTION_FOLDER_NAME = "temporal_status_prediction"
ANNOTATED_NOTES_DATASET_FOLDER_NAME = "annotated_notes_dataset"
INFERENCE_RESULTS_UNSEEN_FOLDER_NAME = "inference_results_unseen"
FREE_TEXT_FOLDER_NAME = "FreeTextData"
MODELS_FOLDER_NAME = "models"
DATA_FOLDER_NAME = "data"
OLD_TABLES_FOLDER_NAME = "old_tables"
CLUSTERING_IMAGES_FOLDER_NAME = "clustering_images"
MERGED_FOLDER_NAME = "merged"
EXPORTED_ANNOTATIONS_FOLDER_NAME = "exported_annotations"
RESULTS_FOLDER_NAME = "results"
FORMATTED_FOLDER_NAME = "formatted"
COMPARISONS_FOLDER_NAME = "comparisons"
SPLITS_FOLDER_NAME = "splits"
ZS_COMPLETE_FOLDER_NAME = "zs_complete"
OPT_SUBSET_FOLDER_NAME = "opt_subset"
MASKED_NYHA_PREDICTION_FOLDER_NAME = "masked_nyha_prediction"
APPOINTMENT_TYPE_FOLDER_NAME = "appointment_type"
APPOINTMENT_FOLDER_NAME = "appointment"
INPUT_ANNOTATIONS_FOLDER_NAME = "input_annotations"

# FILE NAMES
NONE_CSV_FILE_NAME = "none_ids.csv"
NYHA_COUNTS_FILE_NAME = "nyha_counts.csv"
NYHA_TYPES_CONFUSION_MATRIX_FILE_NAME = "type_nyha_confusion_matrix.png"
NYHA_NOTES_PARQUET_DATASET_FILE_NAME = "nyha_notes_deidentified_tableoutput.parquet"
NYHA_MENTIONS_PARQUET_FILE_NAME = "nyha_note_mentions_extracted.parquet"
ALL_NOTES_METADATA_PARQUET_FILE_NAME = "all_notes_metadata.parquet"
ALL_NOTES_TEXTS_PARQUET_FILE_NAME = "all_notes_texts.parquet"
PROCESSED_META_FILE_NAME_SUFFIX = "processed_meta"
NON_EMPTY_SOURCE_FILES_FILE_NAME = "non_empty_source_files.json"
FILE_COLUMN_MAPPING_FILE = "file_column_mapping.json"
NEW_COLUMN_TYPES_FILE = "new_column_types.json"
ORIGINAL_FILE_NEW_FILES_MAPPING_FILE = "file_table_mapping.json"
PK_MAPPING_FILE = "primary_key_mapping.json"
ALREADY_PROCESSED_FILES_TABLES_MAPPING = "already_processed_files_tables.json"
DEFAULT_COLUMN_TYPES_MAPPING_FILE = "new_column_types.json"
ID_SPLITS_FILE_NAME = "id_splits.json"
SINGLE_SPLIT_FILE_NAME = "single_split.json"
BASELINES_SPLITS_FILE_NAME = "baselines_splits.json"
CONTACT_NOTE_TABLE_NAME = "DT4H_UC3_NotitiePatientContact.parquet"
ALL_NOTES_METADATA_FILE_NAME = "all_notes_metadata.parquet"
ALL_NOTES_TEXTS_FILE_NAME = "all_notes_texts.parquet"
NOTES_DF_FIRST_FILTER_FILE_NAME = "notes_df_first_filter.parquet"

# COLUMN NAMES
BMI_VALUE_COLUMN = "bmi_value"
BODY_TEMPERATURE_COLUMN = "body_temperature"
BODY_HEIGHT_COLUMN = "body_height"
BODY_WEIGHT_COLUMN = "body_weight"
COUNTRY_CODE_COLUMN = "country_code"
DBC_DIAGNOSE_BKEYS = "specialty_code_and_dbc_code"
DBC_DIAGNOSE_CODE = "dbc_diagnose_code"
DIASTOLIC_BLOOD_PRESSURE_COLUMN = "diastolic_blood_pressure"
END_DATE_COLUMN = "end_date"
ID_COLUMN = "id"
HAS_SPECIALTY_ID_COLUMN = "has_specialty_id"
SPECIALTY_ID_COLUMN = "specialty_id"
SPECIALTY_NAME_COLUMN = "specialty_name"
OLD_ID_COLUMN = "old_id"
OLD_NOTE_ID_COLUMN = "old_note_id"
PATIENT_CONTACT_ID_COLUMN = "patient_contact_id"
OLD_PATIENT_CONTACT_ID_COLUMN = "old_patient_contact_id"
PULSE_RATE_COLUMN = "pulse_rate"
SPECIALTY_CODE_COLUMN = "specialty_code"
STANDARD_CODE_ID_COLUMN = "standard_code_id"
STANDARD_CODE_NAME_COLUMN = "standard_code_name"
START_DATE_COLUMN = "start_date"
SYSTOLIC_DIASTOLIC_BLOOD_PRESSURE_COLUMN = "systolic_diastolic_blood_pressure"
SYSTOLIC_BLOOD_PRESSURE_COLUMN = "systolic_blood_pressure"
UNIT_COLUMN = "unit"
DIAGNOSIS_REGISTRATION_DATE_COLUMN = "diagnosis_registration_date"
REGISTRATION_DATE_COLUMN = "registration_date"
PROBLEM_STATUS_COLUMN = "problem_status"
CLOSING_DATE_COLUMN = "closing_date"
ENCOUNTERED_PATIENT_PROBLEM_ID_COLUMN = "encountered_patient_problem_id"
STATUS_COLUMN = "status"
AUTHORED_DATETIME_COLUMN = "authored_datetime"
CONTACT_DATE_COLUMN_NAME = "contact_date"
CONTACT_TYPE_COLUMN_NAME = "contact_type"
CANCELLATION_TYPE_COLUMN_NAME = "cancellation_type"
NYHA_VECTOR_COLUMN = "nyha_vector"
DEIDENTIFIED_TEXT_COLUMN = "deidentified_text"
NOTE_TYPE_COLUMN = "note_type"
DEATH_DATETIME_COLUMN = "death_datetime"
PSEUDO_PATIENT_ID_COLUMN = "pseudo_patient_id"
PATIENT_ID_COLUMN = "patient_id"
TYPE_COLUMN = "type"
AUTHORED_COLUMN = "authored"
ACTUAL_ID_COLUMN = "actual_id"
PART_COLUMN = "part"
IS_DECEASED_COLUMN = "is_deceased"
IS_MULTIPLE_BIRTH_COLUMN = "is_multiple_birth"
IS_TWIN_COLUMN = "is_twin"
FIRST_CONTACT_DATE_COLUMN = "first_contact_date"
LAST_CONTACT_DATE_COLUMN = "last_contact_date"
IS_FROM_DIAGNOSIS_COLUMN = "is_from_diagnosis"
GENERAL_MEASUREMENT_DATE_COLUMN = "general_measurement_date"
MEASUREMENT_DATE_COLUMN = "measurement_date"
BODY_HEIGHT_DATE_COLUMN = "body_height_measurement_date"
BODY_WEIGHT_DATE_COLUMN = "body_weight_measurement_date"
ALCOHOL_USE_EXPLANATION_COLUMN = "alchol_use_explanation"
STATUS_ALCOHOL_USE_COLUMN = "status_alcohol_use"
TYPE_OF_ALCOHOL_COLUMN = "type_of_alcohol"
WEEKLY_AMOUNT_LOWER_LIMIT_COLUMN = "weekly_amount_lower_limit"
WEEKLY_AMOUNT_UPPER_LIMIT_COLUMN = "weekly_amount_upper_limit"
WEEKLY_USE_HABIT_COLUMN = "weekly_use_habit"
ALCOHOL_TYPE_COLUMN = "alcohol_type"
SMOKING_STATUS_COLUMN = "smoking_status"
SMOKING_DAILY_FREQUENCY_COLUMN = "smoking_daily_frequency"
WAS_SMOKING_AT_AUTHORED_DATETIME_COLUMN = "was_smoking_at_authored_datetime"
SMOKING_USAGE_NOTES_COLUMN = "smoking_usage_notes"
PACKS_PER_DAY_COLUMN = "packs_per_day"
ECG_REST_RESULT_ID_COLUMN = "ecg_rest_result_id"
ECG_STRESS_RESULT_ID_COLUMN = "ecg_stress_result_id"
DATE_COLUMN = "date"
VENTRICULAR_RATE_REST_COLUMN = "ventricular_rate_rest"
ATRIAL_RATE_REST_COLUMN = "atrial_rate_rest"
P_AXIS_REST_COLUMN = "p_axis_rest"
R_AXIS_REST_COLUMN = "r_axis_rest"
T_AXIS_REST_COLUMN = "t_axis_rest"
P_ONSET_REST_COLUMN = "p_onset_rest"
P_OFFSET_REST_COLUMN = "p_offset_rest"
Q_ONSET_REST_COLUMN = "q_onset_rest"
Q_OFFSET_REST_COLUMN = "q_offset_rest"
T_OFFSET_REST_COLUMN = "t_offset_rest"
PR_INTERVAL_REST_COLUMN = "pr_interval_rest"
QRS_COUNT_REST_COLUMN = "qrs_count_rest"
QRS_DURATION_REST_COLUMN = "qrs_duration_rest"
QT_INTERVAL_REST_COLUMN = "qt_interval_rest"
QTC_CALCULATION_REST_COLUMN = "qtc_calculation_rest"
QTC_FREDERICIA_REST_COLUMN = "qtc_fredericia_rest"
QTC_FRAMINGHAM_REST_COLUMN = "qtc_framingham_rest"
SAMPLE_BASE_REST_COLUMN = "sample_base_rest"
SAMPLE_EXPONENT_REST_COLUMN = "sample_exponent_rest"
P_ONSET_STRESS_COLUMN = "p_onset_stress"
P_OFFSET_STRESS_COLUMN = "p_offset_stress"
Q_ONSET_STRESS_COLUMN = "q_onset_stress"
Q_OFFSET_STRESS_COLUMN = "q_offset_stress"
T_ONSET_STRESS_COLUMN = "t_onset_stress"
T_OFFSET_STRESS_COLUMN = "t_offset_stress"
PR_INTERVAL_STRESS_COLUMN = "pr_interval_stress"
VENTRICULAR_RATE_STRESS_COLUMN = "ventricular_rate_stress"
AVG_RR_STRESS_COLUMN = "avg_rr_stress"
QRS_COUNT_STRESS_COLUMN = "qrs_count_stress"
QRS_DURATION_STRESS_COLUMN = "qrs_duration_stress"
QT_INTERVAL_STRESS_COLUMN = "qt_interval_stress"
QTC_BAZETT_STRESS_COLUMN = "qtc_bazett_stress"
ECG_MEASUREMENT_ID_COLUMN = "ecg_measurement_id"
OLD_ECG_SUMMARY_ID_COLUMN = "old_ecg_summary_id"
ECG_SUMMARY_CODE_ID_COLUMN = "ecg_summary_code_id"
IS_CONFIRMED_COLUMN = "is_confirmed"
ECG_SUMMARY_DESCRIPTION_COLUMN = "ecg_summary_description"
CONCATENATED_TEXT_COLUMN = "concatenated_text"
ECG_LLM_SUMMARY_COLUMN = "ecg_llm_summary"
ECG_SENTENCE_COLUMN = "ecg_sentence"
CLUSTER_ID_COLUMN = "cluster_id"
CLUSTER_IDENTIFIER_COLUMN = "cluster_identifier"
CLUSTER_NAME_COLUMN = "cluster_name"
CLINICAL_NOTES_SENTENCE_COLUMN = "clinical_notes_sentence"
CLINICAL_NOTES_LLM_SUMMARY_COLUMN = "clinical_notes_llm_summary"
HAS_NYHA_MENTION_COLUMN = "has_nyha_mention"
HAS_INVALID_NYHA_MENTION_COLUMN = "has_invalid_nyha_mention"
NOTE_ID_COLUMN = "note_id"
NOTE_NUMBER_COLUMN = "note_number"
TEXT_LENGTH_COLUMN = "text_length"
OLD_START_INDEX_COLUMN = "old_start_index"
OLD_END_INDEX_COLUMN = "old_end_index"
NEW_START_INDEX_COLUMN = "new_start_index"
NEW_END_INDEX_COLUMN = "new_end_index"
ROW_IDX_COLUMN = "row_idx"
APPOINTMENT_TYPE_ID_COLUMN = "appointment_type_id"
MAIN_PATIENT_CONTACT_ID_COLUMN = "main_patient_contact_id"
SPECIALTY_NAME = "specialty_name"

# TABLE NAMES
REMOVED_SENTENCES = [
    "Dit is een voorlopige en ongesuperviseerde ontslagbrief",
    "Dit is een voorlopige ontslagbrief, niet gesuperviseerd",
    "aanvraag Actueel Medicatie Overzicht",
    "Ontslagrecept/AMO",
    "Een uitgebreide ontslagbrief wordt later door de behandelend arts toegestuurd.",
    "Wij stellen het op prijs het medicatieoverzicht te ontvangen",
    "Fax bestemd voor",
    "Voorlopige ontslagbrief",
    "Dit AMO sturen wij u toe",
    "Betreft: verpleegkundige overdracht",
    "\\*\\*\\*",
]

# ANNOTATOR NAMES
ANNOTATOR_1 = "mboonstra"
ANNOTATOR_2 = "ndormosh"
ANNOTATOR_3 = "solved_merged"
ANNOTATION_RESULTS_FOLDER_NAMES = [
    ANNOTATOR_1,
    ANNOTATOR_2,
    MERGED_FOLDER_NAME,
    ANNOTATOR_3,
]

# JSON KEYS/VALUES
NO_LINKED_DATE_ID = "date_0"
HISTORICAL_DATE_ID = "nyha_type_0"
CURRENT_DATE_ID = "nyha_type_1"
SPECIAL_IDS = {
    NO_LINKED_DATE_ID,
    HISTORICAL_DATE_ID,
    CURRENT_DATE_ID,
}
SPECIAL_ID_NAME_MAP = {
    NO_LINKED_DATE_ID: "NO_LINKED_DATE",
    HISTORICAL_DATE_ID: "HISTORICAL_DATE",
    CURRENT_DATE_ID: "CURRENT_DATE",
}
SPECIAL_ID_LABELS_MAP = {
    NO_LINKED_DATE_ID: ["date"],
    HISTORICAL_DATE_ID: ["nyha_type"],
    CURRENT_DATE_ID: ["nyha_type"],
}
DATA_KEY = "data"
TEXT_KEY = "text"
PREDICTIONS_KEY = "predictions"
MODEL_VERSION_KEY = "model_version"
MODEL_VERSION_VALUE = "manual"
RESULT_KEY = "result"
TASK_KEY = "task"
FROM_NAME_KEY = "from_name"
FROM_ID_KEY = "from_id"
FROM_NAME_VALUE = "label"
TO_NAME_KEY = "to_name"
TO_ID_KEY = "to_id"
TO_NAME_VALUE = "text"
VALUE_KEY = "value"
LABELS_KEY = "labels"
ENTITY_VALUE = "labels"
RELATION_VALUE = "relation"
ORIGIN_KEY = "origin"
PREDICTION_VALUE = "prediction"

NYHA_CURRENT_STATUS = "current"
NYHA_HISTORICAL_STATUS = "historical"

# FILE NAMES
ERROR_WINDOW_ANALYZED = 120  # chars around mention for anchor search

# PSEUDONYMISATION
RESOURCES_FOLDER_NAME = "resources"
PSEUDONYMISATION_FOLDER_NAME = "pseudonymisation"
DUTCH_FIRST_NAMES_FILE_NAME = "dutch_first_names.csv"
DUTCH_SURNAMES_FILE_NAME = "dutch_surnames.csv"
DUTCH_PLACES_FILE_NAME = "dutch_places.csv"
DUTCH_HOSPITALS_FILE_NAME = "dutch_hospitals.csv"
PSEUDONYMISE_KEY = "pseudonymise"
PSEUDONYMISATION_DATA_DIR_KEY = "pseudonymisation_data_dir"
