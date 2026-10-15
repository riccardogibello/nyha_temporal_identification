# ── Configurable paths & constants ───────────────────────────────────────────
from pathlib import Path

DATA_PATH: str = "data/"
OUTPUT_DIR: Path = Path("outputs")
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.15
TARGET_COL: str = "target"
POSITIVE_CLASS: str = "current"
BASE_FEATURES: list[str] = ["char_distance", "days_distance"]
MI_DROP_THRESHOLD: float = 0.01
THRESHOLD_GRID_STEPS: int = 19
NOT_SET_INDEX = -1  # for numeric features with missing values
NOT_SET_STRING = "NOT_SET"

# Text-branch constants
DEFAULT_WINDOW_SIZE: int = 200  # characters on each side of the span

# TRAIN/VAL/TEST dataframe columns
NOTE_ID_COL: str = "note_id"
TEXT_COL: str = "text"
NYHA_PRED_START_INDEX: str = "nyha_pred_start_index"
NYHA_PRED_END_INDEX: str = "nyha_pred_end_index"
NYHA_PRED_VEC_COL: str = "nyha_pred_class"
NYHA_GOLD_START_INDEX: str = "nyha_gold_start_index"
NYHA_GOLD_END_INDEX: str = "nyha_gold_end_index"
NYHA_GOLD_VEC_COL: str = "nyha_gold_class"
CHAR_DISTANCE_COL: str = "char_distance"
DAYS_DISTANCE_COL: str = "days_distance"
PRED_COL: str = "pred"
BASELINES_FOLDER_NAME: str = "baselines"
FEATURE_MODE_BASE: str = "base"
FEATURE_MODE_BASE_TEXT: str = "base_text"
FEATURE_MODE_TEXT: str = "text"
FEATURE_MODES: tuple[str, ...] = ("base", "base_text", "text")
