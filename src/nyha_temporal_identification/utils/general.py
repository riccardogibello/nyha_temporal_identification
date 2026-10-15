import argparse
import os
import tomllib

from nyha_temporal_identification.constants import (
    PSEUDONYMISATION_DATA_DIR_KEY,
    PSEUDONYMISE_KEY,
)

# CONFIG KEYS — top-level section
COMMANDS_KEY = "commands"
USE_IDENTIFICATION_KEY = "use_identification"
USE_WINDOWED_CLASSIFICATION_CONTEXT = "use_windowed_classification_context"
SEED_KEY = "seed"
HF_TOKEN_FILE = "hf_token_file"
N_BOOTSTRAP_KEY = "n_bootstrap"
BASELINE_FEATURE_MODES_KEY = "baseline_feature_modes"
FAILSAFE_DROPPED_CURRENT_KEY = "failsafe_dropped_current"
FAILSAFE_NOTE_DROP_POLICY_KEY = "failsafe_note_drop_policy"

_VALID_FEATURE_MODES: frozenset[str] = frozenset({"base", "base_text", "text"})

# CONFIG KEYS — [dspy_pipeline] section
TASK_LM_KEY = "task_lm"
OPTIMISER_KEY = "optimiser"

TOP_LEVEL_REQUIRED_KEYS = [
    COMMANDS_KEY,
    SEED_KEY,
    USE_IDENTIFICATION_KEY,
    USE_WINDOWED_CLASSIFICATION_CONTEXT,
]

TOP_LEVEL_OPTIONAL_KEYS = [
    HF_TOKEN_FILE,
    PSEUDONYMISE_KEY,
    PSEUDONYMISATION_DATA_DIR_KEY,
    N_BOOTSTRAP_KEY,
    BASELINE_FEATURE_MODES_KEY,
    FAILSAFE_DROPPED_CURRENT_KEY,
    FAILSAFE_NOTE_DROP_POLICY_KEY,
]

DSPY_PIPELINE_REQUIRED_KEYS = [
    TASK_LM_KEY,
    OPTIMISER_KEY,
]

# COMMAND NAMES
EXTRACT_NYHA_COMMAND_NAME = "extract_nyha"
AUGMENT_DATA_COMMAND_NAME = "augment_data"
RUN_DSPY_PIPELINE_COMMAND_NAME = "run_dspy_pipeline"
ANNOTATE_TRAINING_NYHA_MENTIONS_COMMAND_NAME = "annotate_training_nyha_mentions"
TRAIN_BASELINES_COMMAND_NAME = "train_baselines"
ALLOWED_COMMANDS = [
    EXTRACT_NYHA_COMMAND_NAME,
    AUGMENT_DATA_COMMAND_NAME,
    RUN_DSPY_PIPELINE_COMMAND_NAME,
    ANNOTATE_TRAINING_NYHA_MENTIONS_COMMAND_NAME,
    TRAIN_BASELINES_COMMAND_NAME,
]


class Config:
    def __init__(
        self,
        # Top-level required
        commands: list[str],
        seed: int,
        use_identification: bool,
        use_windowed_classification_context: bool,
        # [dspy_pipeline] required
        task_lm: str,
        optimiser: str,
        # Top-level optional
        hf_token_file: str = ".hf_token",
        pseudonymise: bool = False,
        pseudonymisation_data_dir: str | None = None,
        n_bootstrap: int = 0,
        baseline_feature_modes: list[str] | None = None,
        failsafe_dropped_current: bool = False,
        failsafe_note_drop_policy: str = "balanced",
        # [dspy_pipeline] optional — LM endpoint
        prompt_lm: str = "",
        base_url: str = "",
        vllm_node_file: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        # [dspy_pipeline] optional — output budgets
        max_tokens: int = 1536,
        identify_max_tokens: int = 2048,
        prompt_lm_max_tokens: int = 5000,
        # [dspy_pipeline] optional — input budgets
        max_demo_tokens: int = 3000,
        max_eval_tokens: int = 3000,
        # [dspy_pipeline] optional — concurrency and sampling
        num_threads: int = 4,
        repetition_penalty: float = 0.0,
        dspy_demo_pool_size: int = 50,
        dspy_opt_val_size: int = 300,
        dspy_demo_min_per_class: int = 1,
        # [dspy_pipeline] optional — Bootstrap hyperparams
        bootstrap_max_bootstrapped_demos: int = 4,
        bootstrap_max_labeled_demos: int = 4,
        bootstrap_num_candidate_programs: int = 10,
        # [dspy_pipeline] optional — MIPROv2 hyperparams
        mipro_num_candidates: int = 10,
        mipro_num_trials: int = 15,
        mipro_max_bootstrapped_demos: int = 4,
        mipro_max_labeled_demos: int = 4,
        mipro_view_data_batch_size: int = 15,
    ):
        for tmp_command in commands:
            if tmp_command not in ALLOWED_COMMANDS:
                raise ValueError(
                    f"Invalid command '{tmp_command}' in config. "
                    f"Allowed commands are: {ALLOWED_COMMANDS}"
                )

        self.commands = commands
        self.seed = seed
        self.use_identification = use_identification
        self.use_windowed_classification_context = use_windowed_classification_context

        self.task_lm = task_lm
        self.optimiser = optimiser

        self.token_file = hf_token_file
        self.pseudonymise = pseudonymise
        self.pseudonymisation_data_dir = pseudonymisation_data_dir
        self.n_bootstrap = n_bootstrap
        self.failsafe_dropped_current = failsafe_dropped_current
        self.failsafe_note_drop_policy = failsafe_note_drop_policy

        resolved_modes = (
            baseline_feature_modes
            if baseline_feature_modes is not None
            else ["base_text", "text"]
        )
        invalid = [m for m in resolved_modes if m not in _VALID_FEATURE_MODES]
        if invalid:
            raise ValueError(
                f"Invalid baseline_feature_modes value(s): {invalid}. "
                f"Each must be one of {sorted(_VALID_FEATURE_MODES)}."
            )
        self.baseline_feature_modes: list[str] = resolved_modes

        self.prompt_lm = prompt_lm
        self.base_url = base_url
        self.vllm_node_file = vllm_node_file
        self.api_key_env = api_key_env

        self.max_tokens = max_tokens
        self.identify_max_tokens = identify_max_tokens
        self.prompt_lm_max_tokens = prompt_lm_max_tokens

        self.max_demo_tokens = max_demo_tokens
        self.max_eval_tokens = max_eval_tokens

        self.num_threads = num_threads
        self.repetition_penalty = repetition_penalty
        self.dspy_demo_pool_size = dspy_demo_pool_size
        self.dspy_opt_val_size = dspy_opt_val_size
        self.dspy_demo_min_per_class = dspy_demo_min_per_class

        self.bootstrap_max_bootstrapped_demos = bootstrap_max_bootstrapped_demos
        self.bootstrap_max_labeled_demos = bootstrap_max_labeled_demos
        self.bootstrap_num_candidate_programs = bootstrap_num_candidate_programs

        self.mipro_num_candidates = mipro_num_candidates
        self.mipro_num_trials = mipro_num_trials
        self.mipro_max_bootstrapped_demos = mipro_max_bootstrapped_demos
        self.mipro_max_labeled_demos = mipro_max_labeled_demos
        self.mipro_view_data_batch_size = mipro_view_data_batch_size

    def checkpoint_snapshot(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "task_lm": self.task_lm,
            "prompt_lm": self.prompt_lm,
            "optimiser": self.optimiser,
            "max_tokens": self.max_tokens,
            "identify_max_tokens": self.identify_max_tokens,
            "prompt_lm_max_tokens": self.prompt_lm_max_tokens,
            "max_demo_tokens": self.max_demo_tokens,
            "max_eval_tokens": self.max_eval_tokens,
            "dspy_demo_pool_size": self.dspy_demo_pool_size,
            "dspy_opt_val_size": self.dspy_opt_val_size,
            "dspy_demo_min_per_class": self.dspy_demo_min_per_class,
            "bootstrap_max_bootstrapped_demos": self.bootstrap_max_bootstrapped_demos,
            "bootstrap_max_labeled_demos": self.bootstrap_max_labeled_demos,
            "bootstrap_num_candidate_programs": self.bootstrap_num_candidate_programs,
            "mipro_num_candidates": self.mipro_num_candidates,
            "mipro_num_trials": self.mipro_num_trials,
            "mipro_max_bootstrapped_demos": self.mipro_max_bootstrapped_demos,
            "mipro_max_labeled_demos": self.mipro_max_labeled_demos,
            "mipro_view_data_batch_size": self.mipro_view_data_batch_size,
            "failsafe_dropped_current": self.failsafe_dropped_current,
            "failsafe_note_drop_policy": self.failsafe_note_drop_policy,
        }


def ensure_dir_exists(folder_path: str | None) -> None:
    if folder_path is None:
        return
    os.makedirs(folder_path, exist_ok=True)
    print(f"Ensured existence of folder: {folder_path}")


def check_config(given_config: dict, required_keys: list[str]) -> None:
    missing_keys = [k for k in required_keys if k not in given_config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def load_config(config_path: str) -> Config:
    if not config_path.endswith(".toml"):
        raise ValueError(f"Configuration file must be a .toml file, got: {config_path}")

    try:
        with open(config_path, "rb") as f:
            raw_config = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(
            f"Failed to parse TOML configuration file '{config_path}': {e}"
        ) from e

    print(f"Loaded configuration from {config_path}")

    # Stage 1: validate top-level required keys
    check_config(raw_config, TOP_LEVEL_REQUIRED_KEYS)

    # Stage 2: extract and validate [dspy_pipeline] section
    dspy_pipeline: dict = raw_config.pop("dspy_pipeline", {})
    check_config(dspy_pipeline, DSPY_PIPELINE_REQUIRED_KEYS)

    # Merge top-level keys with [dspy_pipeline] keys into a flat kwargs dict
    kwargs: dict = {k: raw_config[k] for k in TOP_LEVEL_REQUIRED_KEYS}
    for k in TOP_LEVEL_OPTIONAL_KEYS:
        if k in raw_config:
            kwargs[k] = raw_config[k]
    kwargs.update(dspy_pipeline)

    return Config(**kwargs)


def parse_args() -> tuple[argparse.Namespace, Config]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        "-s",
        dest="data_folder",
        help="Path to the data folder.",
        required=True,
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="Path to the configuration TOML file.",
        required=True,
    )
    parser.add_argument(
        "--reset-previous-data",
        action="store_true",
        dest="reset_previous_data",
        help="Reset previous data before running the command.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError(f"Configuration file not found: {args.config}")

    config = load_config(args.config)
    ensure_dir_exists(args.data_folder)

    return args, config
