"""Tests for load_config two-stage validation and Config.checkpoint_snapshot."""

import os
import tempfile

import pytest

from nyha_temporal_identification.utils.general import Config, load_config


def _write_toml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False)
    f.write(content)
    f.flush()
    f.close()
    return f.name


MINIMAL_VALID = """
commands = ["extract_nyha"]
seed = 42
use_identification = false
use_windowed_classification_context = true

[dspy_pipeline]
task_lm = "google/gemma-3-12b-it"
optimiser = "both"
"""


def test_load_config_valid():
    path = _write_toml(MINIMAL_VALID)
    try:
        config = load_config(path)
        assert config.commands == ["extract_nyha"]
        assert config.seed == 42
        assert config.use_identification is False
        assert config.use_windowed_classification_context is True
        assert config.task_lm == "google/gemma-3-12b-it"
        assert config.optimiser == "both"
    finally:
        os.unlink(path)


def test_load_config_defaults_populated():
    path = _write_toml(MINIMAL_VALID)
    try:
        config = load_config(path)
        assert config.num_threads == 4
        assert config.max_tokens == 1536
        assert config.identify_max_tokens == 2048
        assert config.max_demo_tokens == 3000
        assert config.max_eval_tokens == 3000
        assert config.mipro_num_trials == 15
        assert config.mipro_num_candidates == 10
        assert config.bootstrap_max_bootstrapped_demos == 4
        assert config.bootstrap_max_labeled_demos == 4
        assert config.mipro_view_data_batch_size == 15
        assert config.n_bootstrap == 0
    finally:
        os.unlink(path)


def test_load_config_n_bootstrap_override():
    toml = """
commands = ["train_baselines"]
seed = 42
use_identification = false
use_windowed_classification_context = true
n_bootstrap = 1000

[dspy_pipeline]
task_lm = "google/gemma-3-12b-it"
optimiser = "both"
"""
    path = _write_toml(toml)
    try:
        config = load_config(path)
        assert config.n_bootstrap == 1000
    finally:
        os.unlink(path)


def test_load_config_optional_override():
    toml = """
commands = ["run_dspy_pipeline"]
seed = 7
use_identification = true
use_windowed_classification_context = false
pseudonymise = true

[dspy_pipeline]
task_lm = "google/gemma-3-27b-it"
optimiser = "bootstrap"
num_threads = 8
max_tokens = 1024
mipro_num_trials = 20
mipro_view_data_batch_size = 10
"""
    path = _write_toml(toml)
    try:
        config = load_config(path)
        assert config.pseudonymise is True
        assert config.num_threads == 8
        assert config.max_tokens == 1024
        assert config.mipro_num_trials == 20
        assert config.mipro_view_data_batch_size == 10
    finally:
        os.unlink(path)


def test_load_config_missing_top_level_required_key():
    toml = """
seed = 42
use_identification = false
use_windowed_classification_context = true

[dspy_pipeline]
task_lm = "google/gemma-3-12b-it"
optimiser = "both"
"""
    path = _write_toml(toml)
    try:
        with pytest.raises(ValueError, match="Missing required"):
            load_config(path)
    finally:
        os.unlink(path)


def test_load_config_missing_dspy_pipeline_key():
    toml = """
commands = ["run_dspy_pipeline"]
seed = 42
use_identification = false
use_windowed_classification_context = true

[dspy_pipeline]
optimiser = "both"
"""
    path = _write_toml(toml)
    try:
        with pytest.raises(ValueError, match="Missing required"):
            load_config(path)
    finally:
        os.unlink(path)


def test_load_config_non_toml_extension():
    with pytest.raises(ValueError, match=".toml"):
        load_config("config.yaml")


def test_baseline_feature_modes_default():
    path = _write_toml(MINIMAL_VALID)
    try:
        config = load_config(path)
        assert config.baseline_feature_modes == ["base_text", "text"]
    finally:
        os.unlink(path)


def test_baseline_feature_modes_custom():
    toml = """
commands = ["train_baselines"]
seed = 42
use_identification = false
use_windowed_classification_context = true
baseline_feature_modes = ["base"]

[dspy_pipeline]
task_lm = "google/gemma-3-12b-it"
optimiser = "both"
"""
    path = _write_toml(toml)
    try:
        config = load_config(path)
        assert config.baseline_feature_modes == ["base"]
    finally:
        os.unlink(path)


def test_baseline_feature_modes_invalid():
    toml = """
commands = ["train_baselines"]
seed = 42
use_identification = false
use_windowed_classification_context = true
baseline_feature_modes = ["base_text", "bad_mode"]

[dspy_pipeline]
task_lm = "google/gemma-3-12b-it"
optimiser = "both"
"""
    path = _write_toml(toml)
    try:
        with pytest.raises(ValueError, match="bad_mode"):
            load_config(path)
    finally:
        os.unlink(path)


def test_checkpoint_snapshot_keys():
    path = _write_toml(MINIMAL_VALID)
    try:
        config = load_config(path)
        snap = config.checkpoint_snapshot()
        expected_keys = {
            "seed", "task_lm", "prompt_lm", "optimiser",
            "max_tokens", "identify_max_tokens", "prompt_lm_max_tokens",
            "max_demo_tokens", "max_eval_tokens",
            "dspy_demo_pool_size", "dspy_opt_val_size", "dspy_demo_min_per_class",
            "bootstrap_max_bootstrapped_demos", "bootstrap_max_labeled_demos",
            "bootstrap_num_candidate_programs",
            "mipro_num_candidates", "mipro_num_trials",
            "mipro_max_bootstrapped_demos", "mipro_max_labeled_demos",
            "mipro_view_data_batch_size",
        }
        assert expected_keys <= set(snap.keys())
        assert snap["task_lm"] == "google/gemma-3-12b-it"
        assert snap["seed"] == 42
    finally:
        os.unlink(path)
