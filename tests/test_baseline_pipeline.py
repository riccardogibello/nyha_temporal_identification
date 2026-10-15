"""Unit tests for feature_mode routing in models/pipeline.py.

All tests are pure unit tests — no real data is required. They verify:
- feature_mode derivation of has_text / has_numeric
- numeric feature list is empty in text mode
- ThresholdClassifier is absent from results in text mode
- Invalid feature_mode raises ValueError
- Per-mode output paths do not collide
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nyha_temporal_identification.models.constants import (
    BASELINES_FOLDER_NAME,
    FEATURE_MODE_BASE,
    FEATURE_MODE_BASE_TEXT,
    FEATURE_MODE_TEXT,
    FEATURE_MODES,
)
import polars as pl

from nyha_temporal_identification.models.features_handling import (
    analyse_features,
    build_feature_matrix,
)
from nyha_temporal_identification.models.pipeline import train_baselines


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_feature_mode_constants_values():
    assert FEATURE_MODE_BASE == "base"
    assert FEATURE_MODE_BASE_TEXT == "base_text"
    assert FEATURE_MODE_TEXT == "text"
    assert set(FEATURE_MODES) == {"base", "base_text", "text"}
    assert BASELINES_FOLDER_NAME == "baselines"


# ---------------------------------------------------------------------------
# feature_mode validation
# ---------------------------------------------------------------------------

def test_train_baselines_invalid_feature_mode(tmp_path):
    with pytest.raises(ValueError, match="bad_mode"):
        train_baselines(
            base_folder_path=str(tmp_path),
            documents=[],
            feature_mode="bad_mode",
        )


# ---------------------------------------------------------------------------
# Path isolation: base vs text paths must not collide
# ---------------------------------------------------------------------------

def test_per_mode_paths_do_not_collide():
    """Output paths for 'base' and 'text' must differ at the feature_mode segment."""
    base = Path("/data")
    path_base = base / "temporal_status_prediction" / BASELINES_FOLDER_NAME / "cv" / FEATURE_MODE_BASE
    path_text = base / "temporal_status_prediction" / BASELINES_FOLDER_NAME / "cv" / FEATURE_MODE_TEXT
    assert path_base != path_text
    assert str(path_base) != str(path_text)
    # The only differing segment should be the feature_mode slug
    assert path_base.parent == path_text.parent


# ---------------------------------------------------------------------------
# Feature mode routing via a minimal in-process smoke test
# ---------------------------------------------------------------------------

def _minimal_doc(note_id: str = "note_001", temporality: str = "current"):
    """Return a mock NYHANote with one mention of the given temporality."""
    mention = MagicMock()
    mention.temporality = temporality
    mention.char_distance = 10
    mention.linked_date = None

    doc = MagicMock()
    doc.old_note_id = note_id
    doc.mentions = [mention]
    return doc


def _four_docs():
    """Four docs (2 current, 2 historical) — enough for n_splits=2 without stratification."""
    return [
        _minimal_doc("note_001", "current"),
        _minimal_doc("note_002", "current"),
        _minimal_doc("note_003", "historical"),
        _minimal_doc("note_004", "historical"),
    ]


_DEFAULT_NOTE_TO_GROUP = {
    "note_001": "patient_A",
    "note_002": "patient_A",
    "note_003": "patient_B",
    "note_004": "patient_B",
}


def _patch_pipeline(note_to_group: dict | None = None):
    """Context manager that patches away all I/O and model-fitting in pipeline.py."""
    if note_to_group is None:
        note_to_group = _DEFAULT_NOTE_TO_GROUP
    patches = [
        patch(
            "nyha_temporal_identification.models.pipeline.load_note_patient_groups",
            return_value=note_to_group,
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.build_train_val_dfs",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.build_test_dfs",
            return_value=(MagicMock(), [], []),
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.analyse_features",
            return_value=([], {}),
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.build_feature_matrix",
            return_value=MagicMock(),
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.build_logistic_regression",
            return_value=MagicMock(),
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.build_gradient_boosting",
            return_value=MagicMock(),
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.plot_feature_importance",
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.evaluate_model",
            side_effect=lambda name, **kw: {"merged_micro_f1": 0.5},
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.TextFeatureExtractor",
        ),
        patch(
            "nyha_temporal_identification.models.pipeline.ThresholdClassifier",
            return_value=MagicMock(),
        ),
    ]
    return patches


def _apply_patches(patches):
    started = [p.start() for p in patches]
    return started, patches


def _stop_patches(patches):
    for p in patches:
        p.stop()


@pytest.fixture
def patched_pipeline():
    patches = _patch_pipeline()
    mocks, _ = _apply_patches(patches)
    yield mocks
    _stop_patches(patches)


def test_text_mode_no_threshold_in_results(tmp_path, patched_pipeline):
    """text mode must not include threshold_classifier in results."""
    docs = _four_docs()
    results = train_baselines(
        base_folder_path=str(tmp_path),
        documents=docs,
        feature_mode=FEATURE_MODE_TEXT,
        n_splits=2,
    )
    assert "threshold_classifier" not in results
    assert "logistic_regression" in results
    assert "gradient_boosting" in results


def test_base_mode_has_threshold_in_results(tmp_path, patched_pipeline):
    """base mode must include threshold_classifier in results."""
    docs = _four_docs()
    results = train_baselines(
        base_folder_path=str(tmp_path),
        documents=docs,
        feature_mode=FEATURE_MODE_BASE,
        n_splits=2,
    )
    assert "threshold_classifier" in results


def test_text_mode_no_threshold_folder_created(tmp_path, patched_pipeline):
    """text mode must not create threshold_classifier/ output folder."""
    docs = _four_docs()
    train_baselines(
        base_folder_path=str(tmp_path),
        documents=docs,
        feature_mode=FEATURE_MODE_TEXT,
        n_splits=2,
    )
    # Walk created dirs — threshold_classifier must be absent under text/
    created = {p.name for p in tmp_path.rglob("*") if p.is_dir()}
    assert "threshold_classifier" not in created


def test_feature_mode_inserted_in_output_path(tmp_path, patched_pipeline):
    """feature_mode slug must appear in the output paths created on disk."""
    docs = _four_docs()
    train_baselines(
        base_folder_path=str(tmp_path),
        documents=docs,
        feature_mode=FEATURE_MODE_BASE_TEXT,
        n_splits=2,
    )
    created_paths = [str(p) for p in tmp_path.rglob("*") if p.is_dir()]
    assert any(FEATURE_MODE_BASE_TEXT in p for p in created_paths)


# ---------------------------------------------------------------------------
# analyse_features: empty feature list (text-only mode)
# ---------------------------------------------------------------------------

def test_build_feature_matrix_empty_numeric_returns_bow():
    """build_feature_matrix with no numeric features must return the BoW matrix as-is.

    Regression test: before the fix, sp.csr_matrix(to_numpy_features(df, []))
    produced a (0,0)-shaped sparse matrix that caused an hstack dimension mismatch.
    """
    import scipy.sparse as sp

    bow = sp.csr_matrix([[1, 0], [0, 1], [1, 1]])
    df = pl.DataFrame({"target": [0, 1, 0]})
    result = build_feature_matrix(df, [], bow)
    assert (result - bow).nnz == 0
    assert result.shape == (3, 2)


def test_build_feature_matrix_empty_numeric_no_bow_raises():
    """build_feature_matrix with no numeric features and no bow_matrix must raise ValueError."""
    df = pl.DataFrame({"target": [0, 1, 0]})
    with pytest.raises(ValueError, match="no features available"):
        build_feature_matrix(df, [], None)


def test_analyse_features_empty_list_returns_empty():
    """analyse_features must return ([], {}) when no numeric features are given.

    This is the live path for feature_mode='text'. Before the fix, passing an
    empty list caused mutual_info_classif to receive an (0, 0)-shaped array
    and raise ValueError.
    """
    df = pl.DataFrame({"target": [0, 1, 0, 1]})
    selected, scores = analyse_features(df, [])
    assert selected == []
    assert scores == {}


# ---------------------------------------------------------------------------
# Patient-grouped CV: no patient leaks across outer or inner split boundaries
# ---------------------------------------------------------------------------

_SIX_DOC_NOTE_TO_GROUP = {
    "n1": "pA", "n2": "pA",
    "n3": "pB", "n4": "pB",
    "n5": "pC", "n6": "pC",
}


def _six_docs():
    """Six docs across 3 patients (2 each), alternating temporality."""
    pairs = [
        ("n1", "current"), ("n2", "historical"),
        ("n3", "current"), ("n4", "historical"),
        ("n5", "current"), ("n6", "historical"),
    ]
    return [_minimal_doc(nid, t) for nid, t in pairs]


def test_outer_fold_no_patient_overlap(tmp_path):
    """No patient should appear in both train_ids and test_ids in any outer fold."""
    docs = _six_docs()
    note_to_group = _SIX_DOC_NOTE_TO_GROUP

    patches = _patch_pipeline(note_to_group=note_to_group)
    mocks, _ = _apply_patches(patches)
    try:
        train_baselines(
            base_folder_path=str(tmp_path),
            documents=docs,
            feature_mode=FEATURE_MODE_BASE,
            n_splits=2,
        )
    finally:
        _stop_patches(patches)

    # Locate the baselines_splits.json written by _train_kfold_cv
    splits_files = list(tmp_path.rglob("baselines_splits.json"))
    assert splits_files, "baselines_splits.json was not written"
    with open(splits_files[0]) as f:
        data = json.load(f)

    assert data.get("grouping") == "patient", "grouping key must be 'patient'"
    for fold in data["folds"]:
        train_patients = {note_to_group[nid] for nid in fold["train_ids"]}
        test_patients = {note_to_group[nid] for nid in fold["test_ids"]}
        assert train_patients.isdisjoint(test_patients), (
            f"Fold {fold['fold']}: patient(s) {train_patients & test_patients} "
            "appear in both train and test"
        )


def test_inner_split_no_patient_overlap(tmp_path):
    """The inner train/val split must not put the same patient in both sides."""
    docs = _six_docs()
    note_to_group = _SIX_DOC_NOTE_TO_GROUP

    inner_train_id_calls: list[list[str]] = []
    inner_val_id_calls: list[list[str]] = []

    original_build = __import__(
        "nyha_temporal_identification.models.data_loading",
        fromlist=["build_train_val_dfs"],
    ).build_train_val_dfs

    def capturing_build(documents, train_ids, val_ids):
        inner_train_id_calls.append(list(train_ids))
        inner_val_id_calls.append(list(val_ids))
        return MagicMock(), MagicMock()

    patches = _patch_pipeline(note_to_group=note_to_group)
    # Replace the build_train_val_dfs patch with our capturing version
    patches[1] = patch(
        "nyha_temporal_identification.models.pipeline.build_train_val_dfs",
        side_effect=capturing_build,
    )
    mocks, _ = _apply_patches(patches)
    try:
        train_baselines(
            base_folder_path=str(tmp_path),
            documents=docs,
            feature_mode=FEATURE_MODE_BASE,
            n_splits=2,
        )
    finally:
        _stop_patches(patches)

    assert inner_train_id_calls, "build_train_val_dfs was never called"
    for train_ids_inner, val_ids_inner in zip(inner_train_id_calls, inner_val_id_calls):
        train_patients = {note_to_group[nid] for nid in train_ids_inner if nid in note_to_group}
        val_patients = {note_to_group[nid] for nid in val_ids_inner if nid in note_to_group}
        assert train_patients.isdisjoint(val_patients), (
            f"Inner split: patient(s) {train_patients & val_patients} "
            "appear in both train and val"
        )
