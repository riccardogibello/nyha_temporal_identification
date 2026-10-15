"""
Checkpoint management for resumeable pipeline execution on HPC systems.

Saves/loads pipeline state to allow resuming from the last completed step if
a batch job is interrupted. Includes trial-level checkpointing for long-running
optimisation steps like MIPROv2.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from nyha_temporal_identification.constants import (
    ID_SPLITS_FILE_NAME,
    SPLITS_FOLDER_NAME,
)


CHECKPOINT_CRITICAL_KEYS = frozenset({"task_lm"})


class PipelineCheckpoint:
    """Manages checkpoint state for the optimization pipeline."""

    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_FILE = "pipeline_state.json"
    EXTRACTORS_DIR = "extractors"

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / self.CHECKPOINT_DIR
        self.extractors_dir = self.checkpoint_dir / self.EXTRACTORS_DIR
        self.checkpoint_file = self.checkpoint_dir / self.CHECKPOINT_FILE
        self.splits_dir = self.output_dir / SPLITS_FOLDER_NAME
        self.split_file = self.splits_dir / ID_SPLITS_FILE_NAME

    def exists(self) -> bool:
        """Check if a checkpoint exists."""
        return self.checkpoint_file.exists()

    def save(
        self,
        step: int,
        best: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        **state_data,
    ) -> None:
        """
        Save checkpoint state after a step completes.

        Args:
            step: Step number (1, 2, 3a, 3b, 4)
            best: Best model label, if known (baseline, bootstrap, or mipro)
            config_snapshot: Snapshot of the current pipeline config.
            **state_data: Additional state to save (extractor names, predictions, etc.)
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.extractors_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_state = {
            "last_completed_step": step,
            "best": best,
            "config": config_snapshot,
            "state": state_data,
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint_state, f, indent=2)
        print(f"  [CHECKPOINT] Saved state after step {step}")

    def load(self) -> Dict[str, Any] | None:
        """
        Load checkpoint state.

        Returns:
            Dict with 'last_completed_step' and 'state' keys.
        """
        if not self.exists():
            return None

        with open(self.checkpoint_file, "r") as f:
            checkpoint_state = json.load(f)
        return checkpoint_state

    def save_document_id_splits(
        self,
        train_ids: list[str],
        val_ids: list[str],
        test_ids: list[str],
        *,
        seed: Optional[int] = None,
        total_documents: Optional[int] = None,
        grouping: Optional[str] = None,
        group_counts: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the train/val/test document IDs to a JSON cache file."""
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        split_cache = {
            "version": 2,
            "seed": seed,
            "total_documents": total_documents,
            "grouping": grouping,
            "group_counts": group_counts,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        }

        with open(self.split_file, "w") as f:
            json.dump(split_cache, f, indent=2)

        print(f"  [CHECKPOINT] Saved document ID splits to {self.split_file}")

    def load_document_id_splits(self) -> Dict[str, Any] | None:
        """Load the cached train/val/test document IDs, if available."""
        if not self.split_file.exists():
            return None

        with open(self.split_file, "r") as f:
            split_cache = json.load(f)
        return split_cache

    def should_skip_step1(self) -> bool:
        """Check if step 1 should be skipped (already completed)."""
        checkpoint = self.load()
        return bool(checkpoint and checkpoint.get("last_completed_step", 0) >= 1)

    def should_skip_step2(self) -> bool:
        """Check if step 2 should be skipped (already completed)."""
        checkpoint = self.load()
        return bool(checkpoint and checkpoint.get("last_completed_step", 0) >= 2)

    def should_skip_step3a(self) -> bool:
        """Check if step 3a should be skipped (already completed)."""
        checkpoint = self.load()
        return bool(checkpoint and checkpoint.get("last_completed_step", 0) >= 3)

    def should_skip_step3b(self) -> bool:
        """Check if step 3b should be skipped (already completed)."""
        checkpoint = self.load()
        return bool(checkpoint and checkpoint.get("last_completed_step", 0) >= 4)

    def get_last_completed_step(self) -> int:
        """Get the number of the last completed step."""
        checkpoint = self.load()
        return checkpoint.get("last_completed_step", 0) if checkpoint else 0

    def get_cached_state(self, key: str) -> Optional[Any]:
        """Get a cached state value from the checkpoint."""
        checkpoint = self.load()
        if checkpoint:
            return checkpoint.get("state", {}).get(key)
        return None

    def get_config_snapshot(self) -> Dict[str, Any] | None:
        """Get the config snapshot stored in the checkpoint, if present."""
        checkpoint = self.load()
        if checkpoint:
            config_snapshot = checkpoint.get("config")
            if isinstance(config_snapshot, dict):
                return config_snapshot
        return None

    def config_matches(self, current_config: Dict[str, Any]) -> bool:
        """Check whether the stored config snapshot matches the current config."""
        stored_config = self.get_config_snapshot()
        if stored_config is None:
            return False
        return stored_config == current_config

    def assert_config_matches(self, current_config: Dict[str, Any]) -> None:
        """Raise on critical key mismatch; warn and proceed on non-critical differences."""
        stored_config = self.get_config_snapshot()
        if stored_config is None:
            raise ValueError(
                f"Checkpoint at {self.checkpoint_file} does not contain a config snapshot."
            )

        diffs = {
            k: (stored_config.get(k), current_config.get(k))
            for k in set(stored_config) | set(current_config)
            if stored_config.get(k) != current_config.get(k)
        }
        if not diffs:
            return

        critical = {k: v for k, v in diffs.items() if k in CHECKPOINT_CRITICAL_KEYS}
        non_critical = {k: v for k, v in diffs.items() if k not in CHECKPOINT_CRITICAL_KEYS}

        if critical:
            detail = ", ".join(
                f"{k}: {v[0]!r} → {v[1]!r}" for k, v in sorted(critical.items())
            )
            raise ValueError(
                f"Checkpoint config mismatch on critical key(s) — {detail}"
            )

        detail = ", ".join(
            f"{k}: {v[0]!r} → {v[1]!r}" for k, v in sorted(non_critical.items())
        )
        print(
            f"  [CHECKPOINT] Config differs on non-critical keys (proceeding): {detail}"
        )

    def clear(self) -> None:
        """Clear the checkpoint (for fresh runs)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.checkpoint_dir.exists():
            import shutil

            shutil.rmtree(self.checkpoint_dir)
        print(f"  [CHECKPOINT] Cleared all checkpoints in {self.checkpoint_dir}")
