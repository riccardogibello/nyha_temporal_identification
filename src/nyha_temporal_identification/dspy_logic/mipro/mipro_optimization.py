import csv
from pathlib import Path
from typing import Any, List

import dspy
from dspy.teleprompt import MIPROv2

from nyha_temporal_identification.dspy_logic.mipro.examples_selector import (
    reorder_by_rarity,
)
from nyha_temporal_identification.dspy_logic.modules import (
    NYHAExtractor,
    build_nyha_extractor,
    get_extractor_run_variant,
)
from nyha_temporal_identification.dspy_logic.optimizer import (
    load_optimised_extractor,
)
from nyha_temporal_identification.evaluation.metrics_computation import nyha_dspy_metric


def _recover_best_from_log_dir(
    mipro_log_dir: Path,
    output_dir: Path,
    identify_lm: dspy.LM,
    classify_lm: dspy.LM,
    use_identification: bool,
    use_windowed_classification_context: bool,
    tokenizer: Any = None,
    max_eval_tokens: int = 0,
) -> NYHAExtractor | None:
    """
    Scan mipro_log/ for candidate programs saved during a previous (possibly
    interrupted) MIPROv2 run and promote the one with the highest full-eval
    score to output_dir/optimised_mipro.json.

    DSPy saves candidate programs only at full-evaluation checkpoints
    (every minibatch_full_eval_steps trials and at the final trial).
    Each JSON filename is candidate_program_{i}.json where i is an
    integer index — NOT a trial number. The score for each checkpoint
    is recorded in the accompanying trial_scores.csv if the run
    completed cleanly, or must be inferred from the filename index
    if the run was interrupted.

    Returns the loaded best extractor, or None if no candidates exist.
    """
    candidates = sorted(mipro_log_dir.glob("candidate_program_*.json"))
    if not candidates:
        return None

    # Try to read scores from trial_scores.csv (present on clean runs)
    trial_csv = mipro_log_dir / "trial_scores.csv"
    score_map: dict[str, float] = {}
    if trial_csv.exists():
        import csv

        with open(trial_csv, "r") as f:
            for row in csv.DictReader(f):
                path = row.get("full_eval_program_path", "")
                score = row.get("full_eval_score")
                if path and score is not None:
                    try:
                        score_map[Path(path).name] = float(score)
                    except ValueError:
                        pass

    if score_map:
        # Pick the candidate with the highest recorded full-eval score
        best_file = max(
            candidates,
            key=lambda p: score_map.get(p.name, -1.0),
        )
        best_score = score_map.get(best_file.name, "unknown")
    else:
        # No CSV available (interrupted run): fall back to the last-written
        # file, which is the most recent full-eval checkpoint
        best_file = candidates[-1]
        best_score = "unknown (run was interrupted)"

    print(f"  [RECOVERY] Best candidate: {best_file.name}  score={best_score}")
    extractor = load_optimised_extractor(
        str(best_file),
        identify_lm=identify_lm,
        classify_lm=classify_lm,
        use_identification=use_identification,
        use_windowed_classification_context=use_windowed_classification_context,
        tokenizer=tokenizer,
        max_eval_tokens=max_eval_tokens,
    )

    variant = get_extractor_run_variant(
        use_identification,
        use_windowed_classification_context,
    )
    save_path = output_dir / f"optimised_mipro_{variant}.json"
    extractor.save(str(save_path))
    print(f"  [RECOVERY] Promoted to {save_path}")

    return extractor


def _get_instructions(pred: dspy.Predict) -> str:
    """
    Extract the instruction string from a predictor, compatible across DSPy versions.
    DSPy >= 2.4 stores instructions in the Signature class docstring.
    """
    sig = getattr(pred, "extended_signature", None) or pred.signature
    if hasattr(sig, "__doc__") and sig.__doc__:
        return sig.__doc__
    return getattr(sig, "instructions", "")  # type: ignore[attr-defined]


def step3b_mipro(
    trainset: List[dspy.Example],
    valset: List[dspy.Example],
    output_dir: Path,
    identify_lm: dspy.LM,
    classify_lm: dspy.LM,
    prompt_lm: dspy.LM,
    num_trials: int = 15,
    num_candidates: int = 10,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    num_threads: int = 4,
    use_identification: bool = True,
    use_windowed_classification_context: bool = False,
    view_data_batch_size: int = 15,
    tokenizer: Any = None,
    max_eval_tokens: int = 0,
) -> NYHAExtractor:
    """
    Step 3b — MIPROv2 (Multiprompt Instruction PRoposal Optimizer v2).
    ...
    """
    print("\n[STEP 3b] MIPROv2 optimisation (instruction + demo search)...")

    print(
        f"  [MIPRO] Configuration: num_trials={num_trials}, "
        f"num_candidates={num_candidates}, "
        f"view_data_batch_size={view_data_batch_size}, "
        f"max_bootstrapped_demos={max_bootstrapped_demos}, "
        f"max_labeled_demos={max_labeled_demos}, "
        f"num_threads={num_threads}"
    )

    # ── Early warnings ────────────────────────────────────────────────────────
    if len(valset) < 20:
        print(
            f"  [WARNING] valset has only {len(valset)} examples. "
            "The optimizer may lack enough signal to prefer any instruction rewrite. "
            "Consider using at least 20–50 validation examples."
        )
    if num_candidates < 2 * num_trials:
        print(
            f"  [WARNING] num_candidates={num_candidates} is less than "
            f"2 × num_trials={num_trials}. The Bayesian search may converge back "
            "to the original instruction. Increase num_candidates."
        )

    # ── Fast exit if already completed ───────────────────────────────────────
    variant = get_extractor_run_variant(
        use_identification,
        use_windowed_classification_context,
    )
    save_path = output_dir / f"optimised_mipro_{variant}.json"
    if save_path.exists():
        print(f"  Cached MIPRO extractor found at {save_path}; loading it.")
        return load_optimised_extractor(
            str(save_path),
            identify_lm=identify_lm,
            classify_lm=classify_lm,
            use_identification=use_identification,
            use_windowed_classification_context=use_windowed_classification_context,
            tokenizer=tokenizer,
            max_eval_tokens=max_eval_tokens,
        )

    # ── Recovery: if a previous run left checkpoints, promote the best one ───
    mipro_log_dir = output_dir / "mipro_log"
    mipro_log_dir.mkdir(parents=True, exist_ok=True)

    recovered = _recover_best_from_log_dir(
        mipro_log_dir,
        output_dir,
        identify_lm=identify_lm,
        classify_lm=classify_lm,
        use_identification=use_identification,
        use_windowed_classification_context=use_windowed_classification_context,
        tokenizer=tokenizer,
        max_eval_tokens=max_eval_tokens,
    )
    if recovered is not None:
        print("  [RECOVERY] Skipping reoptimisation; using recovered extractor.")
        return recovered

    # ── Build extractor and snapshot original instructions ───────────────────
    extractor = build_nyha_extractor(
        identify_lm=identify_lm,
        classify_lm=classify_lm,
        use_identification=use_identification,
        use_windowed_classification_context=use_windowed_classification_context,
        tokenizer=tokenizer,
        max_eval_tokens=max_eval_tokens,
        failsafe_dropped_current=False,
    )
    original_instructions: dict[str, str] = {
        name: _get_instructions(pred) for name, pred in extractor.named_predictors()
    }

    # ── Reorder trainset by rarity of NYHA class × temporality combinations ──
    ordered_trainset = reorder_by_rarity(
        trainset,
    )
    print(
        f"  [MIPRO] Trainset reordered by rarity ({len(ordered_trainset)} examples); "
        f"first {view_data_batch_size} will be used for dataset summary."
    )

    # ── Fresh optimisation run ────────────────────────────────────────────────
    optimizer = MIPROv2(
        metric=nyha_dspy_metric,
        prompt_model=prompt_lm,
        task_model=classify_lm,
        auto=None,
        num_candidates=num_candidates,
        num_threads=num_threads,
        verbose=True,
        log_dir=str(mipro_log_dir),
        track_stats=True,
        metric_threshold=0.85,
    )

    optimised_extractor = optimizer.compile(
        extractor,
        trainset=ordered_trainset,  # reordered — rarest combos first
        valset=valset,
        num_trials=num_trials,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        minibatch_size=min(25, len(valset)),
        minibatch_full_eval_steps=max(1, min(5, num_trials)),
        requires_permission_to_run=False,
        view_data_batch_size=view_data_batch_size,
        tip_aware_proposer=False,  # adds noise without clear benefit
        program_aware_proposer=False,  # signature already describes the task
        data_aware_proposer=True,  # rewrites instructions from dataset patterns
        fewshot_aware_proposer=True,  # grounds proposals in real examples
    )

    # ── Validate that instructions actually changed ───────────────────────────
    changed_predictors: list[str] = []
    unchanged_predictors: list[str] = []
    for name, pred in optimised_extractor.named_predictors():
        compiled_instr = _get_instructions(pred)
        if compiled_instr != original_instructions.get(name, ""):
            changed_predictors.append(name)
        else:
            unchanged_predictors.append(name)

    if unchanged_predictors:
        print(
            f"  [WARNING] Predictors with unchanged instructions after compilation: "
            f"{unchanged_predictors}. The optimizer may have converged back to the "
            "original prompt — consider increasing num_candidates or valset size."
        )
    if changed_predictors:
        print(f"  [INFO] Instructions updated for predictors: {changed_predictors}")

    # The comparable performance report is produced on the held-out test set by
    # step4_final_evaluation (stage "final_mipro"); no per-mention valset report
    # is emitted here, as it used a different set and granularity than the other
    # stages and was not comparable.

    # ── Persist trial-level scores ────────────────────────────────────────────
    trial_logs = getattr(optimised_extractor, "trial_logs", None)
    if isinstance(trial_logs, dict) and trial_logs:
        rows: List[dict[str, Any]] = [
            v for v in trial_logs.values() if isinstance(v, dict)
        ]
        if rows:
            trial_csv_path = mipro_log_dir / "trial_scores.csv"
            fieldnames = list(rows[0].keys())
            with open(trial_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Trial scores written to {trial_csv_path}")

    # ── Save final best program ───────────────────────────────────────────────
    optimised_extractor.save(str(save_path))
    print(f"  Optimised program saved to {save_path}")

    return optimised_extractor
