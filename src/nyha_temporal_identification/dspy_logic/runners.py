"""
runners.py — Pipeline runner and inference utilities.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from dspy.evaluate import Evaluate
from tqdm import tqdm

from nyha_temporal_identification.constants import (
    OPT_SUBSET_FOLDER_NAME,
    SPLITS_FOLDER_NAME,
    ZS_COMPLETE_FOLDER_NAME,
)
from nyha_temporal_identification.data_handling.data_conversion import (
    documents_to_dspy_examples,
    subsample_examples,
)
from nyha_temporal_identification.data_handling.data_loading import (
    load_note_patient_groups,
    patient_grouped_split,
)
from nyha_temporal_identification.data_model.dspy_model import (
    NYHAMention,
    NYHANote,
    nyha_mention_from_json_dict,
)
from nyha_temporal_identification.dspy_logic.checkpoint import PipelineCheckpoint
from nyha_temporal_identification.dspy_logic.failsafe import assess_current_consistency
from nyha_temporal_identification.dspy_logic.mipro.mipro_optimization import (
    step3b_mipro,
)
from nyha_temporal_identification.dspy_logic.modules import NYHAExtractor
from nyha_temporal_identification.dspy_logic.optimizer import (
    configure_lm,
    select_best,
    step1_zero_shot_baseline,
    step2_error_analysis,
    step3a_bootstrap_fewshot,
    step4_final_evaluation,
)
from nyha_temporal_identification.evaluation.metrics_computation import nyha_dspy_metric
from nyha_temporal_identification.utils.general import Config


def _process_single_document(
    doc: NYHANote,
    extractor: NYHAExtractor,
    results_dir: Path,
    failsafe_note_drop_policy: str = "balanced",
) -> tuple[str, list[NYHAMention], bool]:
    """
    Process a single document: check cache, call extractor if needed, write output.

    A cached file is only honoured when it contains a non-empty prediction list.
    Files with an empty ``predictions`` list (e.g. left by a failed or
    incomplete earlier run) are treated as a cache miss and recomputed. Results
    are only written back when the extractor returns at least one mention, so
    failed/empty extractions never poison the cache.

    When extractor.failsafe_dropped_current is True, a "failsafe" block is
    appended to the written JSON (result of assess_current_consistency).

    Returns:
        (doc.old_note_id, predictions: List[NYHAMention], was_cached: bool)
    """
    out_path = results_dir / f"{doc.old_note_id}.json"

    if out_path.exists():
        with open(out_path) as f:
            cached_result = json.load(f)
        predictions = [
            nyha_mention_from_json_dict(prediction)
            for prediction in cached_result.get("predictions", [])
        ]
        if predictions:
            return doc.old_note_id, predictions, True

    try:
        predictions = list(
            extractor(
                deidentified_text=doc.deidentified_text,
                authored_datetime=doc.authored_datetime,
            )
        )
    except Exception as e:
        logging.error(f"Extraction failed for {doc.old_note_id}: {e}")
        predictions = []

    # Only persist genuine, non-empty results. Empty extractions (errors or
    # no mentions found) are left uncached so they are retried on the next run.
    # Documents whose only mentions are failsafe re-injections count as non-empty.
    if predictions:
        output_payload: dict = {
            "old_note_id": doc.old_note_id,
            "deidentified_text": doc.deidentified_text,
            "authored_datetime": doc.authored_datetime,
            "predictions": [m.model_dump() for m in predictions],
        }
        if extractor.failsafe_dropped_current:
            output_payload["failsafe"] = assess_current_consistency(
                predictions, failsafe_note_drop_policy
            )
        with open(out_path, "w") as f:
            json.dump(output_payload, f, indent=2)

    return doc.old_note_id, predictions, False


def run_inference(
    extractor: NYHAExtractor,
    documents: List[NYHANote],
    output_dir: Path,
    stage: str = "unseen",
    max_workers: int = 4,
    failsafe_note_drop_policy: str = "balanced",
) -> None:
    """
    Run inference on all documents with optional multi-threading for I/O-bound vLLM calls.

    max_workers must stay ≤ serve.sh --max-num-seqs. Set to 1 for sequential.

    When extractor.failsafe_dropped_current is True, a per-document "failsafe" block is
    written to each JSON, and an aggregate failsafe_audit_{stage}.json is written after
    the run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / f"inference_results_{stage}"
    results_dir.mkdir(parents=True, exist_ok=True)

    cached_predictions_by_doc_id: dict[str, list[NYHAMention]] = {}
    processed_count = 0
    skipped_count = 0

    if max_workers == 1:
        for doc in tqdm(documents, desc="Inference", unit="doc"):
            doc_id, predictions, was_cached = _process_single_document(
                doc, extractor, results_dir, failsafe_note_drop_policy
            )
            cached_predictions_by_doc_id[doc_id] = predictions
            if was_cached:
                skipped_count += 1
            else:
                processed_count += 1
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_document,
                    doc,
                    extractor,
                    results_dir,
                    failsafe_note_drop_policy,
                ): doc
                for doc in documents
            }

            for future in tqdm(
                as_completed(futures),
                total=len(documents),
                desc="Inference",
                unit="doc",
            ):
                try:
                    doc_id, predictions, was_cached = future.result()
                    cached_predictions_by_doc_id[doc_id] = predictions
                    if was_cached:
                        skipped_count += 1
                    else:
                        processed_count += 1
                except Exception as e:
                    doc = futures[future]
                    logging.error(f"Failed to process document {doc.old_note_id}: {e}")

    print(
        f"  Results saved to {results_dir} "
        f"({processed_count} processed, {skipped_count} cached)"
    )

    # Aggregate failsafe audit when failsafe re-injection is active.
    if extractor.failsafe_dropped_current:
        _write_failsafe_audit(
            cached_predictions_by_doc_id,
            output_dir,
            stage,
            failsafe_note_drop_policy,
        )


def _write_failsafe_audit(
    predictions_by_doc_id: dict[str, list[NYHAMention]],
    output_dir: Path,
    stage: str,
    policy: str,
) -> None:
    """Aggregate failsafe stats across all documents and write the audit JSON."""
    import re

    docs_total = len(predictions_by_doc_id)
    docs_with_reinjection = 0
    docs_flagged_conflict = 0
    reinjected_total = 0
    reinjected_by_class: dict[str, int] = {}
    reinjected_by_reason: dict[str, int] = {}

    for predictions in predictions_by_doc_id.values():
        failsafe_mentions = [m for m in predictions if m.source == "failsafe"]
        if failsafe_mentions:
            docs_with_reinjection += 1
            reinjected_total += len(failsafe_mentions)
            for m in failsafe_mentions:
                cls = m.get_nyha_class() or "unknown"
                reinjected_by_class[cls] = reinjected_by_class.get(cls, 0) + 1
                # Evidence format: "failsafe re-injection (reason)"
                match = re.search(r"\(([^)]+)\)$", m.evidence)
                reason = match.group(1) if match else "unknown"
                reinjected_by_reason[reason] = reinjected_by_reason.get(reason, 0) + 1

        consistency = assess_current_consistency(predictions, policy)
        if consistency["note_status"] == "conflict":
            docs_flagged_conflict += 1

    audit = {
        "policy": policy,
        "documents": docs_total,
        "documents_with_reinjection": docs_with_reinjection,
        "documents_flagged_conflict": docs_flagged_conflict,
        "reinjected_total": reinjected_total,
        "reinjected_by_class": reinjected_by_class,
        "reinjected_by_reason": reinjected_by_reason,
    }
    audit_path = output_dir / f"failsafe_audit_{stage}.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"  Failsafe audit written to {audit_path}")


def run_dspy_pipeline(
    documents: List[NYHANote],
    output_dir: Path,
    config: Config,
    base_data_folder_path: str,
) -> NYHAExtractor:
    """
    End-to-end pipeline runner: split → Step 1 → Step 2 → Step 3a/3b → Step 4.

    configure_lm is called once; its max_demo_tokens_override (from the vllm
    node file) takes precedence over config.max_demo_tokens for demo building.
    """
    # ── Configure LMs (single call, Bug 2 fix) ───────────────────────────────
    identify_lm, classify_lm, prompt_lm, tokenizer, max_demo_tokens_override = (
        configure_lm(
            base_data_folder_path=base_data_folder_path,
            token_file_name=config.token_file,
            task_lm=config.task_lm,
            prompt_lm=config.prompt_lm,
            vllm_node_file=config.vllm_node_file,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            max_tokens=config.max_tokens,
            identify_max_tokens=config.identify_max_tokens,
            prompt_lm_max_tokens=config.prompt_lm_max_tokens,
            repetition_penalty=config.repetition_penalty,
        )
    )

    # max_demo_tokens_override from node file takes precedence over config default
    effective_max_demo_tokens = (
        max_demo_tokens_override
        if max_demo_tokens_override is not None
        else config.max_demo_tokens
    )

    # Initialize checkpoint system
    checkpoint = PipelineCheckpoint(output_dir)

    # Build output folder structure
    out = Path(output_dir)
    complete_examples_out_path = out / ZS_COMPLETE_FOLDER_NAME
    opt_subset_examples_out_path = out / OPT_SUBSET_FOLDER_NAME
    folder_paths = [
        out,
        out / SPLITS_FOLDER_NAME,
        complete_examples_out_path,
        opt_subset_examples_out_path,
    ]
    for folder in folder_paths:
        folder.mkdir(parents=True, exist_ok=True)

    # ── Dataset split (patient-grouped) ──────────────────────────────────────
    total = len(documents)
    document_by_id = {doc.old_note_id: doc for doc in documents}

    def _docs_from_ids(doc_ids: List[str]) -> List[NYHANote] | None:
        try:
            return [document_by_id[doc_id] for doc_id in doc_ids]
        except KeyError as missing_id:
            logging.warning(
                "Cached split references missing document ID %s; recomputing split.",
                missing_id.args[0],
            )
            return None

    cached_splits = checkpoint.load_document_id_splits()
    loaded_splits: tuple[List[NYHANote], List[NYHANote], List[NYHANote]] | None = None

    if cached_splits is not None:
        version_ok = (
            cached_splits.get("version", 1) >= 2
            and cached_splits.get("grouping") is not None
        )
        if not version_ok:
            logging.info(
                "Cached split predates patient grouping (version < 2); recomputing."
            )
        else:
            cached_train_docs = _docs_from_ids(cached_splits.get("train_ids", []))
            cached_val_docs = _docs_from_ids(cached_splits.get("val_ids", []))
            cached_test_docs = _docs_from_ids(cached_splits.get("test_ids", []))

            if (
                cached_train_docs is not None
                and cached_val_docs is not None
                and cached_test_docs is not None
            ):
                loaded_splits = (cached_train_docs, cached_val_docs, cached_test_docs)
                print(
                    f"  [CHECKPOINT] Loaded cached document splits from {checkpoint.split_file}"
                )

    if loaded_splits is None:
        note_to_group = load_note_patient_groups(
            base_data_folder_path, [doc.old_note_id for doc in documents]
        )
        train_docs, val_docs, test_docs = patient_grouped_split(
            documents, note_to_group, seed=config.seed
        )
        checkpoint.save_document_id_splits(
            train_ids=[doc.old_note_id for doc in train_docs],
            val_ids=[doc.old_note_id for doc in val_docs],
            test_ids=[doc.old_note_id for doc in test_docs],
            seed=config.seed,
            total_documents=total,
            grouping="pseudo_patient_id",
            group_counts={
                "train": len({note_to_group[d.old_note_id] for d in train_docs}),
                "val": len({note_to_group[d.old_note_id] for d in val_docs}),
                "test": len({note_to_group[d.old_note_id] for d in test_docs}),
            },
        )
    else:
        train_docs, val_docs, test_docs = loaded_splits

    logging.info(
        f"\n  Dataset split: train={len(train_docs)}  "
        f"val={len(val_docs)}  test={len(test_docs)}"
    )

    # ── Step 1 ────────────────────────────────────────────────────────────────
    if not checkpoint.should_skip_step1():
        baseline_extractor, baseline_preds = step1_zero_shot_baseline(
            documents=documents,
            output_dir=complete_examples_out_path,
            identify_lm=identify_lm,
            classify_lm=classify_lm,
            use_identification=config.use_identification,
            tokenizer=tokenizer,
            use_windowed_classification_context=config.use_windowed_classification_context,
            max_eval_tokens=config.max_eval_tokens,
        )
        checkpoint.save(
            1,
            config_snapshot=config.checkpoint_snapshot(),
            baseline_extractor_saved=True,
        )
        logging.info("\n[INFO] Baseline zero-shot extractor ended.")
    else:
        print("[CHECKPOINT] Step 1 already completed, skipping...")
        baseline_extractor, baseline_preds = step1_zero_shot_baseline(
            documents=documents,
            output_dir=complete_examples_out_path,
            identify_lm=identify_lm,
            classify_lm=classify_lm,
            use_identification=config.use_identification,
            tokenizer=tokenizer,
            use_windowed_classification_context=config.use_windowed_classification_context,
            max_eval_tokens=config.max_eval_tokens,
        )

    # Zero-shot comparison report on the held-out test set (document-level),
    # produced regardless of whether Step 1 ran or was resumed from cache, so it
    # is directly comparable to the bootstrap / MIPRO test reports below.
    step4_final_evaluation(
        baseline_extractor,
        test_docs,
        opt_subset_examples_out_path,
        tokenizer=tokenizer,
        label="base",
        n_bootstrap=config.n_bootstrap,
        bootstrap_seed=config.seed,
    )

    # ── Step 2 ────────────────────────────────────────────────────────────────
    if not checkpoint.should_skip_step2():
        _ = step2_error_analysis(
            documents,
            baseline_preds,
            complete_examples_out_path,
        )
        checkpoint.save(
            2,
            config_snapshot=config.checkpoint_snapshot(),
            error_analysis_complete=True,
        )
    else:
        print("[CHECKPOINT] Step 2 already completed, skipping...")

    # ── Step 3 — build examples and cap demo pool / valset ────────────────────
    best_extractor = baseline_extractor
    trainset = documents_to_dspy_examples(
        train_docs,
        config,
        tokenizer=tokenizer,
        text_token_budget=effective_max_demo_tokens,
    )
    # Valset is built document-level (explode=False) so the optimiser ranks
    # candidates — and reports are produced — on whole notes exactly as
    # inference runs (the extractor windows internally per span). Exploding the
    # valset per-mention would create in-window false positives from
    # neighbouring mentions and make scores incomparable to the test reports.
    valset = documents_to_dspy_examples(
        val_docs,
        config,
        tokenizer=tokenizer,
        text_token_budget=config.max_eval_tokens,
        explode=False,
    )

    trainset = subsample_examples(
        trainset,
        config.dspy_demo_pool_size,
        config.dspy_demo_min_per_class,
        config.seed,
    )
    valset = subsample_examples(valset, config.dspy_opt_val_size, 0, config.seed)

    evaluator = Evaluate(
        devset=valset,
        metric=nyha_dspy_metric,
        num_threads=config.num_threads,
    )

    baseline_score = evaluator(baseline_extractor)
    print(f"Baseline (no optimization): {baseline_score}")
    _prev_best_label = ((checkpoint.load() or {}).get("best") or "")
    best_label = "baseline"
    best_score = baseline_score
    _optimiser_ran = False

    # Sanity-check metric on first 10 trainset examples
    failures = []
    successes = []

    for i, ex in enumerate(trainset[:10]):
        try:
            pred = baseline_extractor(
                deidentified_text=ex.deidentified_text,
                authored_datetime=ex.authored_datetime,
            )
            score = nyha_dspy_metric(ex, pred)

            if score:
                successes.append((i, ex, pred, score))
                print(f"✓ Example {i}: PASS (score={score})")
            else:
                failures.append((i, ex, pred, score))
                print(f"✗ Example {i}: FAIL (score={score})")
                print(
                    f"  Expected: {ex.nyha_class if hasattr(ex, 'nyha_class') else ex}"
                )
                print(f"  Got: {pred}")
        except Exception as e:
            failures.append((i, ex, None, str(e)))
            print(f"✗ Example {i}: ERROR - {e}")

    print(f"\n{len(successes)}/{len(trainset[:10])} examples passed")

    if config.optimiser in ("bootstrap", "both"):
        if not checkpoint.should_skip_step3a():
            _optimiser_ran = True
            bs_extractor = step3a_bootstrap_fewshot(
                trainset,
                valset,
                complete_examples_out_path,
                identify_lm=identify_lm,
                classify_lm=classify_lm,
                max_bootstrapped_demos=config.bootstrap_max_bootstrapped_demos,
                max_labeled_demos=config.bootstrap_max_labeled_demos,
                num_candidate_programs=config.bootstrap_num_candidate_programs,
                num_threads=config.num_threads,
                use_identification=config.use_identification,
                use_windowed_classification_context=config.use_windowed_classification_context,
                tokenizer=tokenizer,
                max_eval_tokens=config.max_eval_tokens,
            )
            bs_score = evaluator(bs_extractor)
            if bs_score > best_score:
                best_extractor = bs_extractor
                best_score = bs_score
                best_label = "bootstrap"
            checkpoint.save(
                3,
                config_snapshot=config.checkpoint_snapshot(),
                bootstrap_complete=True,
            )
            step4_final_evaluation(
                bs_extractor,
                test_docs,
                opt_subset_examples_out_path,
                label="bs",
                tokenizer=tokenizer,
                n_bootstrap=config.n_bootstrap,
                bootstrap_seed=config.seed,
            )
        else:
            print("[CHECKPOINT] Step 3a (Bootstrap) already completed, skipping...")

    best_idx = None
    if config.optimiser in ("mipro", "both"):
        if not checkpoint.should_skip_step3b():
            _optimiser_ran = True
            mipro_extractor = step3b_mipro(
                trainset,
                valset,
                complete_examples_out_path,
                identify_lm=identify_lm,
                classify_lm=classify_lm,
                prompt_lm=prompt_lm,
                num_trials=config.mipro_num_trials,
                num_candidates=config.mipro_num_candidates,
                max_bootstrapped_demos=config.mipro_max_bootstrapped_demos,
                max_labeled_demos=config.mipro_max_labeled_demos,
                num_threads=config.num_threads,
                use_identification=config.use_identification,
                use_windowed_classification_context=config.use_windowed_classification_context,
                view_data_batch_size=config.mipro_view_data_batch_size,
                tokenizer=tokenizer,
                max_eval_tokens=config.max_eval_tokens,
            )
            step4_final_evaluation(
                mipro_extractor,
                test_docs,
                opt_subset_examples_out_path,
                label="mipro",
                tokenizer=tokenizer,
                n_bootstrap=config.n_bootstrap,
                bootstrap_seed=config.seed,
            )
            best_extractor, best_idx = select_best(
                [best_extractor, mipro_extractor],
                valset,
                label="MIPRO vs Bootstrap",
            )
            if best_idx == 1:
                best_label = "mipro"
            checkpoint.save(
                4,
                config_snapshot=config.checkpoint_snapshot(),
                mipro_complete=True,
            )
        else:
            print("[CHECKPOINT] Step 3b (MIPRO) already completed, skipping...")

    # ── Step 4 ────────────────────────────────────────────────────────────────
    final_results = step4_final_evaluation(
        best_extractor,
        test_docs,
        opt_subset_examples_out_path,
        label="optimised",
        tokenizer=tokenizer,
        n_bootstrap=config.n_bootstrap,
        bootstrap_seed=config.seed,
    )
    weak = final_results.get("weak_cells", [])
    if weak:
        print("\n  [RECOMMENDATION] Synthetic augmentation priority cells:")
        for cell in weak:
            print(f"    → {cell}")
    # When resuming a fully-completed run (no optimiser steps ran this session),
    # _prev_best_label preserves the winner recorded by the original run.
    if not _optimiser_ran and _prev_best_label in ("bootstrap", "mipro"):
        best_label = _prev_best_label
    checkpoint.save(
        5,
        best=best_label,
        config_snapshot=config.checkpoint_snapshot(),
        final_evaluation_complete=True,
    )

    print(
        "\n[CHECKPOINT] Pipeline complete. Checkpoint files saved in:",
        checkpoint.checkpoint_dir,
    )
    return best_extractor
