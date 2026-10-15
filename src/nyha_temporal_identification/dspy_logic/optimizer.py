"""
optimizer.py — DSPy-based iterative prompt optimisation (Steps 1–4).

Stages:
  Step 1 — Zero-shot baseline evaluation
  Step 2 — Error analysis (automatic categorisation + human-readable report)
  Step 3 — Targeted optimisation with BootstrapFewShot and/or MIPRO
  Step 4 — Freeze best prompt; export for downstream use
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Tuple

import dspy
from dspy import BootstrapFewShotWithRandomSearch
from tqdm import tqdm
from transformers import AutoTokenizer  # ty:ignore[unresolved-import]

from nyha_temporal_identification.data_handling.utils import remove_extra_text
from nyha_temporal_identification.data_model.dspy_model import NYHAMention, NYHANote
from nyha_temporal_identification.dspy_logic.modules import (
    NYHAExtractor,
    build_nyha_extractor,
    get_extractor_run_variant,
)
from nyha_temporal_identification.evaluation.entities import ErrorSummary
from nyha_temporal_identification.evaluation.evaluation import (
    analyse_errors,
    per_class_report,
    print_evaluation_report,
    regex_candidate_recall_by_temporality,
    spacy_candidate_recall_by_temporality,
    store_records,
)
from nyha_temporal_identification.evaluation.metrics_computation import (
    mention_f1,
    nyha_dspy_metric,
)


def configure_lm(
    base_data_folder_path: str,
    token_file_name: str,
    task_lm: str,
    prompt_lm: str = "",
    vllm_node_file: str = "",
    base_url: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.1,
    max_tokens: int = 1536,
    identify_max_tokens: int = 2048,
    prompt_lm_max_tokens: int = 5000,
    repetition_penalty: float = 0.0,
) -> tuple[dspy.LM, dspy.LM, dspy.LM, Any, int | None]:
    """
    Configure DSPy LMs pointing at a vLLM OpenAI-compatible endpoint.

    Returns (identify_lm, classify_lm, prompt_lm_instance, tokenizer,
    max_demo_tokens_override).

    Endpoint resolution (first match wins):
    1. VLLM_BASE_URL env var — used verbatim (set by run_nyha.sh to the SSH
       tunnel address, e.g. http://localhost:8000/v1).
    2. base_url arg — used verbatim.
    3. vllm_node_file arg or VLLM_NODE_FILE env var — host:port on line 1.
    4. Fallback: http://localhost:8000/v1.

    MAX_DEMO_TOKENS=<n> on line 2 of the node file is always applied regardless
    of which URL source wins.

    Three distinct LM instances are built over task_lm (or prompt_lm for the
    proposer), differing only in their output max_tokens so the high-volume
    classification path is not over-allocated.
    """
    # ── Endpoint resolution ───────────────────────────────────────────────────
    _env_node_file = os.environ.get("VLLM_NODE_FILE", "")
    if vllm_node_file and os.path.exists(vllm_node_file):
        node_file = vllm_node_file
    elif _env_node_file and os.path.exists(_env_node_file):
        node_file = _env_node_file
    else:
        candidate = vllm_node_file or _env_node_file
        if candidate:
            print(f"[WARNING] vllm_node_file '{candidate}' not found; falling back to base_url.")
        node_file = ""
    max_demo_tokens_override: int | None = None

    # VLLM_BASE_URL (exported by run_nyha.sh as the SSH-tunnel address) takes
    # priority so that cross-partition direct access is not attempted.
    _env_base_url = os.environ.get("VLLM_BASE_URL", "")
    api_url = _env_base_url or base_url or "http://localhost:8000/v1"

    if node_file:
        with open(node_file) as f:
            lines = [line.strip() for line in f.readlines()]
        # Only use the node file's host:port when no explicit URL was provided.
        if lines and not (_env_base_url or base_url):
            host_port = lines[0]
            api_url = f"http://{host_port}/v1"
        for line in lines[1:]:
            if line.startswith("MAX_DEMO_TOKENS="):
                try:
                    max_demo_tokens_override = int(line.split("=", 1)[1])
                except ValueError:
                    pass

    print(f"\n[INFO] Configuring DSPy to use LM at {api_url} with model {task_lm}...")

    # ── HF token ─────────────────────────────────────────────────────────────
    token_path = os.path.join(base_data_folder_path, token_file_name)
    hf_token: str | None = None
    if os.path.exists(token_path):
        with open(token_path) as f:
            hf_token = f.read().strip() or None

    # ── API key ───────────────────────────────────────────────────────────────
    if api_url.startswith("https://") and api_key_env in os.environ:
        used_api_key = os.environ[api_key_env]
    else:
        used_api_key = hf_token or "ignored"
        if hf_token:
            os.environ.setdefault("OPENAI_API_KEY", hf_token)

    # ── Model IDs ─────────────────────────────────────────────────────────────
    def _prefix(model_id: str) -> str:
        return model_id if model_id.startswith("openai/") else "openai/" + model_id

    task_model = _prefix(task_lm)
    prompt_model = _prefix(prompt_lm) if prompt_lm else task_model

    # ── Extra body params ─────────────────────────────────────────────────────
    extra_kwargs: dict = {}
    if repetition_penalty != 0.0:
        extra_kwargs["repetition_penalty"] = repetition_penalty

    # ── Build three LM instances ──────────────────────────────────────────────
    def _make_lm(model: str, output_tokens: int) -> dspy.LM:
        return dspy.LM(
            model=model,
            api_base=api_url,
            api_key=used_api_key,
            temperature=temperature,
            max_tokens=output_tokens,
            cache=True,
            **extra_kwargs,
        )

    identify_lm = _make_lm(task_model, identify_max_tokens)
    classify_lm = _make_lm(task_model, max_tokens)
    prompt_lm_instance = _make_lm(prompt_model, prompt_lm_max_tokens)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        task_lm,  # no "openai/" prefix for HuggingFace
        token=hf_token,
    )

    # classify_lm is the high-volume default; identify_lm is the richer fallback
    dspy.configure(lm=classify_lm)

    if max_demo_tokens_override is not None:
        print(f"  MAX_DEMO_TOKENS override from node file: {max_demo_tokens_override}")
    print("  DSPy LMs configured successfully.")
    return identify_lm, classify_lm, prompt_lm_instance, tokenizer, max_demo_tokens_override


def step1_zero_shot_baseline(
    documents: List[NYHANote],
    output_dir: Path,
    identify_lm: dspy.LM,
    classify_lm: dspy.LM,
    tokenizer: Any,
    use_identification: bool = False,
    use_windowed_classification_context: bool = False,
    max_eval_tokens: int = 0,
) -> Tuple[NYHAExtractor, List[List[NYHAMention]]]:
    """
    Step 1 — Zero-shot baseline evaluation.

    Runs an unoptimised NYHAExtractor over the full dataset to establish a
    performance baseline before any DSPy optimisation.
    """
    print("\n[STEP 1] Zero-shot baseline evaluation...")

    extractor: NYHAExtractor = build_nyha_extractor(
        identify_lm=identify_lm,
        classify_lm=classify_lm,
        use_identification=use_identification,
        use_windowed_classification_context=use_windowed_classification_context,
        tokenizer=tokenizer,
        max_eval_tokens=max_eval_tokens,
        failsafe_dropped_current=False,
    )

    variant = get_extractor_run_variant(
        use_identification,
        use_windowed_classification_context,
    )
    output_file_path = output_dir / f"baseline_predictions_{variant}.json"
    if output_file_path.exists():
        print(f"  Baseline predictions file already exists at {output_file_path}")
        with open(output_file_path, "r") as f:
            serialisable = json.load(f)

        saved_ids = set(serialisable.keys())
        current_ids = {doc.old_note_id for doc in documents}
        if saved_ids != current_ids:
            raise ValueError(
                f"Cached predictions don't match current documents. "
                f"Missing: {current_ids - saved_ids}, Extra: {saved_ids - current_ids}"
            )
        predictions: list[list[NYHAMention]] = [
            [NYHAMention(**m) for m in serialisable[doc.old_note_id]["predictions"]]
            for doc in documents
        ]
        print("  Baseline predictions loaded from file.")
    else:
        predictions: List[List[NYHAMention]] = []
        json_output = {}

        for doc in tqdm(documents, desc="Zero-shot baseline"):
            deidentified_text = remove_extra_text(doc.deidentified_text)
            preds = extractor(
                deidentified_text=deidentified_text,
                authored_datetime=doc.authored_datetime,
            )
            if not isinstance(preds, list) or (
                len(preds) > 0 and not isinstance(preds[0], NYHAMention)
            ):
                print(
                    f"[WARNING] Unexpected prediction format for "
                    f"old_note_id={doc.old_note_id}: {preds!r} — treating as []"
                )
                preds = []
            predictions.append(preds)
            json_output[doc.old_note_id] = {
                "deidentified_text": deidentified_text,
                "authored_datetime": doc.authored_datetime,
                "predictions": [p.model_dump() for p in preds],
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"  Baseline predictions saved to {output_file_path}")

    print_evaluation_report(
        "ZERO-SHOT BASELINE",
        documents,
        predictions,
        output_dir_path=output_dir,
    )

    return extractor, predictions


def step2_error_analysis(
    documents: List[NYHANote],
    predictions: List[List[NYHAMention]],
    output_dir: Path,
    top_n: int = 10,
) -> ErrorSummary:
    """
    Categorise errors, surface hard cases, export full report.
    Returns error summary dict for use in Step 3 prompt design.
    """
    print("\n[STEP 2] Error analysis...")

    records, summary = analyse_errors(
        documents,
        predictions,
    )
    store_records(
        records,
        output_dir_path=output_dir,
    )
    cell_report = per_class_report(documents, predictions)

    records_sorted = sorted(records, key=lambda r: len(r.error_types), reverse=True)

    print(f"\n  Top {top_n} hardest documents:")
    for r in records_sorted[:top_n]:
        print(f"\n  old_note_id={r.old_note_id}  errors={r.error_types}")
        print(f"  GOLD: {[m.model_dump() if m is not None else None for m in r.gold]}")
        print(f"  PRED: {[m.model_dump() if m is not None else None for m in r.pred]}")

    weak_cells = [
        cell
        for cell, metrics in cell_report.items()
        if metrics["f1"] < 0.5 and (metrics["tp"] + metrics["fn"]) > 0
    ]

    print(f"\n  Under-performing cells (F1 < 0.5): {weak_cells}")
    print("  These cells are priority targets for synthetic augmentation.")

    report_path = output_dir / "error_analysis.json"
    with open(report_path, "w") as f:
        json.dump(summary.to_json(), f, indent=2)
    print(f"\n  Full error report saved to {report_path}")

    return summary


def step3a_bootstrap_fewshot(
    trainset: List[dspy.Example],
    valset: List[dspy.Example],
    output_dir: Path,
    identify_lm: dspy.LM,
    classify_lm: dspy.LM,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4,
    num_candidate_programs: int = 10,
    num_threads: int = 4,
    use_identification: bool = True,
    use_windowed_classification_context: bool = False,
    tokenizer: Any = None,
    max_eval_tokens: int = 0,
) -> NYHAExtractor:
    """
    Step 3a — BootstrapFewShotWithRandomSearch.

    All demo/search/thread counts are config-driven (previously hard-coded).
    Fixed optimizer settings: max_rounds=2, max_errors=10, metric_threshold=0.85.
    """
    print("\n[STEP 3a] BootstrapFewShot optimisation...")

    variant = get_extractor_run_variant(
        use_identification,
        use_windowed_classification_context,
    )
    save_path = output_dir / f"optimised_bootstrap_{variant}.json"
    if save_path.exists():
        print(f"  Cached bootstrap extractor found at {save_path}; loading it.")
        return load_optimised_extractor(
            str(save_path),
            identify_lm=identify_lm,
            classify_lm=classify_lm,
            use_identification=use_identification,
            use_windowed_classification_context=use_windowed_classification_context,
            tokenizer=tokenizer,
            max_eval_tokens=max_eval_tokens,
        )

    extractor: NYHAExtractor = build_nyha_extractor(
        identify_lm=identify_lm,
        classify_lm=classify_lm,
        use_identification=use_identification,
        use_windowed_classification_context=use_windowed_classification_context,
        tokenizer=tokenizer,
        max_eval_tokens=max_eval_tokens,
        failsafe_dropped_current=False,
    )

    optimizer = BootstrapFewShotWithRandomSearch(
        metric=nyha_dspy_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=2,
        max_errors=10,
        num_candidate_programs=num_candidate_programs,
        num_threads=num_threads,
        metric_threshold=0.85,
    )

    optimised = optimizer.compile(
        extractor,
        trainset=trainset,
        valset=valset,
    )

    # The comparable performance report is produced on the held-out test set by
    # step4_final_evaluation (stage "final_bs"); no per-mention valset report is
    # emitted here, as it used a different set and granularity than the other
    # stages and was not comparable.
    optimised.save(str(save_path))
    print(f"  Optimised program saved to {save_path}")

    return optimised


def step4_final_evaluation(
    optimised_extractor: NYHAExtractor,
    test_documents: List[NYHANote],
    output_dir: Path,
    tokenizer: Any,
    label: str = "OPTIMISED",
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> dict:
    """
    Step 4 — Final evaluation on the held-out test set.

    Runs the optimised extractor on documents it has never seen during training
    or optimisation, writes predictions and cell-level report.
    """
    print("\n[STEP 4] Final evaluation on held-out test set...")

    report_path = output_dir / f"final_{label}_report.json"
    predictions_path = output_dir / f"final_{label}_predictions.json"
    if report_path.exists() and predictions_path.exists():
        print(f"  Cached final evaluation found at {report_path}; loading it.")
        with open(report_path, "r") as f:
            cached_results = json.load(f)
        return cached_results

    raw_predictions = [
        optimised_extractor(
            deidentified_text=doc.deidentified_text,
            authored_datetime=doc.authored_datetime,
        )
        for doc in test_documents
    ]

    cleaned_predictions: List[List[NYHAMention]] = []
    for pred in raw_predictions:
        if not isinstance(pred, list) or (
            len(pred) > 0 and not isinstance(pred[0], NYHAMention)
        ):
            print(f"[WARNING] Unexpected prediction format: {pred!r} — treating as []")
            pred = []
        cleaned_predictions.append(pred)

    print_evaluation_report(
        f"final_{label}",
        test_documents,
        cleaned_predictions,
        output_dir_path=output_dir,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )

    cell_report = per_class_report(test_documents, cleaned_predictions)
    records, error_summary = analyse_errors(
        test_documents,
        cleaned_predictions,
    )
    store_records(
        records,
        output_dir_path=output_dir,
    )

    results = {
        "cell_report": cell_report,
        "error_summary": error_summary.to_json(),
    }

    _save_predictions(cleaned_predictions, predictions_path)

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # Write spaCy current-recall diagnostic once (stage-independent).
    spacy_recall_path = output_dir / "spacy_recall.json"
    if not spacy_recall_path.exists():
        spacy_recall = spacy_candidate_recall_by_temporality(test_documents)
        with open(spacy_recall_path, "w") as f:
            json.dump(spacy_recall, f, indent=2)
        print(f"  spaCy recall diagnostic saved to {spacy_recall_path}")

    # Write regex pass-2 recall diagnostic once (stage-independent).
    regex_recall_path = output_dir / "regex_recall.json"
    if not regex_recall_path.exists():
        regex_recall = regex_candidate_recall_by_temporality(test_documents)
        with open(regex_recall_path, "w") as f:
            json.dump(regex_recall, f, indent=2)
        print(f"  Regex recall diagnostic saved to {regex_recall_path}")

    print(f"\n  Results saved to {output_dir}")
    return results


def select_best(
    extractors: List[NYHAExtractor],
    valset: List[dspy.Example],
    label: str = "",
) -> Tuple[NYHAExtractor, int]:
    """Evaluate each extractor on valset; return the one with highest micro-averaged F1."""
    best_idx = -1
    best_f1 = -1.0
    best_extractor = extractors[0]
    val_docs = _examples_to_documents(valset)
    for ext in extractors:
        preds = [
            ext(
                deidentified_text=ex.deidentified_text,
                authored_datetime=ex.authored_datetime,
            )
            for ex in valset
        ]
        total_tp, total_fp, total_fn = 0.0, 0, 0
        for doc, doc_preds in zip(val_docs, preds):
            r = mention_f1(
                doc.mentions,
                doc_preds if isinstance(doc_preds, list) else [],
            )
            total_tp += r["tp"]
            total_fp += r["fp"]
            total_fn += r["fn"]
        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        print(f"  [{label}] extractor micro-F1 on val = {micro_f1:.4f}")
        if micro_f1 > best_f1:
            best_f1 = micro_f1
            best_extractor = ext
            best_idx = extractors.index(ext)
    return best_extractor, best_idx


def _save_predictions(
    predictions: List[List[NYHAMention]],
    path: Path,
) -> None:
    serialisable = [[m.model_dump() for m in preds] for preds in predictions]
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


def _examples_to_documents(
    examples: List[dspy.Example],
) -> List[NYHANote]:
    return [
        NYHANote(
            old_note_id=str(i),
            deidentified_text=ex.deidentified_text,
            mentions=ex.mentions,
            authored_datetime=getattr(ex, "authored_datetime", None),
        )
        for i, ex in enumerate(examples)
    ]


def load_optimised_extractor(
    path: str,
    identify_lm: dspy.LM,
    classify_lm: dspy.LM,
    use_identification: bool = True,
    use_windowed_classification_context: bool = False,
    tokenizer: Any = None,
    max_eval_tokens: int = 0,
    failsafe_dropped_current: bool = False,
) -> NYHAExtractor:
    """Reload a previously saved optimised extractor."""
    extractor = build_nyha_extractor(
        identify_lm=identify_lm,
        classify_lm=classify_lm,
        use_identification=use_identification,
        use_windowed_classification_context=use_windowed_classification_context,
        tokenizer=tokenizer,
        max_eval_tokens=max_eval_tokens,
        failsafe_dropped_current=failsafe_dropped_current,
    )
    extractor.load(path)
    # DSPy restores the saved LM (with its old api_base) onto each Predict during
    # load(); re-apply the current session's LMs so inference goes to the right endpoint.
    if classify_lm:
        extractor.classify.set_lm(classify_lm)
    if identify_lm and extractor.identify is not None:
        extractor.identify.set_lm(identify_lm)
    return extractor
