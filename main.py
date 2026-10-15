import os
from pathlib import Path

from nyha_temporal_identification.constants import (
    PSEUDONYMISATION_FOLDER_NAME,
    TEMPORAL_STATUS_PREDICTION_FOLDER_NAME,
    ZS_COMPLETE_FOLDER_NAME,
)
from nyha_temporal_identification.data_handling.data_loading import (
    load_dataset,
    load_whitelisted_note_ids,
)
from nyha_temporal_identification.dspy_logic.checkpoint import PipelineCheckpoint
from nyha_temporal_identification.dspy_logic.modules import (
    NYHAExtractor,
    get_extractor_run_variant,
)
from nyha_temporal_identification.dspy_logic.optimizer import (
    configure_lm,
    load_optimised_extractor,
)
from nyha_temporal_identification.dspy_logic.runners import (
    run_dspy_pipeline,
    run_inference,
)
from nyha_temporal_identification.extraction.nyha_extraction import (
    extract_nyha_labels,
)
from nyha_temporal_identification.models.pipeline import train_baselines
from nyha_temporal_identification.utils.general import (
    ANNOTATE_TRAINING_NYHA_MENTIONS_COMMAND_NAME,
    EXTRACT_NYHA_COMMAND_NAME,
    RUN_DSPY_PIPELINE_COMMAND_NAME,
    TRAIN_BASELINES_COMMAND_NAME,
    Config,
    parse_args,
)


def _resolve_saved_extractor_path(
    output_dir: Path,
    config: Config,
) -> Path:
    variant = get_extractor_run_variant(
        config.use_identification,
        config.use_windowed_classification_context,
    )
    subpath = output_dir / ZS_COMPLETE_FOLDER_NAME
    state = PipelineCheckpoint(output_dir).load()
    best_label = (state or {}).get("best", "")
    if best_label in ("bootstrap", "mipro"):
        return subpath / f"optimised_{best_label}_{variant}.json"
    # Fallback: no checkpoint or baseline won — use config.optimiser heuristic
    if config.optimiser == "bootstrap":
        return subpath / f"optimised_bootstrap_{variant}.json"
    return subpath / f"optimised_mipro_{variant}.json"


def main() -> None:
    args, config = parse_args()

    for command in config.commands:
        print(f"\n[INFO] Running command: {command}")
        if command == EXTRACT_NYHA_COMMAND_NAME:
            _pseudo_data_dir = config.pseudonymisation_data_dir
            if config.pseudonymise and _pseudo_data_dir is None:
                _pseudo_data_dir = os.path.join(args.data_folder, PSEUDONYMISATION_FOLDER_NAME)
            # Extract all the NYHA and date mentions from the note dataset
            extract_nyha_labels(
                data_folder_path=args.data_folder,
                reset_previous_data=args.reset_previous_data,
                pseudonymise=config.pseudonymise,
                seed=config.seed,
                pseudonymisation_data_dir=_pseudo_data_dir,
            )
        elif (
            command == RUN_DSPY_PIPELINE_COMMAND_NAME
            or command == ANNOTATE_TRAINING_NYHA_MENTIONS_COMMAND_NAME
        ):
            documents = load_dataset(
                data_folder_path=args.data_folder,
                whitelisted_note_ids=load_whitelisted_note_ids(args.data_folder)
                if command == ANNOTATE_TRAINING_NYHA_MENTIONS_COMMAND_NAME
                else None,
                is_inference=command == ANNOTATE_TRAINING_NYHA_MENTIONS_COMMAND_NAME,
            )
            output_dir = Path(args.data_folder) / TEMPORAL_STATUS_PREDICTION_FOLDER_NAME

            if command == RUN_DSPY_PIPELINE_COMMAND_NAME:
                run_dspy_pipeline(
                    documents=documents,
                    output_dir=output_dir,
                    config=config,
                    base_data_folder_path=args.data_folder,
                )
            else:
                checkpoint = PipelineCheckpoint(output_dir)
                if not checkpoint.exists():
                    raise ValueError(
                        f"No checkpoint found at {checkpoint.checkpoint_file}; "
                        "run the pipeline before loading a saved extractor."
                    )
                checkpoint.assert_config_matches(config.checkpoint_snapshot())

                identify_lm, classify_lm, prompt_lm, tokenizer, _ = configure_lm(
                    base_data_folder_path=args.data_folder,
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
                extractor_path = _resolve_saved_extractor_path(
                    output_dir,
                    config,
                )
                extractor: NYHAExtractor = load_optimised_extractor(
                    str(extractor_path),
                    identify_lm=identify_lm,
                    classify_lm=classify_lm,
                    use_identification=config.use_identification,
                    use_windowed_classification_context=config.use_windowed_classification_context,
                    tokenizer=tokenizer,
                    max_eval_tokens=config.max_eval_tokens,
                    failsafe_dropped_current=config.failsafe_dropped_current,
                )
                run_inference(
                    extractor,
                    documents,
                    output_dir,
                    max_workers=config.num_threads,
                    failsafe_note_drop_policy=config.failsafe_note_drop_policy,
                )
        elif command == TRAIN_BASELINES_COMMAND_NAME:
            for fm in config.baseline_feature_modes:
                docs = load_dataset(data_folder_path=args.data_folder)
                train_baselines(
                    base_folder_path=args.data_folder,
                    documents=docs,
                    feature_mode=fm,
                    window_size=300,
                    apply_stemming=True,
                    max_bow_features=50000,
                    n_bootstrap=config.n_bootstrap,
                )
                train_baselines(
                    base_folder_path=args.data_folder,
                    documents=docs,
                    feature_mode=fm,
                    window_size=300,
                    apply_stemming=True,
                    max_bow_features=50000,
                    use_kfold_cv=False,
                    n_bootstrap=config.n_bootstrap,
                )
        else:
            print(f"[WARNING] Unknown command in configuration: {command}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    main()
