# NYHA Temporal Identification

Extract **NYHA functional class** mentions (I–IV, including ranges such as
`II-III`) from **Dutch de-identified clinical notes**, and classify each mention's
**temporality**: is the class the patient's **current** status, or a
**historical** one (e.g. *"bij opname NYHA III, nu NYHA II"*)?

The repository is a research pipeline that compares three families of approaches
on the same patient-grouped hold-out test set:

| Approach | What it is |
|---|---|
| **Rule-based** | spaCy + regex detection of NYHA spans and dates, with date linkage |
| **LLM** | A [DSPy](https://dspy.ai) extractor (Gemma 3 served by [vLLM](https://docs.vllm.ai)), evaluated zero-shot and after BootstrapFewShot / MIPROv2 prompt optimisation |
| **Classical ML baselines** | Threshold, logistic regression and gradient boosting temporality classifiers on distance and/or bag-of-words features |

---

## Contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Data folder setup](#data-folder-setup)
5. [Configuration](#configuration)
6. [Running the pipeline](#running-the-pipeline)
7. [Evaluating and comparing results](#evaluating-and-comparing-results)
8. [Running on Snellius (SLURM)](#running-on-snellius-slurm)
9. [Tests](#tests)
10. [Project layout](#project-layout)
11. [Data privacy](#data-privacy)
12. [Citation](#citation)
13. [License](#license)

---

## How it works

The LLM extractor (`NYHAExtractor`) is a layered pipeline:

```
Layer 0  spaCy + regex      high-recall NYHA span detection, date extraction and
                            linkage of each mention to its nearest date
Layer 1  LLM identify       (optional, use_identification = true) the LLM reviews
                            the spaCy seeds and adds or removes spans
Layer 2  LLM classify       per-span NYHA class + temporality (current/historical),
                            prompted with the linked-date context
Layer 3  post-processing    the LLM outputs (span_text, occurrence) pairs, which are
                            resolved deterministically to character offsets
```

LLMs cannot count characters reliably, so they never output offsets. Spans are
identified by their text plus an occurrence index and resolved in code.

Everything runs through a single entry point, `main.py`. A TOML config selects
one or more **commands**:

| Command | Needs GPU / vLLM | Purpose |
|---|---|---|
| `extract_nyha` | no | Rule-based extraction over all notes (optionally pseudonymised first) |
| `run_dspy_pipeline` | yes | Zero-shot → error analysis → prompt optimisation → final test evaluation |
| `train_baselines` | no | Train and evaluate the classical ML temporality baselines |
| `annotate_training_nyha_mentions` | yes | Run the best saved extractor over unseen notes (inference) |

---

## Requirements

- **Linux x86-64** (the only platform pinned in `pyproject.toml`).
- **[pixi](https://pixi.sh)** to manage the environments. pixi installs Python 3.12
  and all dependencies; nothing needs to be installed system-wide.
- For the LLM commands: an **NVIDIA GPU with CUDA 12** to host the vLLM server
  (any OpenAI-compatible endpoint also works, see [Start an LLM server](#3-start-an-llm-server)).
  Rough sizing: Gemma 3 12B in bf16 fits on one 40 GB+ GPU, and the 4-bit
  quantised Gemma 3 27B (`RedHatAI/gemma-3-27b-it-quantized.w4a16`) fits on one
  A100/H100.
- A **HuggingFace token** with access to the gated Gemma models, used both by vLLM
  and by the pipeline's tokenizer for token budgeting.
- **Internet access once**, to download the pseudonymisation lookup data.

---

## Installation

```bash
git clone <this-repo-url> nyha_temporal_identification
cd nyha_temporal_identification

# CPU environment: extraction, baselines, tests, and the pipeline client
pixi install

# GPU environment: additionally installs vLLM + CUDA toolkit (to host the model)
pixi install -e gpu

# HuggingFace token (gitignored)
echo "hf_xxxxxxxxxxxxxxxxx" > .hf_token

# Sanity check
pixi run test
```

The pipeline is a client of the vLLM HTTP API, so `run_dspy_pipeline` can run from
the CPU environment as long as a vLLM server is reachable. The `gpu` environment
is only needed on the machine that **hosts** the model.

---

## Data folder setup

All commands take a single data folder (`-s / --data-folder`). Inputs are read
from it, and every output is written back under it. Create the following layout:

```
<data>/
├── old_tables/
│   ├── DT4H_UC3_NotitiePatientContact.parquet   # note → patient-contact mapping
│   └── FreeTextData/
│       ├── DT4H_UC3_Text_P1_tableoutput.parquet # clinical notes (any number of
│       ├── DT4H_UC3_Text_P2_tableoutput.parquet #   files, name must start "DT4H")
│       └── ...
├── pseudonymisation/                            # lookup CSVs (see below)
└── exported_annotations/
    └── results/
        └── <annotator>/                         # Label Studio JSON exports,
            └── <note_id>.json                   #   one file per annotated note
```

### Clinical notes: `old_tables/FreeTextData/DT4H*.parquet`

Each parquet file must contain these columns:

| Column | Description |
|---|---|
| `old_id` | `<note_id>-<part>`. Long notes are split into parts, which are concatenated in `part` order |
| `type` | Note type |
| `authored` | Note authoring datetime (used for date-lag temporality signals) |
| `deidentified_text` | Note text, de-identified with [deduce](https://github.com/vmenger/deduce) (placeholders such as `[PERSON-1]`, `[DATE-3]`) |

### Note → patient-contact mapping: `old_tables/DT4H_UC3_NotitiePatientContact.parquet`

Columns `PatientNotitieId` (= the `<note_id>` part of `old_id`) and
`PatientContactId`. Only notes that appear in this mapping are kept. The mapping
is also used to group notes for the train/val/test split: notes from the same
patient contact always land in the same split. When the internal upstream tables
that link contacts to patients are available, the split groups by patient
instead, so splits never share a patient. With only the public inputs, one
patient with several contacts can still appear in more than one split. A warning
is logged when this fallback is used.

### Pseudonymisation lookup data: `pseudonymisation/`

When `pseudonymise = true` (recommended), every deduce placeholder (`[PERSON-1]`,
`[HOSPITAL-2]`, `[DATE-3]`, …) is replaced with a realistic random value before
extraction. This keeps NYHA patterns and the LLM away from bracketed tokens.
Replacements are reproducible per note (seeded by `seed` + note id). Non-PHI
labels that deduce wraps in brackets, such as `[NYHA]`, are passed through
unchanged.

The replacement values come from four CSVs that you build **once** from public
sources:

```bash
pixi run python scripts/build_pseudonymisation_data.py --data-dir <data>
```

This writes the following files to `<data>/pseudonymisation/`:

| File | Source |
|---|---|
| `dutch_first_names.csv` | Wikidata (given names, Dutch labels) |
| `dutch_surnames.csv` | [digitalheir/family-names-in-the-netherlands](https://github.com/digitalheir/family-names-in-the-netherlands) |
| `dutch_places.csv` | GeoNames NL (CC-BY 4.0) |
| `dutch_hospitals.csv` | Wikidata (hospitals in the Netherlands) |

Run it on a machine with internet access (on a cluster, the login node). The
pipeline finds the folder automatically at `<data>/pseudonymisation/`. Set
`pseudonymisation_data_dir` in the config only if the CSVs live elsewhere.

### Gold annotations: `exported_annotations/results/<annotator>/`

The LLM pipeline and the baselines are trained and evaluated on manually annotated
notes. Annotation is done in [Label Studio](https://labelstud.io): each NYHA span
is labelled and linked to a `current` / `historical` temporality node (and
optionally to its date). Place the per-note JSON exports in one sub-folder per
annotator. The folder names the code reads are listed in
`ANNOTATION_RESULTS_FOLDER_NAMES` in `src/nyha_temporal_identification/constants.py`;
a `merged/` folder is ignored. When the pipeline loads the annotations, it
normalises them into `exported_annotations/formatted/`.

### What the pipeline generates

You do not need to create these folders:

```
<data>/
├── old_tables/FreeTextData/
│   ├── DT4H_*_processed.json            # extract_nyha: NYHA-positive notes + mentions
│   ├── DT4H_*_processed_meta.parquet    # extract_nyha: per-file checkpoints
│   ├── all_notes_metadata.parquet       # extract_nyha: all notes, has_nyha flags
│   └── all_notes_texts.parquet
├── exported_annotations/formatted/      # normalised gold annotations
└── temporal_status_prediction/          # LLM pipeline + baselines (see below)
```

---

## Configuration

Runs are configured with a TOML file. Start from one of the shipped examples:

- `configs/config_cpu.toml`: `train_baselines` (or `extract_nyha`) on CPU
- `configs/config_gpu.toml`: LLM commands against a vLLM server

```toml
commands = ["run_dspy_pipeline"]           # run in order
seed = 42                                  # pseudonymisation + splits + sampling
use_identification = false                 # true = add the LLM identification layer
use_windowed_classification_context = true # classify on a window around each span
pseudonymise = true
hf_token_file = "./.hf_token"
n_bootstrap = 1000                         # bootstrap CIs on test reports (0 = off)

[dspy_pipeline]                            # always required (task_lm + optimiser)
task_lm = "google/gemma-3-12b-it"          # must match the model vLLM serves
optimiser = "both"                         # "bootstrap" | "mipro" | "both"
base_url = "http://localhost:8000/v1"      # or vllm_node_file = "logs/vllm_node_<JOB>.txt"
num_threads = 16                           # concurrent requests; <= vLLM --max-num-seqs
```

### Top-level keys

| Key | Required | Default | Description |
|---|---|---|---|
| `commands` | yes | — | Commands to run, in order |
| `seed` | yes | — | Seed for pseudonymisation, data splits and DSPy sampling |
| `use_identification` | yes | — | Add the LLM span-identification layer (otherwise spaCy spans go straight to classification) |
| `use_windowed_classification_context` | yes | — | Feed only a character window around each span to the classifier |
| `pseudonymise` | no | `false` | Replace deduce placeholders before extraction |
| `pseudonymisation_data_dir` | no | `<data>/pseudonymisation` | Override the lookup-CSV location |
| `hf_token_file` | no | `.hf_token` | HuggingFace token file |
| `n_bootstrap` | no | `0` | Document-level bootstrap resamples for confidence intervals on test reports |
| `baseline_feature_modes` | no | `["base_text", "text"]` | Baseline feature sets: `base` (distance features), `base_text` (distance + bag-of-words), `text` (bag-of-words only) |
| `failsafe_dropped_current` | no | `false` | At inference, re-inject spaCy candidates the LLM dropped as `current` mentions, so a downstream masker never leaks a current NYHA value |
| `failsafe_note_drop_policy` | no | `"balanced"` | Note-level conflict policy with the failsafe on: `balanced`, `conservative` or `off` |

### `[dspy_pipeline]` keys

| Key | Default | Description |
|---|---|---|
| `task_lm` *(required)* | — | HuggingFace model id used for identification and classification |
| `optimiser` *(required)* | — | `bootstrap`, `mipro` or `both` |
| `prompt_lm` | `task_lm` | Model that proposes instructions for MIPROv2 |
| `base_url` | — | OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1` |
| `vllm_node_file` | — | Node file written by `serve.sh`. Overrides `base_url` and sets `max_demo_tokens` (also read from the `VLLM_NODE_FILE` env var) |
| `api_key_env` | `OPENAI_API_KEY` | Env var holding the API key (ignored for local vLLM) |
| `num_threads` | `4` | Concurrent LLM calls. Keep ≤ the server's `--max-num-seqs` |
| `max_tokens` / `identify_max_tokens` / `prompt_lm_max_tokens` | `1536` / `2048` / `5000` | Output token budgets |
| `max_demo_tokens` / `max_eval_tokens` | `3000` / `3000` | Input caps per few-shot demo / per classification context (`0` = no cap) |
| `repetition_penalty` | `0.0` (off) | vLLM repetition penalty |
| `dspy_demo_pool_size` / `dspy_opt_val_size` | `50` / `300` | Caps on the demo pool and on the optimisation validation set |
| `bootstrap_*`, `mipro_*` | see `configs/config_gpu.toml` | Optimiser knobs (demos per prompt, candidates, trials) |

The input token caps must fit the server's context window:
few-shot demos + the note context + the output budget ≤ `--max-model-len`.
`serve.sh` computes a safe `MAX_DEMO_TOKENS` for you (see the
[Snellius section](#running-on-snellius-slurm)).

---

## Running the pipeline

The general form is:

```bash
pixi run python main.py -s <data> --config <config.toml> [--reset-previous-data]
```

Typical end-to-end order:

### 1. Rule-based extraction (`extract_nyha`, CPU)

```toml
commands = ["extract_nyha"]
pseudonymise = true
```

```bash
pixi run python main.py -s <data> --config configs/my_extract.toml
```

This pseudonymises every note (if enabled), runs spaCy/regex NYHA and date
detection, links each mention to its nearest date, and writes
`DT4H_*_processed.json` plus `all_notes_metadata.parquet` /
`all_notes_texts.parquet` to `old_tables/FreeTextData/`. Progress is checkpointed
per source file. Add `--reset-previous-data` to move the previous outputs aside
and start over.

### 2. Annotate a sample

Take a sample of the NYHA-positive notes from step 1, annotate them in Label Studio
(spans, class, current/historical, linked date), and export them to
`exported_annotations/results/<annotator>/` as described in
[Gold annotations](#gold-annotations-exported_annotationsresultsannotator).

### 3. Start an LLM server

Any OpenAI-compatible endpoint works. With vLLM, from the `gpu` environment:

```bash
export HF_TOKEN=$(cat .hf_token)
pixi run -e gpu vllm serve google/gemma-3-12b-it \
    --dtype bfloat16 --max-model-len 32768 --max-num-seqs 16 \
    --enable-prefix-caching --port 8000
```

Then set `base_url = "http://<host>:8000/v1"` in `[dspy_pipeline]`, and set
`task_lm` to the same model id. On Snellius, use `serve.sh` instead (see
[below](#running-on-snellius-slurm)).

### 4. LLM pipeline (`run_dspy_pipeline`)

```toml
commands = ["run_dspy_pipeline"]
```

The annotated notes are split **by patient** into ≈14 % train (few-shot demo
pool) / 56 % validation (optimiser selection) / 30 % **held-out test**. The split
is cached in `temporal_status_prediction/splits/id_splits.json`, so every later
run and the baselines reuse it. The pipeline then runs:

| Step | What happens |
|---|---|
| 1 | Zero-shot baseline over the full annotated set, plus a test-set report |
| 2 | Error analysis: failure categories and weak (class × temporality) cells |
| 3a | BootstrapFewShot (random search over bootstrapped chain-of-thought demos) |
| 3b | MIPROv2 instruction optimisation (when `optimiser = "mipro"` or `"both"`) |
| 4 | Final evaluation of every stage on the held-out test set; the best extractor on the validation set is recorded |

Every completed step is **checkpointed** (`temporal_status_prediction/checkpoints/`),
so an interrupted run resumes where it stopped. Changing `task_lm` against an
existing checkpoint is refused. Other config changes only produce a warning.

### 5. Classical baselines (`train_baselines`, CPU)

```toml
commands = ["train_baselines"]
baseline_feature_modes = ["base_text", "text"]
```

For each feature mode, this trains a threshold classifier (skipped in `text` mode,
which has no numeric features), logistic regression and gradient boosting, under
two protocols:

- **`cv/`**: patient-grouped k-fold cross-validation.
- **`single/`**: one train/test split that reuses the DSPy `id_splits.json`, so
  the test set is identical to the LLM test set. Run step 4 first if you want this
  comparison.

At test time, the baselines classify **spaCy-predicted** spans. Missed gold spans
count as false negatives, which makes the evaluation end-to-end and comparable
with the LLM.

### 6. Inference on unseen notes (`annotate_training_nyha_mentions`)

```toml
commands = ["annotate_training_nyha_mentions"]
failsafe_dropped_current = true   # recommended if the output feeds a NYHA masker
```

This loads the best optimised extractor from the step 4 checkpoint (it requires a
completed `run_dspy_pipeline` with the same `task_lm`), runs it over the extracted
notes, and writes one JSON per note to
`temporal_status_prediction/inference_results_unseen/`, plus a
`failsafe_audit_unseen.json` summary.

---

## Evaluating and comparing results

All LLM and baseline outputs live under `<data>/temporal_status_prediction/`:

```
temporal_status_prediction/
├── splits/id_splits.json                    # patient-grouped train/val/test ids
├── checkpoints/pipeline_state.json          # step checkpoint + best optimiser
├── zs_complete/                             # full-dataset artefacts
│   ├── evaluation_report_zero_shot_baseline.json
│   ├── error_analysis.json                  # step 2
│   ├── optimised_bootstrap_<variant>.json   # saved DSPy programs
│   └── optimised_mipro_<variant>.json
├── opt_subset/                              # HELD-OUT TEST SET
│   ├── evaluation_report_final_base.json    # zero-shot
│   ├── evaluation_report_final_bs.json      # BootstrapFewShot
│   ├── evaluation_report_final_mipro.json   # MIPROv2
│   ├── evaluation_report_final_optimised.json # selected best
│   ├── final_<stage>_predictions.json
│   └── spacy_recall.json / regex_recall.json
└── baselines/<cv|single>/<feature_mode>/<model>/evaluation_report_*.json
```

**Compare like with like:** the `opt_subset/evaluation_report_final_*.json`
files and the `baselines/single/...` reports are computed on the same held-out
test notes, with the same document-level metrics. These are the numbers to
report.

Every evaluation report contains:

| Metric | Meaning |
|---|---|
| `span_identification` | Span-detection F1 (label-agnostic) |
| `class` | NYHA class macro-F1 |
| `temporality` | current/historical macro-F1 and MCC |
| `composite` | 0.10 × span F1 + 0.10 × class macro-F1 + 0.80 × temporality macro-F1 |
| `confidence_intervals` | Mean, 95 % CI and std from document-level bootstrap (when `n_bootstrap > 0`) |

---

## Running on Snellius (SLURM)

Two batch scripts wrap the steps above for the
[Snellius](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius) cluster. Adapt
the `#SBATCH` headers (partition, memory, time) to your allocation.

### Start the vLLM server: `serve.sh`

```bash
mkdir -p logs

# Default: RedHatAI/gemma-3-27b-it-quantized.w4a16, ctx 25200, 24 seqs, 1 GPU
sbatch serve.sh

# Other model / two GPUs / longer context
sbatch serve.sh -m google/gemma-3-27b-it -tp 2 -l 32768

# Tune concurrency and memory utilisation
sbatch serve.sh -m google/gemma-3-12b-it -l 32768 -s 16 -g 0.85
```

| Flag | Description |
|---|---|
| `-m` | HuggingFace model id (must equal `task_lm`) |
| `-tp` | Tensor parallel size (number of GPUs) |
| `-l` | `--max-model-len` (context window, tokens) |
| `-s` | `--max-num-seqs` (concurrent sequences, ≥ `num_threads`) |
| `-g` | GPU memory utilisation (0–1) |
| `-P` | Port (default `8000`) |
| `-t` | HF token file (default `./.hf_token`) |
| `-H` | HuggingFace cache dir (default: `HF_HOME` env var, otherwise `~/.cache/huggingface`) |
| `-B` | Override the computed `MAX_DEMO_TOKENS` (`0` = don't write it) |

Once the server is up, it writes `logs/vllm_node_<JOB_ID>.txt`:

```
<node>:<port>
MAX_DEMO_TOKENS=<n>
```

Line 2 is the per-demo input budget that fits the served context length. The
pipeline uses it automatically when it is pointed at the node file.

Partitions used by the script (switch the commented `#SBATCH` block at the top of
`serve.sh`):

| Partition | GPU | Memory | CPUs |
|---|---|---|---|
| `gpu_a100` (active default) | A100 | 60 GB | 12 |
| `gpu_h100` | H100 | 120 GB | 12 |
| `gpu_mig` | A100 MIG slice | 40 GB | 4 |

### Run the pipeline: `run_nyha.sh`

```bash
# LLM commands against a running server job
sbatch run_nyha.sh \
    -d /path/to/data \
    -f configs/config_gpu.toml \
    -p /path/to/nyha_temporal_identification \
    -s <SERVER_JOB_ID> \
    -k                                         # scancel the server on success

# CPU-only commands (extract_nyha, train_baselines): no server needed
sbatch run_nyha.sh -d /path/to/data -f configs/config_cpu.toml -p "$PWD"
```

`run_nyha.sh` waits for the server's node file and `/health` endpoint, exports
`VLLM_BASE_URL` and `VLLM_NODE_FILE` (so the config's `vllm_node_file` can stay a
placeholder), installs the pixi environment if needed, and runs `main.py`.

Other options:

| Flag | Description |
|---|---|
| `-e default\|gpu` | pixi environment to run in |
| `-r` | Pass `--reset-previous-data` to `main.py` |
| `-b <url>` | Use an explicit `VLLM_BASE_URL` |
| `-O KEY=VALUE` | Override a config value without editing the file (repeatable). Examples: `-O seed=1`, `-O dspy_pipeline.num_threads=32`. |

---

## Tests

```bash
pixi run test          # pytest tests/ -v
```

The test suite covers regex parsing, date linkage, pseudonymisation, data loading
and splits, the DSPy helpers and checkpointing, evaluation and bootstrap CIs, the
failsafe, and the baselines. It needs neither a GPU nor a vLLM server.

---

## Project layout

```
├── main.py                      # CLI entry point (command dispatch)
├── configs/                     # example TOML configs (CPU / GPU)
├── serve.sh                     # SLURM job: vLLM server + node file
├── run_nyha.sh                  # SLURM job: pipeline client
├── scripts/
│   └── build_pseudonymisation_data.py
├── src/nyha_temporal_identification/
│   ├── extraction/              # spaCy pipeline, NYHA regexes, date parsing/linking
│   ├── data_model/              # NYHANote / NYHAMention, NYHA class ↔ vector maps
│   ├── data_handling/           # annotation formatting, loading, patient-grouped split
│   ├── dspy_logic/              # NYHAExtractor, signatures, optimisers, checkpoints
│   ├── evaluation/              # span/class/temporality metrics, error analysis, bootstrap
│   ├── models/                  # baseline classifiers and features
│   ├── utils/                   # config, I/O, text cleaning, pseudonymisation
│   └── constants.py             # all folder/file/column names
└── tests/
```

---

## Data privacy

The notes are de-identified clinical records. **Never commit data.** The data
folder lives outside the repository, and `.hf_token`, `data/` and `logs/` are
gitignored. With `pseudonymise = true`, the text seen by the extractor and the LLM
contains no deduce placeholders, but the substituted values are synthetic and the
notes must still be handled under your institution's data agreement.

---

## Citation

If you use this code, the pipeline design or the results in your work, please
cite:

```bibtex
@article{gibello_nyha_temporal_TODO,
  title   = {TODO: paper title},
  author  = {Gibello, Riccardo and TODO},
  journal = {TODO: journal / conference},
  year    = {TODO},
  volume  = {TODO},
  pages   = {TODO},
  doi     = {TODO},
  url     = {TODO}
}
```

---

## License

This project is released for **non-commercial research use only** under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
The full legal text is in the [`LICENSE`](LICENSE) file.

You may use, share and adapt the code, provided that:

- **Attribution:** you credit the authors and cite the work (see [Citation](#citation)).
- **NonCommercial:** you do not use it for commercial purposes. This includes
  clinical, diagnostic or other production use.
- **ShareAlike:** you distribute any modified version under the same license.

The software is provided "as is", without warranty of any kind. It is a research
prototype and **not a medical device**: do not use its output to make clinical
decisions.

Third-party data and models keep their own licenses. These include the
pseudonymisation lookup data downloaded by `scripts/build_pseudonymisation_data.py`
(e.g. GeoNames, CC BY 4.0) and the Gemma model weights (Gemma Terms of Use).
