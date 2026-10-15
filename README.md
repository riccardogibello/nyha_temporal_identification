# NYHA Extraction Pipeline — Stages 1–4 (v2)

Multi-span NYHA functional class extraction with temporal classification
(current / historical), using a three-layer pipeline:

```
Layer 0  spaCy nyha_component   — span detection + date linkage (pre-LLM)
Layer 1  DSPy LLM (MedGemma)    — span identification, guided by Layer 0
Layer 2  DSPy LLM (MedGemma)    — per-span classification with date context
Layer 3  Post-processing        — index resolution + multi-signal temporality
```

Infrastructure: DSPy + MedGemma-27B-IT via vLLM.

---

## What's new in v2 (vs. v1)

### spaCy pre-processing (nyha_parsing.py integration)

The `nyha_component` spaCy pipe from `nyha_parsing.py` now runs **before**
the LLM as Layer 0. It contributes three things v1 lacked:

**1. Date linkage**
Each NYHA mention is linked to the nearest preceding date within a
configurable character window. The linked date and character distance
are injected into the LLM's classification prompt explicitly:

```
linked_date_context: "'March 2019' (180 chars before mention)"
```

This directly addresses `implicit_temporality` errors — cases where no
anchor word is present but a preceding date unambiguously signals
a historical status.

**2. Multi-signal temporality inference**
`infer_temporality()` now runs a priority-ordered signal stack:

```
1. Date distance heuristic   (char_distance > 200 → historical)
2. Authored-date lag         (linked_date > 30 days before note → historical)
3. Linguistic anchor words   (previously / currently / etc.)
4. Fallback                  → ambiguous
```

The LLM's temporality label is enriched (not blindly overridden) by this
stack after generation — ambiguous LLM outputs are resolved where possible,
confident ones are preserved.

**3. Improved span detection**
The spaCy regex patterns handle:
- Dutch clinical text (`ny_nl`, `een/twee/drie/vier`)
- Multi-class range mentions: "NYHA III-IV" → `nyha_vector = [0,0,1,1]`
- Arbitrary separators between tokens (`nyha-class-iii`, `nyha / class / 3`)
- Reverse word order: "Class III NYHA"

### Subdivided error types

`implicit_temporality` (v1) is now split into two subtypes:

| Error type | Meaning |
|---|---|
| `implicit_no_date` | No anchor words AND no linked date found — truly ambiguous context |
| `implicit_date_missed` | No anchor words but a linked date WAS available — date-linkage inference failed to resolve |

This distinction guides debugging: `implicit_date_missed` suggests tuning
`HISTORICAL_DATE_LAG_DAYS` or `HISTORICAL_DISTANCE_THRESHOLD` in `models.py`;
`implicit_no_date` cases need synthetic augmentation with implicit-temporality
examples.

### `authored_datetime` propagated throughout

`NYHADocument.authored_datetime` (ISO string) is now passed through every
evaluation call in `optimizer.py`, enabling the date-lag heuristic at
inference time. Notes without an authored date gracefully skip that signal.

---

## Repository structure

```
nyha_pipeline/
├── models.py          # Data models, date linkage, multi-signal temporality
├── dspy_modules.py    # Three-layer extraction module (spaCy + LLM + post-proc)
├── evaluation.py      # Mention-level F1, error analysis, per-cell reporting
├── optimizer.py       # Steps 1–4 pipeline runner with authored_datetime support
├── main.py            # CLI entry point
├── requirements.txt
└── README.md

# External dependency (your existing codebase):
src/nyha_parsing.py    # nyha_component, find_nyha_class, link_nyha_to_date
```

---

## Design decisions

### Span identity = character indices, not text

Multiple mentions can share identical text ("NYHA Class II" twice).
Resolution strategy:

1. LLM outputs: `span` (text) + `occurrence` (1-indexed)
2. `resolve_indices()` finds the nth occurrence deterministically
3. `validate_against_text()` cross-checks against note text
4. LLMs never output raw character counts (unreliable)

### Division of labour: spaCy vs. LLM

| Task | Assigned to | Reason |
|---|---|---|
| Span detection (recall) | spaCy regex | Deterministic; high recall |
| Span detection (coverage check) | LLM | Catches patterns regex misses |
| Date extraction & linking | spaCy (existing code) | Reliable; already tested |
| Temporality classification | LLM + multi-signal stack | Requires contextual reasoning |
| Character index resolution | Deterministic post-proc | LLMs cannot count reliably |

### Temporality inference priority

```python
# From models.infer_temporality()
1. char_distance > HISTORICAL_DISTANCE_THRESHOLD (200)  → historical
2. authored_date - linked_date > HISTORICAL_DATE_LAG_DAYS (30d) → historical
3. linked_date within lag threshold                      → current
4. Linguistic anchor words (CURRENT_RE / HISTORICAL_RE)
5. Any linked date at non-trivial distance               → historical
6. Fallback                                              → ambiguous
```

Thresholds are configurable constants in `models.py`.

---

## Infrastructure

### vLLM server

```bash
# Two GPUs, full precision (recommended)
vllm serve google/medgemma-27b-it \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --served-model-name medgemma-27b-it

# Single GPU, 4-bit AWQ quantisation
vllm serve google/medgemma-27b-it \
    --dtype bfloat16 \
    --quantization awq \
    --max-model-len 4096 \
    --served-model-name medgemma-27b-it
```

HuggingFace authentication (MedGemma is gated):
```bash
huggingface-cli login
```

---

## Dataset format

```json
{
  "6398937" : {
    "old_note_id": "6398937",
    "authored": "2023-04-15T10:30:00",
    "deidentified_text": "Patient was previously NYHA Class III. Now NYHA Class II after optimisation.",
    "extracted_nyha_mentions": [
      {
        "original_text": "NYHA III",
        "text": "nyha iii",
        "classes": [
            3
        ],
        "one_hot": [
            0,
            0,
            1,
            0
        ],
        "start": 22,
        "end": 36,


        "char_start": 22,
        "char_end": 36,
        "span": "NYHA Class III",
        "nyha_class": "III",
        "nyha_vector": [0, 0, 1, 0],
        "temporality": "historical",
        "evidence": "previously",
        "linked_date": null,
        "char_distance": null
      },
      {
        "original_text": "NYHA Class II",
        "text": "nyha class ii",
        "classes": [
            2
        ],
        "one_hot": [
            0,
            1,
            0,
            0
        ],
        "start": 42,
        "end": 55,


"span": "NYHA Class II",
        "nyha_class": "II",
        "nyha_vector": [0, 1, 0, 0],
        "temporality": "current",
        "evidence": "Now",
        "linked_date": null,
        "char_distance": null
      }
    ]
  }
}
```

Notes:
- `authored_datetime`: optional but strongly recommended — enables date-lag
  temporality inference. ISO 8601 format.
- `nyha_vector`: optional in gold data; populated automatically at inference.
  For range mentions ("NYHA III-IV"): `[0, 0, 1, 1]`.
- `linked_date` / `char_distance`: optional in gold data; populated by
  spaCy pre-processing at inference time.
- `char_start` / `char_end` are validated on load. Mismatched spans are
  skipped with a warning.

---

## Running the pipeline

### Full optimisation run (Steps 1–4)

```bash
python main.py \
    --dataset data/nyha_annotated.json \
    --output  outputs/run_01 \
    --optimiser bootstrap
```

### MIPRO (rewrites prompt instructions, slower)

```bash
python main.py \
    --dataset data/nyha_annotated.json \
    --output  outputs/run_mipro \
    --optimiser mipro
```

### Both optimisers (auto-selects best on val set)

```bash
python main.py \
    --dataset data/nyha_annotated.json \
    --output  outputs/run_both \
    --optimiser both
```

### Inference only

```bash
python main.py \
    --dataset     data/new_notes.json \
    --load-extractor outputs/run_01/optimised_bootstrap.json \
    --infer-only \
    --output      outputs/inference_01
```

---

## Output artefacts

```
outputs/run_01/
├── baseline_predictions.json     # Step 1: zero-shot predictions
├── error_analysis.json           # Step 2: error categorisation + weak cells
├── optimised_bootstrap.json      # Step 3a: saved DSPy program
├── optimised_mipro.json          # Step 3b (if run)
├── final_predictions.json        # Step 4: test-set predictions (with linked_date)
└── final_report.json             # Step 4: cell F1 + weak_cells list
```

`error_analysis.json` now distinguishes `implicit_no_date` vs.
`implicit_date_missed`, giving clearer guidance on whether to tune
thresholds or generate more synthetic data.

`weak_cells` in `final_report.json` feeds directly into the Stage 5
synthetic augmentation pipeline as priority generation targets.

---

## Connecting to Stage 5 (soft prompt tuning)

After Step 4, the optimised extractor serves two roles:

**Synthetic data annotator**: run it over generated notes. The
`linked_date` and `char_distance` fields in predictions provide
additional validation signal — if a high-confidence LLM label
contradicts the date-distance heuristic, flag for human review.

**Performance ceiling**: soft prompt tuning must beat this F1 on the
same held-out test set to justify training cost.

---

## Tunable constants (models.py)

| Constant | Default | Effect |
|---|---|---|
| `HISTORICAL_DATE_LAG_DAYS` | 30 | Days lag between note date and linked date to trigger historical |
| `CURRENT_DISTANCE_THRESHOLD` | 50 | Chars — linked date closer than this → lean current |
| `HISTORICAL_DISTANCE_THRESHOLD` | 200 | Chars — linked date farther than this → lean historical |
| `WINDOW` | 120 | Chars around mention searched for anchor words |
