#!/bin/bash
#SBATCH --job-name=nyha_vllm_server

#SBATCH --partition=gpu_a100
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12

##SBATCH --partition=gpu_h100
##SBATCH --mem=120G
##SBATCH --cpus-per-task=12

##SBATCH --partition=gpu_mig
##SBATCH --mem=40G
##SBATCH --cpus-per-task=4

#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --output=logs/server_%j.out
#SBATCH --error=logs/server_%j.err

set -e

# --- Defaults ---
MODEL="RedHatAI/gemma-3-27b-it-quantized.w4a16"
DTYPE="bfloat16"
TENSOR_PARALLEL_SIZE="1"
MAX_MODEL_LEN="25200"
MAX_NUM_SEQS="24"
GPU_MEM_UTIL="0.90"
SERVER_PORT="8000"
TOKEN_FILE="./.hf_token"
PROJECT_DIR="."
DEMO_TOKEN_BUDGET="auto"
HF_HOME_OVERRIDE=""

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model <name>            HuggingFace model name (default: google/gemma-3-12b-it)"
    echo "  -d, --dtype <dtype>           Model dtype: bfloat16 | float16 | float32 (default: bfloat16)"
    echo "  -tp, --tensor-parallel <n>    Tensor parallel size / number of GPUs (default: 1)"
    echo "  -l, --max-model-len <n>       Max model context length in tokens (default: 32768)"
    echo "  -s, --max-num-seqs <n>        Max concurrent sequences (default: 24)"
    echo "  -g, --gpu-mem-util <f>        GPU memory utilization 0-1 (default: 0.90)"
    echo "  -P, --port <n>                vLLM server port (default: 8000)"
    echo "  -t, --token-file <path>       Path to HuggingFace token file (default: ./.hf_token)"
    echo "  -p, --project-dir <path>      Project directory (default: $PROJECT_DIR)"
    echo "  -B, --demo-token-budget <n>   Override MAX_DEMO_TOKENS written to node file;"
    echo "                                'auto' = compute from budget identity; '0' = skip (default: auto)"
    echo "  -H, --hf-home <path>          HuggingFace cache dir. Falls back to HF_HOME env var,"
    echo "                                then to the HuggingFace default (~/.cache/huggingface)."
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Once running, the server address is written to logs/vllm_node_<JOB_ID>.txt."
    echo "Use that file to find the node and port for notebook or pipeline connections."
    echo ""
    echo "Examples:"
    echo "  sbatch serve.sh"
    echo "  sbatch serve.sh -m google/gemma-3-27b-it -tp 2 -l 32768"
    echo "  sbatch serve.sh -m google/gemma-3-12b-it -l 32768 -s 16 -g 0.85"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)             MODEL="$2";                shift 2 ;;
        -d|--dtype)             DTYPE="$2";                shift 2 ;;
        -tp|--tensor-parallel)  TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        -l|--max-model-len)     MAX_MODEL_LEN="$2";        shift 2 ;;
        -s|--max-num-seqs)      MAX_NUM_SEQS="$2";         shift 2 ;;
        -g|--gpu-mem-util)      GPU_MEM_UTIL="$2";         shift 2 ;;
        -P|--port)              SERVER_PORT="$2";         shift 2 ;;
        -t|--token-file)        TOKEN_FILE="$2";           shift 2 ;;
        -p|--project-dir)       PROJECT_DIR="$2";          shift 2 ;;
        -B|--demo-token-budget) DEMO_TOKEN_BUDGET="$2";    shift 2 ;;
        -H|--hf-home)           HF_HOME_OVERRIDE="$2";    shift 2 ;;
        -h|--help)              print_usage;               exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

# --- Validation ---
if [[ ! -f "$TOKEN_FILE" ]]; then
    echo "Error: token file not found at '$TOKEN_FILE'"
    exit 1
fi

# --- Info ---
echo "Job started on $(hostname) at $(date)"
echo "Model:             $MODEL"
echo "Dtype:             $DTYPE"
echo "Tensor parallel:   $TENSOR_PARALLEL_SIZE"
echo "Max model len:     ${MAX_MODEL_LEN:-'inferred'}"
echo "Max num seqs:      $MAX_NUM_SEQS"
echo "GPU mem util:      $GPU_MEM_UTIL"
echo "Server port:       $SERVER_PORT"
echo "Project directory: $PROJECT_DIR"

# --- Setup ---
cd "$PROJECT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Write node address to a file so pipeline jobs and notebooks can discover it
NODE_FILE="logs/vllm_node_${SLURM_JOB_ID}.txt"
echo "$(hostname):$SERVER_PORT" > "$NODE_FILE"

# Compute and append MAX_DEMO_TOKENS so configure_lm can enforce the input budget.
# Delegates to scripts/dspy_budget_calculator.py (pure stdlib) so the formula stays
# in sync with the Python tool. Falls back to an inline approximation if Python is
# unavailable. Both must match [dspy_pipeline] config defaults.
if [[ "$DEMO_TOKEN_BUDGET" == "auto" ]]; then
    N_LABELED=0
    N_BOOTSTRAPPED=4
    MAX_TOKENS=1536
    MAX_EVAL_TOKENS=2500
    # The 700 calculator default badly under-counts the real classify instruction
    # (~2735 tok). Use the measured overhead so the demo cap is not over-allocated.
    SYSTEM_OVERHEAD=3000
    COMPUTED_DEMO_TOKENS=-1
    if command -v python3 &>/dev/null && [[ -f "scripts/dspy_budget_calculator.py" ]]; then
        CALC_OUT=$(python3 scripts/dspy_budget_calculator.py \
            --context-window "$MAX_MODEL_LEN" \
            --max-tokens-output "$MAX_TOKENS" \
            --n-labeled "$N_LABELED" \
            --n-bootstrapped "$N_BOOTSTRAPPED" \
            --system-overhead "$SYSTEM_OVERHEAD" \
            --max-eval-tokens "$MAX_EVAL_TOKENS" 2>/dev/null) || true
        COMPUTED_DEMO_TOKENS=$(echo "$CALC_OUT" \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['max_demo_tokens_safe'] if d.get('feasible') else -1)" \
            2>/dev/null) || COMPUTED_DEMO_TOKENS=-1
    fi
    if [[ "$COMPUTED_DEMO_TOKENS" -le 0 ]]; then
        echo "Warning: budget calculator unavailable; using inline fallback"
        OVERHEAD="$SYSTEM_OVERHEAD"
        CONTEXT_BUDGET=$(( MAX_MODEL_LEN - OVERHEAD - N_BOOTSTRAPPED * MAX_TOKENS - MAX_EVAL_TOKENS - MAX_TOKENS ))
        COMPUTED_DEMO_TOKENS=$(( CONTEXT_BUDGET / (N_LABELED + N_BOOTSTRAPPED) ))
    fi
    echo "MAX_DEMO_TOKENS=${COMPUTED_DEMO_TOKENS}" >> "$NODE_FILE"
    echo "MAX_DEMO_TOKENS computed as ${COMPUTED_DEMO_TOKENS} (MAX_MODEL_LEN=${MAX_MODEL_LEN})"
elif [[ "$DEMO_TOKEN_BUDGET" != "0" ]]; then
    echo "MAX_DEMO_TOKENS=${DEMO_TOKEN_BUDGET}" >> "$NODE_FILE"
    echo "MAX_DEMO_TOKENS set to ${DEMO_TOKEN_BUDGET} (manual override)"
fi

echo "Server address written to $NODE_FILE"

export HF_TOKEN=$(cat "$TOKEN_FILE" | tr -d '[:space:]')
unset TRANSFORMERS_CACHE  # deprecated; HF_HOME takes precedence
if [[ -n "$HF_HOME_OVERRIDE" ]]; then
    export HF_HOME="$HF_HOME_OVERRIDE"
elif [[ -z "$HF_HOME" ]]; then
    echo "Warning: HF_HOME is not set and --hf-home was not provided. Model weights will be"
    echo "         cached in ~/.cache/huggingface (home NFS), which may be very slow on this cluster."
    echo "         Pass -H <path> or export HF_HOME=<fast-storage-path> before submitting."
fi
echo "HF_HOME:           ${HF_HOME:-'(default: ~/.cache/huggingface)'}"

# --- Build vLLM serve command ---
VLLM_ARGS=(
    serve "$MODEL"
    --host 0.0.0.0
    --port "$SERVER_PORT"
    --dtype "$DTYPE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
)
[ -n "$MAX_MODEL_LEN" ] && VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN") && VLLM_ARGS+=(--enable-prefix-caching)

echo "Starting vLLM server on $(hostname):$SERVER_PORT..."
pixi run -e gpu vllm "${VLLM_ARGS[@]}"