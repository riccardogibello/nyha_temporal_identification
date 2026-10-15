#!/bin/bash
#SBATCH --job-name=nyha_pipeline
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=10:00:00
#SBATCH --mem=60G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

log_phase() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# --- Defaults ---
DATA_FOLDER=""
CONFIG_FILE=""
PROJECT_DIR="."
PIXI_ENV="default"
RESET_PREVIOUS_DATA=false
SERVER_JOB_ID=""
KILL_SERVER=false
BASE_URL=""
TUNNEL_PID=""
LOCAL_VLLM_PORT="8000"
OVERRIDES=()
TEMP_CONFIG=""

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d, --data-folder <path>    Data folder path (required)"
    echo "  -f, --config <path>         Path to TOML configuration file (required)"
    echo "  -p, --project-dir <path>    Project directory (default: .)"
    echo "  -e, --environment <name>    Pixi environment: default | gpu (default: default)"
    echo "  -r, --reset-previous-data   Reset previous data (flag)"
    echo "  -s, --server-job-id <id>    Slurm job ID of a running serve.sh job (optional)."
    echo "                              If provided, reads the node file to discover the GPU"
    echo "                              node address and sets VLLM_BASE_URL directly to it."
    echo "  -b, --base-url <url>        vLLM base URL to use directly, e.g."
    echo "                              http://gcn36.local.snellius.surf.nl:8000/v1."
    echo "                              Exported as VLLM_BASE_URL; use with --server-job-id"
    echo "                              to still read MAX_DEMO_TOKENS from the node file."
    echo "  -k, --kill-server           After the pipeline finishes successfully, scancel the"
    echo "                              serve.sh job given by --server-job-id (flag). On failure"
    echo "                              the server is left running for debugging."
    echo "  -O, --override KEY=VALUE    Override a TOML config value at runtime without editing the file."
    echo "                              Top-level: KEY=VALUE (e.g. seed=1)."
    echo "                              Nested: SECTION.KEY=VALUE (e.g. dspy_pipeline.num_threads=32)."
    echo "                              String values need TOML quotes: dspy_pipeline.task_lm=\\\"model\\\"."
    echo "                              Repeatable; applied in order."
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "All other pipeline options (commands, token file, model name, etc.)"
    echo "are read from the TOML config file passed via --config."
    echo ""
    echo "Examples:"
    echo "  # With vLLM server running as a separate job:"
    echo "  sbatch run_pipeline.sh \\"
    echo "      -d /gpfs/work2/0/aus20644/project/rgibello/data/ \\"
    echo "      -f configs/config_gpu.toml \\"
    echo "      -p /gpfs/home4/rgibello/code/nyha_temporal_identification \\"
    echo "      -s <SERVER_JOB_ID>"
    echo ""
    echo "  # No server (extract_nyha command only, no vLLM needed):"
    echo "  sbatch run_pipeline.sh \\"
    echo "      -d /gpfs/work2/0/aus20644/project/rgibello/data/ \\"
    echo "      -f configs/config_cpu.toml \\"
    echo "      -p /gpfs/home4/rgibello/code/nyha_temporal_identification"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-folder)         DATA_FOLDER="$2";    shift 2 ;;
        -f|--config)              CONFIG_FILE="$2";    shift 2 ;;
        -p|--project-dir)         PROJECT_DIR="$2";    shift 2 ;;
        -e|--environment)         PIXI_ENV="$2";       shift 2 ;;
        -r|--reset-previous-data) RESET_PREVIOUS_DATA=true; shift ;;
        -s|--server-job-id)       SERVER_JOB_ID="$2"; shift 2 ;;
        -k|--kill-server)         KILL_SERVER=true;   shift ;;
        -b|--base-url)            BASE_URL="$2";      shift 2 ;;
        -O|--override)            OVERRIDES+=("$2");  shift 2 ;;
        -h|--help)                print_usage;         exit 0 ;;
        *) echo "Unknown option: $1"; print_usage; exit 1 ;;
    esac
done

# --- Validation ---
if [[ -z "$DATA_FOLDER" ]]; then
    echo "Error: --data-folder is required"
    print_usage; exit 1
fi
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config is required"
    print_usage; exit 1
fi
if [[ "$KILL_SERVER" == true && -z "$SERVER_JOB_ID" ]]; then
    echo "Warning: --kill-server given without --server-job-id; nothing to cancel."
fi

# --- Cleanup trap: ensure SSH tunnel is closed on exit in any case (normal, error, cancelled) ---
cleanup() {
    if [[ -n "$TUNNEL_PID" ]]; then
        echo "Closing SSH tunnel (PID $TUNNEL_PID)..."
        kill "$TUNNEL_PID" 2>/dev/null || true
    fi
    if [[ -n "$TEMP_CONFIG" && -f "$TEMP_CONFIG" ]]; then
        rm -f "$TEMP_CONFIG"
    fi
}
trap cleanup EXIT

# --- Info ---
echo "Job started on $(hostname) at $(date)"
echo "Data folder:       $DATA_FOLDER"
echo "Config file:       $CONFIG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "Pixi environment:  $PIXI_ENV"
echo "Server job ID:     ${SERVER_JOB_ID:-'Not specified'}"
echo "Base URL:          ${BASE_URL:-'Not specified — derived from server job or config'}"

# --- Setup ---
cd "$PROJECT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Check whether any configured command needs a vLLM server ---
VLLM_COMMANDS=("run_dspy_pipeline" "annotate_training_nyha_mentions")
NEEDS_VLLM=false
COMMANDS_IN_CONFIG=$(python3 -c "
import re
content = open('${CONFIG_FILE}').read()
m = re.search(r'commands\s*=\s*\[([^\]]*)\]', content)
if m:
    items = re.findall(r'[\"\'](.*?)[\"\']', m.group(1))
    print(' '.join(items))
" 2>/dev/null || true)
for _cmd in $COMMANDS_IN_CONFIG; do
    for _vllm_cmd in "${VLLM_COMMANDS[@]}"; do
        if [[ "$_cmd" == "$_vllm_cmd" ]]; then
            NEEDS_VLLM=true
            break 2
        fi
    done
done

if [[ "$NEEDS_VLLM" == false ]]; then
    log_phase "No vLLM-dependent commands detected (commands: ${COMMANDS_IN_CONFIG:-unknown}); skipping vLLM setup."
    if [[ -n "$SERVER_JOB_ID" ]]; then
        log_phase "Note: --server-job-id was provided but will be ignored for this command set."
    fi
fi

# --- Export explicit base URL if provided ---
if [[ -n "$BASE_URL" ]]; then
    export VLLM_BASE_URL="$BASE_URL"
    echo "VLLM_BASE_URL=$VLLM_BASE_URL (from --base-url)"
fi

# --- Resolve vLLM server address and open SSH tunnel ---
if [[ -n "$SERVER_JOB_ID" ]] && [[ "$NEEDS_VLLM" == true ]]; then
    NODE_FILE="logs/vllm_node_${SERVER_JOB_ID}.txt"

    log_phase "Waiting for server node file at $NODE_FILE..."
    until [ -f "$NODE_FILE" ]; do
        sleep 10
    done
    VLLM_ADDR=$(head -n 1 "$NODE_FILE")
    VLLM_NODE="${VLLM_ADDR%%:*}"
    VLLM_PORT="${VLLM_ADDR##*:}"
    log_phase "vLLM server is on node: $VLLM_NODE"

    log_phase "Waiting for vLLM server at http://${VLLM_NODE}:${VLLM_PORT}/health..."
    until curl -sf "http://${VLLM_NODE}:${VLLM_PORT}/health" > /dev/null; do
        sleep 15
    done
    log_phase "vLLM server is ready."

    # Only set VLLM_BASE_URL from the node address if --base-url was not provided.
    if [[ -z "$BASE_URL" ]]; then
        export VLLM_BASE_URL="http://${VLLM_NODE}:${VLLM_PORT}/v1"
        echo "VLLM_BASE_URL=$VLLM_BASE_URL"
    fi
    export VLLM_NODE_FILE="$NODE_FILE"
    echo "VLLM_NODE_FILE=$VLLM_NODE_FILE"
fi

# --- Install / verify pixi environment ---
if [[ -d ".pixi/envs/$PIXI_ENV" ]]; then
    log_phase "Pixi environment '${PIXI_ENV}' already exists; skipping install."
else
    log_phase "Installing / verifying pixi environment '${PIXI_ENV}'..."
    pixi install --environment "$PIXI_ENV"
fi

# --- Apply config overrides ---
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    TEMP_CONFIG=$(mktemp --suffix=.toml)
    log_phase "Applying ${#OVERRIDES[@]} config override(s) to a temp copy of $CONFIG_FILE"
    python3 - "$CONFIG_FILE" "$TEMP_CONFIG" "${OVERRIDES[@]}" << 'PYEOF'
import sys, re

def toml_value(v):
    """Auto-quote bare strings; leave already-typed TOML values untouched."""
    if v in ('true', 'false'):
        return v
    try:
        int(v); return v
    except ValueError:
        pass
    try:
        float(v); return v
    except ValueError:
        pass
    # Already quoted or a TOML array/inline-table
    if v and v[0] in ('"', "'", '[', '{'):
        return v
    return f'"{v}"'

config_file, output_file, *overrides = sys.argv[1:]
with open(config_file) as f:
    content = f.read()

for override in overrides:
    dotted_key, _, value = override.partition('=')
    value = toml_value(value)
    parts = dotted_key.split('.', 1)
    if len(parts) == 2:
        section, key = parts
        pattern = rf'(\[{re.escape(section)}\][^\[]*?){re.escape(key)}\s*=\s*[^\n]+'
        new_content, n = re.subn(pattern, rf'\g<1>{key} = {value}', content, flags=re.DOTALL)
        if n == 0:
            content = re.sub(rf'(\[{re.escape(section)}\])', rf'\1\n{key} = {value}', content)
        else:
            content = new_content
    else:
        pattern = rf'^{re.escape(dotted_key)}\s*=\s*[^\n]+'
        new_content, n = re.subn(pattern, f'{dotted_key} = {value}', content, flags=re.MULTILINE)
        content = new_content if n > 0 else f'{dotted_key} = {value}\n{content}'

with open(output_file, 'w') as f:
    f.write(content)
PYEOF
    CONFIG_FILE="$TEMP_CONFIG"
    log_phase "Effective config written to $TEMP_CONFIG"
fi

# --- Build args ---
ARGS=(--data-folder "$DATA_FOLDER" --config "$CONFIG_FILE")
[ "$RESET_PREVIOUS_DATA" = true ] && ARGS+=(--reset-previous-data)

# --- Run ---
log_phase "Executing pipeline..."
pixi run --environment "$PIXI_ENV" -- python -u main.py "${ARGS[@]}"

# --- Cancel the vLLM server only on a successful run ---
if [[ "$KILL_SERVER" == true && -n "$SERVER_JOB_ID" ]]; then
    log_phase "Pipeline succeeded; cancelling server job $SERVER_JOB_ID (--kill-server)..."
    scancel "$SERVER_JOB_ID" 2>/dev/null || true
fi

log_phase "Job finished"