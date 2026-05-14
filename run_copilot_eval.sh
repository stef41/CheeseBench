#!/usr/bin/env bash
# Run CheeseBench against all GitHub Copilot CLI models in parallel.
# Assumes copilot_proxy.py is already running on PROXY_PORT.

set -u

PROXY_PORT="${PROXY_PORT:-9200}"
PROXY_URL="http://127.0.0.1:${PROXY_PORT}/v1/chat/completions"
NUM_TRIALS="${NUM_TRIALS:-10}"
MAX_STEPS="${MAX_STEPS:-200}"
PARALLEL="${PARALLEL:-4}"
OUT_ROOT="results/copilot_eval"

MODELS=(
    "claude-haiku-4.5"
    "claude-sonnet-4.6"
    "claude-opus-4.6"
    "claude-opus-4.7"
    "gpt-4.1"
    "gpt-5.2"
    "gpt-5.2-codex"
)

# Health check
if ! curl -sf "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null; then
    echo "FATAL: proxy not reachable at port ${PROXY_PORT}" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

run_model() {
    local model="$1"
    local safe="${model//\//_}"
    local outdir="${OUT_ROOT}/${safe}"
    local logfile="${outdir}/run.log"
    mkdir -p "$outdir"
    echo "[$(date +%H:%M:%S)] START $model -> $outdir" | tee -a "${OUT_ROOT}/_launcher.log"
    python benchmark.py \
        --model "$model" \
        --num-trials "$NUM_TRIALS" \
        --max-steps "$MAX_STEPS" \
        --view-modes ASCII_2D ASCII_2D_FPV ASCII_3D \
        --api-url "$PROXY_URL" \
        --api-format openai \
        --output-dir "$outdir" \
        --quiet \
        > "$logfile" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE  $model rc=$rc" | tee -a "${OUT_ROOT}/_launcher.log"
}
export -f run_model
export OUT_ROOT PROXY_URL NUM_TRIALS MAX_STEPS

# Use xargs -P for parallelism
printf "%s\n" "${MODELS[@]}" | xargs -n1 -P"$PARALLEL" -I{} bash -c 'run_model "$@"' _ {}

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "${OUT_ROOT}/_launcher.log"
