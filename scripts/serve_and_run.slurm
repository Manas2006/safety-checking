#!/bin/bash
# Serve one model with vLLM on one GPU node, run one experiment config against it, shut down.
#
#   sbatch -A <allocation> scripts/serve_and_run.slurm configs/smoke.yaml
#
# Submit from the repository root. The allocation is never written here; pass it with -A.
# Queue and time limit below are the defaults for short tests and can be overridden on the
# sbatch command line (-p gpu-a100 -t 02:00:00).
#
# Nothing about the model is hard-coded: the vllm serve command line, the port, the startup
# timeout and the pinned vLLM version all come from the model YAML named by the experiment YAML
# (see `sc serve-args`). The server binds to localhost only and is always shut down, including
# when anything below fails or the job is cancelled.
#
#SBATCH -J sc-serve
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100-dev
#SBATCH -t 00:30:00
#SBATCH -o outputs/logs/%x-%j.out
#SBATCH -e outputs/logs/%x-%j.out

set -euo pipefail

EXPERIMENT_CONFIG="${1:?usage: sbatch -A <allocation> scripts/serve_and_run.slurm <experiment.yaml>}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

JOB="${SLURM_JOB_ID:-local$$}"
SC=".venv/bin/sc"
VLLM_VENV="${VLLM_VENV:-$WORK/venvs/vllm}"
LOG_DIR="outputs/logs"
SERVER_LOG="$LOG_DIR/vllm-$JOB.log"
TIMING="$LOG_DIR/timing-$JOB.txt"
mkdir -p "$LOG_DIR"

[ -x "$SC" ] || { echo "no project venv at $SC; run 'uv sync' on the login node first" >&2; exit 2; }
[ -x "$VLLM_VENV/bin/vllm" ] || { echo "no vLLM env at $VLLM_VENV; run scripts/setup_vllm_env.sh" >&2; exit 2; }

# -- everything model-specific comes from YAML, read through `sc serve-args` ----------------
# (which accepts the experiment YAML and follows it to the model YAML: no parsing done here)
field() { "$SC" serve-args "$EXPERIMENT_CONFIG" --field "$1"; }
EXPERIMENT="$(field experiment.experiment)"
MODEL_ID="$(field id)"
HOST="$(field serve.host)"
PORT="$(field serve.port)"
TP="$(field serve.tensor_parallel_size)"
STARTUP_TIMEOUT="$(field serve.startup_timeout_s)"
PINNED_VLLM="$(field serve.vllm_version)"
BASE_URL="http://$HOST:$PORT/v1"
mapfile -t SERVE_ARGS < <("$SC" serve-args "$EXPERIMENT_CONFIG" --lines)
SERVE_ARGS[0]="$VLLM_VENV/bin/vllm"   # replace the bare "vllm" with the pinned environment's

# the installed version must be the pinned one (package metadata only, nothing heavy imported)
INSTALLED_VLLM="$("$VLLM_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm").split("+")[0])')"
if [ "$INSTALLED_VLLM" != "$PINNED_VLLM" ]; then
  echo "vLLM $INSTALLED_VLLM is installed but model config $MODEL_ID pins $PINNED_VLLM" >&2
  exit 2
fi

# -- environment ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-$WORK/cache/huggingface}"
# the same rule huggingface_hub applies, made explicit so the server reads the cache the
# download wrote to (a shell that exports HF_HUB_CACHE=$HF_HOME has no hub/ level)
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
# a placeholder token left in a shell profile would turn any Hub request into a 401
case "${HUGGINGFACE_HUB_TOKEN:-}" in ""|your_token_here) unset HUGGINGFACE_HUB_TOKEN ;; esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # weights are downloaded beforehand; never fetch here
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORK/cache/vllm}"
export TOKENIZERS_PARALLELISM=false
# use exactly as many GPUs as the config asks for (an A100 node has 3, tensor parallel is 2)
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP - 1)))"

# With -n 1 Slurm pins this script, and everything it starts, to a single core (the probe job
# saw nproc=1 on a 128-core node). The node is allocated whole, so widen the affinity to all of
# it; otherwise vLLM's tensor-parallel workers, tokenizer and API server share one core.
ALL_CPUS="$(nproc --all)"
taskset -pc "0-$((ALL_CPUS - 1))" $$ >/dev/null 2>&1 || echo "note: could not widen CPU affinity"
echo "== cpus usable: $(nproc) of $ALL_CPUS"

echo "== job $JOB on $(hostname) at $(date -Is)"
echo "== experiment: $EXPERIMENT ($EXPERIMENT_CONFIG)   model config: $MODEL_ID"
echo "== vLLM $INSTALLED_VLLM   GPUs: $CUDA_VISIBLE_DEVICES   HF_HUB_CACHE: $HF_HUB_CACHE"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi | sed -n '3,4p'   # the driver / CUDA version line, for the record
echo "== ${SERVE_ARGS[*]}"

# -- server lifecycle ----------------------------------------------------------------------
SERVER_PID=""
cleanup() {
  status=$?
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "== stopping vLLM (process group $SERVER_PID)"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
  echo "== exit status $status at $(date -Is)"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

START=$(date +%s)
# its own process group, so the tensor-parallel workers die with it
setsid "${SERVE_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "== vLLM starting (pid $SERVER_PID), log: $SERVER_LOG, timeout ${STARTUP_TIMEOUT}s"

until curl -sf -o /dev/null "http://$HOST:$PORT/health"; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "== vLLM exited during startup; last 60 lines of $SERVER_LOG:" >&2
    tail -n 60 "$SERVER_LOG" >&2
    exit 3
  fi
  if [ $(( $(date +%s) - START )) -ge "$STARTUP_TIMEOUT" ]; then
    echo "== vLLM not healthy after ${STARTUP_TIMEOUT}s; last 60 lines of $SERVER_LOG:" >&2
    tail -n 60 "$SERVER_LOG" >&2
    exit 4
  fi
  sleep 5
done
STARTUP_S=$(( $(date +%s) - START ))
echo "== vLLM healthy after ${STARTUP_S}s"
echo "startup_seconds=$STARTUP_S" > "$TIMING"

# -- what the model actually sees ------------------------------------------------------------
# Render the longest prefix through the served chat template and check that nothing was dropped.
# A failure is loud but does not stop the run unless RENDER_STRICT=1: for a smoke test it is
# more useful to see both the template problem and the behaviour it produces.
RENDER_STATUS=0
$SC render --config "$EXPERIMENT_CONFIG" --scenario sharing_risky --length 50 --base-url "$BASE_URL" \
  || RENDER_STATUS=$?
echo "render_status=$RENDER_STATUS" >> "$TIMING"
if [ "$RENDER_STATUS" -ne 0 ] && [ "${RENDER_STRICT:-0}" = "1" ]; then
  echo "== render check failed and RENDER_STRICT=1: not running the experiment" >&2
  exit 5
fi

# -- the experiment --------------------------------------------------------------------------
RUN_START=$(date +%s)
RUN_STATUS=0
$SC run --config "$EXPERIMENT_CONFIG" --base-url "$BASE_URL" || RUN_STATUS=$?
echo "run_seconds=$(( $(date +%s) - RUN_START ))" >> "$TIMING"
echo "run_status=$RUN_STATUS" >> "$TIMING"

echo "== outcome counts"
$SC counts "outputs/runs/$EXPERIMENT.jsonl" || true
echo "== timing"
cat "$TIMING"
echo "== prefix cache and throughput lines from the server log"
grep -E "Prefix cache hit rate|Avg generation throughput" "$SERVER_LOG" | tail -n 5 || true

exit "$RUN_STATUS"
