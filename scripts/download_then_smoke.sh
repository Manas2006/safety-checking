#!/usr/bin/env bash
# Download each model's weights, check them, and submit ONE smoke job per model. Made to be
# left running under nohup on the login node: network, disk and `sleep` only, no compute.
#
#   nohup scripts/download_then_smoke.sh <allocation> configs/models/a.yaml configs/models/b.yaml \
#     > outputs/logs/download-then-smoke.log 2>&1 &
#
# The allocation is the first argument because it is never written in a file; it is not echoed
# either. Extra sbatch arguments (a node to avoid, a partition) go in SBATCH_ARGS:
#
#   SBATCH_ARGS="--exclude=c301-002" nohup scripts/download_then_smoke.sh ...
#
# Per model, in order: download (up to 3 attempts; it resumes), scripts/verify_weights.py, then
# exactly one `sbatch` of configs/smoke.yaml against that model. A model whose download or
# check fails is skipped, not submitted. A job that fails is NOT resubmitted: that is a decision
# for a person who has read the log. The job for one model queues while the next downloads.
#
# gpu-a100-dev allows 3 submitted jobs and 1 running per user, so a submission refused for a
# queue limit is retried every 2 minutes for up to 6 hours; any other refusal is final.
# To stop everything: `touch outputs/logs/STOP` (checked before every download and every
# submission), then `scancel` what is queued.
set -uo pipefail

ALLOCATION="${1:?usage: download_then_smoke.sh <allocation> <model.yaml>...}"
shift
[ "$#" -gt 0 ] || { echo "no model configs given" >&2; exit 2; }
cd "$(dirname "$0")/.."

EXPERIMENT="configs/smoke.yaml"
LOG_DIR="outputs/logs"
STOP_FILE="$LOG_DIR/STOP"
JOBS_FILE="$LOG_DIR/download-then-smoke.jobs"
mkdir -p "$LOG_DIR"
read -r -a EXTRA <<< "${SBATCH_ARGS:-}"

stamp() { date "+%F %T"; }
stopped() { [ -e "$STOP_FILE" ] && echo "$(stamp) == $STOP_FILE exists: stopping" ; }

for CONFIG in "$@"; do
  stopped && exit 0
  MODEL_ID="$(.venv/bin/sc serve-args "$CONFIG" --field id)" || { echo "cannot read $CONFIG" >&2; continue; }
  echo "$(stamp) == $MODEL_ID: download"

  OK=0
  for ATTEMPT in 1 2 3; do
    if scripts/download_weights.sh "$CONFIG" && .venv/bin/python scripts/verify_weights.py "$CONFIG"; then
      OK=1
      break
    fi
    echo "$(stamp) == $MODEL_ID: attempt $ATTEMPT did not give complete weights"
    sleep 30
  done
  if [ "$OK" -ne 1 ]; then
    echo "$(stamp) == $MODEL_ID: SKIPPED, weights incomplete after 3 attempts; nothing submitted"
    echo "$MODEL_ID skipped-incomplete-weights" >> "$JOBS_FILE"
    continue
  fi

  SUBMITTED=""
  for _ in $(seq 1 180); do
    stopped && exit 0
    if OUT="$(sbatch -A "$ALLOCATION" -J "sc-smoke-$MODEL_ID" "${EXTRA[@]}" \
        scripts/serve_and_run.slurm "$EXPERIMENT" run "$CONFIG" 2>&1)"; then
      SUBMITTED="$(echo "$OUT" | grep -o 'Submitted batch job [0-9]*' | grep -o '[0-9]*$')"
      break
    fi
    # TACC prints a banner; the reason is in its last lines. Only a queue limit is worth
    # waiting out: anything else (a bad argument, an expired allocation) will not fix itself.
    REASON="$(echo "$OUT" | tail -n 3 | tr '\n' ' ')"
    if ! echo "$OUT" | grep -qi "limit"; then
      echo "$(stamp) == $MODEL_ID: sbatch failed and it is not a queue limit: $REASON"
      break
    fi
    echo "$(stamp) == $MODEL_ID: sbatch refused ($REASON); retrying in 2 min"
    sleep 120
  done
  if [ -n "$SUBMITTED" ]; then
    echo "$(stamp) == $MODEL_ID: submitted job $SUBMITTED -> $LOG_DIR/sc-smoke-$MODEL_ID-$SUBMITTED.out"
    echo "$MODEL_ID $SUBMITTED" >> "$JOBS_FILE"
  else
    echo "$(stamp) == $MODEL_ID: NOT submitted (see above)"
    echo "$MODEL_ID not-submitted" >> "$JOBS_FILE"
  fi
done
echo "$(stamp) == done. Jobs:"
cat "$JOBS_FILE" 2>/dev/null
