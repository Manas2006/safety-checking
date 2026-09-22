#!/usr/bin/env bash
# The four models without weights, as one chain for nohup on the login node: the two A100
# models first, then the two H100 models (whose smoke jobs need -p gpu-h100). Network, disk
# and sleep only. The allocation is the argument, never written here.
#
#   nohup scripts/download_round2.sh <allocation> > outputs/logs/download-round2.log 2>&1 &
#
# Progress: tail -f outputs/logs/download-round2.log. Stop: touch outputs/logs/STOP.
set -uo pipefail
ALLOCATION="${1:?usage: download_round2.sh <allocation>}"
cd "$(dirname "$0")/.."
SBATCH_ARGS="-p gpu-a100 -t 01:00:00" scripts/download_then_smoke.sh "$ALLOCATION" \
  configs/models/qwen3.5-27b-nothink.yaml configs/models/ministral-3-14b.yaml
SBATCH_ARGS="-p gpu-h100 -t 01:00:00" scripts/download_then_smoke.sh "$ALLOCATION" \
  configs/models/gemma-4-31b-nothink.yaml configs/models/gpt-oss-120b-low.yaml
