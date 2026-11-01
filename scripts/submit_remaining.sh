#!/usr/bin/env bash
# Batch of 2026-10-27: the lines that did not fit under the 30-job cap on 2026-10-27, in order.
# Run once, after `bash scripts/check_batch.sh`, when squeue shows room, with the allocation as
# the argument (CLAUDE.md: no allocation is written in any script):
#
#   bash scripts/submit_remaining.sh <allocation>
#
# Each job is named after the log file it writes and carries --dependency=singleton, so it
# waits behind any queued or running job of the same name. Delete a line once it is submitted.
set -euo pipefail
cd "$(dirname "$0")/.."
ACCOUNT="${1:?usage: bash scripts/submit_remaining.sh <allocation>}"
