#!/usr/bin/env bash
# Where the batch of 2026-10-27 stands. Safe to run as often as you like: it submits nothing,
# calls no model, and rewrites the batch's sections of results/2026-10-27-followups.md in place.
#
#   bash scripts/check_batch.sh
#
# 1. Slurm's view of every job since 2026-10-27 (names are the log files they write).
# 2. scripts/export_results.py: results/tables from outputs/runs.
# 3. scripts/batch_writeup.py: the tables-only sections for the new configs, and a one-screen
#    summary of the risky-cell check rate for every new arm.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1

echo "== jobs since 2026-10-27"
sacct -u "$USER" -S 2026-10-27 -X -o JobID,JobName%28,Partition,State,ExitCode,Start,Elapsed
echo
echo "== still queued"
squeue -u "$USER" -o "%.10i %.28j %.10P %.9T %.20E %R"
echo
echo "== export"
uv run python scripts/export_results.py | tail -n 3
echo
echo "== write-up and summary"
uv run python scripts/batch_writeup.py
echo
if [ -s scripts/submit_remaining.sh ] && grep -q '^sbatch' scripts/submit_remaining.sh; then
  echo "== not yet submitted: $(grep -c '^sbatch' scripts/submit_remaining.sh) lines in scripts/submit_remaining.sh"
fi
