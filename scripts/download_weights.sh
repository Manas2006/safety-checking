#!/usr/bin/env bash
# Download a model's weights into the Hugging Face cache under $WORK.
#
#   scripts/download_weights.sh configs/models/qwen3.8-27b-nothink.yaml
#
# Repo, revision and the files to skip (`download.exclude`: some repos carry the same weights
# in two formats) come from the model YAML, whose header gives the size. Every configured model
# is Apache-2.0 and not gated, so no token is needed. It is network and disk I/O only, no
# compute; still, prefer an idle moment on the login node. Re-running resumes: finished files
# are skipped.
set -euo pipefail

CONFIG="${1:?usage: download_weights.sh <model.yaml>}"
cd "$(dirname "$0")/.."

: "${HF_HOME:?HF_HOME is not set; it should point under \$WORK (see ~/.bashrc)}"
case "$HF_HOME" in
  "$WORK"/*) ;;
  *) echo "refusing: HF_HOME=$HF_HOME is not under \$WORK=$WORK" >&2; exit 2 ;;
esac

REPO="$(.venv/bin/sc serve-args "$CONFIG" --field hf_repo)"
REVISION="$(.venv/bin/sc serve-args "$CONFIG" --field revision)"
mapfile -t EXCLUDE < <(.venv/bin/sc serve-args "$CONFIG" --field download.exclude)
EXCLUDE_ARGS=()
if [ "${#EXCLUDE[@]}" -gt 0 ]; then EXCLUDE_ARGS=(--exclude "${EXCLUDE[@]}"); fi
echo "repo=$REPO revision=$REVISION exclude=${EXCLUDE[*]:-none} -> $HF_HOME"

export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1
# hf_xet's many threads do not get along with the login node's memory cap; plain HTTP is fine
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60
uvx --from "huggingface_hub>=0.34" hf download "$REPO" --revision "$REVISION" --max-workers 4 \
  "${EXCLUDE_ARGS[@]}"

# huggingface_hub uses HF_HUB_CACHE when it is set, and only otherwise $HF_HOME/hub
du -sh "${HF_HUB_CACHE:-$HF_HOME/hub}/models--${REPO//\//--}"
