#!/usr/bin/env bash
# Download a model's weights into the Hugging Face cache under $WORK.
#
#   scripts/download_weights.sh configs/models/qwen3.8-27b-nothink.yaml
#
# Repo and revision come from the model YAML. For Qwen/Qwen3.8-27B this is 55.6 GB (18
# safetensors shards plus tokenizer and config files), Apache-2.0, not gated, so no token is
# needed. It is network and disk I/O only, no compute; still, prefer an idle moment on the
# login node. Re-running resumes: finished files are skipped.
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
echo "repo=$REPO revision=$REVISION -> $HF_HOME"

export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1
# hf_xet's many threads do not get along with the login node's memory cap; plain HTTP is fine
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60
uvx --from "huggingface_hub>=0.34" hf download "$REPO" --revision "$REVISION" --max-workers 4

du -sh "$HF_HOME/hub/models--${REPO//\//--}"
