#!/usr/bin/env bash
# gpt-oss only. Put the o200k_base vocabulary where openai_harmony looks for it.
#
#   scripts/setup_harmony_encodings.sh
#
# vLLM renders gpt-oss chat requests with openai_harmony, which downloads o200k_base.tiktoken
# the first time it is used. The job script serves offline, so the file has to be on disk
# already: `serve_and_run.slurm` exports TIKTOKEN_ENCODINGS_BASE=outputs/tiktoken_encodings
# when that folder exists. The bytes are the ones tiktoken already cached for our own token
# counts (its cache names the file after the SHA-1 of the URL), so this copies and downloads
# nothing. Seconds, no compute.
set -euo pipefail
cd "$(dirname "$0")/.."

# sha1("https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken")
CACHED="outputs/tiktoken_cache/fb374d419588a4632f3f557e76b4b70aebbca790"
# the hash tiktoken itself pins for o200k_base
EXPECTED="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
TARGET_DIR="outputs/tiktoken_encodings"

[ -f "$CACHED" ] || { echo "no tiktoken cache at $CACHED; see CLAUDE.md for the one download" >&2; exit 2; }
FOUND="$(sha256sum "$CACHED" | cut -d' ' -f1)"
[ "$FOUND" = "$EXPECTED" ] || { echo "$CACHED is not o200k_base (sha256 $FOUND)" >&2; exit 2; }

mkdir -p "$TARGET_DIR"
cp "$CACHED" "$TARGET_DIR/o200k_base.tiktoken"
echo "wrote $TARGET_DIR/o200k_base.tiktoken ($(du -h "$TARGET_DIR/o200k_base.tiktoken" | cut -f1))"
