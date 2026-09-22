#!/bin/bash
# Build outputs/tokenizers/<model id>/ from a Ministral repo in the Hugging Face cache: the four
# Hugging Face tokenizer files and nothing else. transformers 5.16 picks MistralCommonBackend
# for any repo that contains tekken.json, whatever tokenizer_class says, and that backend has no
# chat-template support (jobs 3461089 and 3461125: every chat request 501). Pointing vLLM's
# --tokenizer at a directory without tekken.json makes it load LlamaTokenizerFast from
# tokenizer.json instead. Run once per Ministral model, after its weights are downloaded:
#
#   scripts/setup_ministral_tokenizer.sh configs/models/ministral-3-8b.yaml
set -euo pipefail
MODEL_YAML="${1:?usage: $0 configs/models/<ministral>.yaml}"
cd "$(dirname "$0")/.."
ID=$(uv run python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['id'])" "$MODEL_YAML")
REPO=$(uv run python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['hf_repo'])" "$MODEL_YAML")
REV=$(uv run python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['revision'])" "$MODEL_YAML")
HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
SNAP="$HF_HOME/models--${REPO//\//--}/snapshots/$REV"
[ -d "$SNAP" ] || { echo "no snapshot at $SNAP (download the weights first)" >&2; exit 1; }
DEST="outputs/tokenizers/$ID"
mkdir -p "$DEST"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja; do
  cp -L "$SNAP/$f" "$DEST/$f"
done
echo "wrote $DEST: $(ls "$DEST" | tr '\n' ' ')"
