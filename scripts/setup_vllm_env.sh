#!/usr/bin/env bash
# Create the vLLM serving environment at $WORK/venvs/vllm.
#
# Separate from the project venv on purpose: it is ~8 GB (torch + CUDA libraries), which does not
# fit the $HOME quota, and the project itself must never import torch or vllm.
#
# Safe on the login node: this only downloads and unpacks wheels. It never imports torch or
# vllm (the 8 GB virtual-memory cap would kill that), and the version check at the end reads
# package metadata only.
#
#   scripts/setup_vllm_env.sh cu129     # driver >= 525 (CUDA 12.x), minor-version compatible
#   scripts/setup_vllm_env.sh cu130     # driver >= 580 only (this is the default PyPI wheel)
#
# Pick the variant from the `nvidia-smi` header on a GPU node: "CUDA Version: 12.x" -> cu129,
# "CUDA Version: 13.x" -> cu130. LS6 GPU nodes (2026-09): driver 570.195.03, CUDA 12.8 -> cu129. A cu130 wheel on a 12.x driver fails at startup with
# "CUDA driver version is insufficient". If even cu129 will not load, the fallback is the
# official container (vllm/vllm-openai:v${VLLM_VERSION}-cu129) via tacc-apptainer, pulled and
# run on a compute node only.
#
# llguidance and glibc. vLLM 0.29.0 requires llguidance>=1.7.0,<1.8.0, and every 1.7.x wheel on
# PyPI for x86_64 is manylinux_2_31; LS6 has glibc 2.28. It is the only package in the tree
# without a usable wheel. So it is compiled once on a compute node
# (scripts/build_llguidance.slurm, which writes to $WORK/wheels) and installed from there with
# --find-links. --only-binary makes that explicit: if the local wheel is missing the install
# fails at once instead of trying to compile Rust under the login node's memory cap.
#
#   scripts/setup_vllm_env.sh cu129 --llguidance-override
#
# is the fallback if that build is impossible: it pins llguidance==1.6.1, the last release with a
# manylinux_2_28 wheel. That is OUTSIDE vLLM's declared range and must be recorded as a deviation
# in SPEC.md. llguidance backs one structured-output backend that these runs do not use
# (tool_choice=auto is parsed after the fact, not grammar-constrained).
set -euo pipefail

VARIANT="${1:?usage: setup_vllm_env.sh cu129|cu130 [--llguidance-override]}"
LLGUIDANCE_MODE="${2:-wheel}"
WHEEL_DIR="${WHEEL_DIR:-$WORK/wheels}"
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"   # keep in step with serve.vllm_version in the model YAML
VENV="${VLLM_VENV:-$WORK/venvs/vllm}"

# the login node caps virtual memory at 8 GB; multithreaded uv aborts without these
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_BUILDS=1
export UV_LINK_MODE=copy

case "$VARIANT" in
  cu129)
    SPEC="vllm @ https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    ;;
  cu130)
    SPEC="vllm==${VLLM_VERSION}"
    ;;
  *)
    echo "unknown variant: $VARIANT (expected cu129 or cu130)" >&2
    exit 2
    ;;
esac

EXTRA=()
case "$LLGUIDANCE_MODE" in
  wheel)
    if ! ls "$WHEEL_DIR"/llguidance-*.whl >/dev/null 2>&1; then
      echo "no llguidance wheel in $WHEEL_DIR: run scripts/build_llguidance.slurm first" >&2
      exit 2
    fi
    echo "llguidance: locally built wheel from $WHEEL_DIR"
    EXTRA=(--find-links "$WHEEL_DIR" --only-binary llguidance)
    ;;
  --llguidance-override)
    echo "llguidance: OVERRIDE to 1.6.1, outside vLLM's declared range (record this in SPEC.md)"
    OVERRIDES="$(mktemp)"
    trap 'rm -f "$OVERRIDES"' EXIT
    echo "llguidance==1.6.1" > "$OVERRIDES"
    EXTRA=(--overrides "$OVERRIDES" --only-binary llguidance)
    ;;
  *)
    echo "unknown option: $LLGUIDANCE_MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$VENV")"
uv venv "$VENV" --python 3.12 --allow-existing
uv pip install --python "$VENV/bin/python" --torch-backend="$VARIANT" "${EXTRA[@]}" "$SPEC"

# metadata only: no import of torch or vllm
"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
for name in ("vllm", "torch", "transformers", "huggingface_hub", "flashinfer-python", "llguidance"):
    try:
        print(f"{name:20s} {m.version(name)}")
    except m.PackageNotFoundError:
        print(f"{name:20s} (not installed)")
PY
du -sh "$VENV"
