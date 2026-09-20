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
# "CUDA Version: 13.x" -> cu130. A cu130 wheel on a 12.x driver fails at startup with
# "CUDA driver version is insufficient". If even cu129 will not load, the fallback is the
# official container (vllm/vllm-openai:v${VLLM_VERSION}-cu129) via tacc-apptainer, pulled and
# run on a compute node only.
set -euo pipefail

VARIANT="${1:?usage: setup_vllm_env.sh cu129|cu130}"
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

mkdir -p "$(dirname "$VENV")"
uv venv "$VENV" --python 3.12 --allow-existing
uv pip install --python "$VENV/bin/python" --torch-backend="$VARIANT" "$SPEC"

# metadata only: no import of torch or vllm
"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
for name in ("vllm", "torch", "transformers", "huggingface_hub", "flashinfer-python"):
    try:
        print(f"{name:20s} {m.version(name)}")
    except m.PackageNotFoundError:
        print(f"{name:20s} (not installed)")
PY
du -sh "$VENV"
