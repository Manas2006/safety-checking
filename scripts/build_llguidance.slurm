#!/bin/bash
# Build the llguidance wheel from source on a compute node, against this system's glibc.
#
#   sbatch -A <allocation> scripts/build_llguidance.slurm
#
# Why this exists: vLLM 0.29.0 requires llguidance>=1.7.0,<1.8.0, and every 1.7.x wheel on PyPI
# for x86_64 is tagged manylinux_2_31 (needs glibc 2.31). Lonestar6 is RHEL 8 with glibc 2.28, so
# there is no usable wheel and the package must be compiled. That cannot be done on the login
# node: its 8 GB virtual-memory cap crashes rustc. A compute node has no such cap.
#
# The wheel is written to $WORK/wheels, and scripts/setup_vllm_env.sh installs from that folder
# with --find-links, so the serving environment can be rebuilt later without recompiling. It is
# the only package in vLLM's dependency tree that needs this (checked with a wheels-only dry run).
#
# CPU only. The Rust toolchain is fetched by the build backend if it is not already cached, which
# needs the network; compute nodes can reach it (scripts/probe_gpu_node.slurm).
#
#SBATCH -J sc-llguidance
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p development
#SBATCH -t 00:20:00
#SBATCH -o outputs/logs/%x-%j.out
#SBATCH -e outputs/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

LLGUIDANCE_VERSION="${LLGUIDANCE_VERSION:-1.7.6}"   # the newest release inside vLLM's range
WHEEL_DIR="${WHEEL_DIR:-$WORK/wheels}"
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy

echo "== $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}"
echo "== glibc: $(ldd --version | head -n 1)   cpus: $(nproc)   ulimit -v: $(ulimit -v)"
echo "== uv: $(uv --version)"
mkdir -p "$WHEEL_DIR"

START=$(date +%s)
# pip in a throwaway environment; --no-binary forces the source build, --no-deps keeps it to
# this one package. The wheel is abi3, so it serves any Python >= 3.9.
uvx --python 3.12 pip wheel "llguidance==$LLGUIDANCE_VERSION" \
  --no-binary llguidance --no-deps --wheel-dir "$WHEEL_DIR"
echo "== built in $(( $(date +%s) - START ))s"

WHEEL="$(ls -t "$WHEEL_DIR"/llguidance-"$LLGUIDANCE_VERSION"-*.whl | head -n 1)"
echo "== wheel: $WHEEL ($(du -h "$WHEEL" | cut -f1))"

# the point of the exercise: the newest glibc symbol the extension needs must be <= 2.28
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
unzip -q "$WHEEL" -d "$TMP"
SO="$(find "$TMP" -name '*.so' | head -n 1)"
NEEDED="$(objdump -T "$SO" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -n 1)"
echo "== newest glibc symbol required by $(basename "$SO"): $NEEDED"
sha256sum "$WHEEL" | tee "$WHEEL.sha256"
echo "== done at $(date -Is)"
