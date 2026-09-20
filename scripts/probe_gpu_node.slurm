#!/bin/bash
# One-minute probe of a GPU node: which driver and CUDA version it has (decides the vLLM wheel,
# cu129 or cu130) and whether it can reach Hugging Face and PyPI (decides whether anything can
# be fetched inside a job, or everything must be staged from the login node first).
#
#   sbatch -A <allocation> scripts/probe_gpu_node.slurm
#
# Imports nothing, downloads nothing, and takes a few seconds.
#
#SBATCH -J sc-probe
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu-a100-dev
#SBATCH -t 00:05:00
#SBATCH -o outputs/logs/%x-%j.out
#SBATCH -e outputs/logs/%x-%j.out

echo "== probe on $(hostname) at $(date -Is), job ${SLURM_JOB_ID:-none}, partition ${SLURM_JOB_PARTITION:-?}"

echo "== nvidia-smi header (driver / CUDA version)"
nvidia-smi | sed -n '1,4p'
echo "== GPUs"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv

reach() {
  local code
  code="$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$1" 2>/dev/null)"
  if [ -n "$code" ] && [ "$code" != "000" ]; then
    echo "REACHABLE    $1 (HTTP $code)"
  else
    echo "UNREACHABLE  $1"
  fi
}
echo "== reachability from the compute node"
reach https://huggingface.co
reach https://pypi.org/simple/

echo "== also relevant"
echo "glibc: $(ldd --version 2>&1 | head -n 1)"          # the vLLM wheel is manylinux_2_28
echo "kernel: $(uname -r)"
echo "cpus: $(nproc)   memory: $(free -g | awk '/^Mem:/ {print $2}') GB"
echo "/tmp free: $(df -h /tmp | awk 'NR==2 {print $4}')"
echo "WORK visible: $([ -d "$WORK" ] && echo yes || echo no)   HOME venv visible: $([ -x .venv/bin/sc ] && echo yes || echo no)"
echo "HF_HOME=${HF_HOME:-unset}"
echo "== done at $(date -Is)"
