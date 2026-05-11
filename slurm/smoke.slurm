#!/usr/bin/env bash
#
# smoke.slurm -- end-to-end sanity for the DS 6210 spectral harness.
#
# Runs reproduce.sh: env diagnostic, AdamW baseline, parity test, debug
# mode on your optimizer, and a 1-config 1-seed smoke grid.  ~15-25 min
# on an A100.  Use this BEFORE launching the full grid.
#
# Submit with:
#     sbatch slurm/smoke.slurm
#
#SBATCH --job-name=spec_smoke
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/smoke_%j.out
#SBATCH --error=slurm/logs/smoke_%j.err

set -euo pipefail

# --- Pin working directory to THIS repo copy.  The insights showed past
# --- SLURM jobs got bitten by editing one repo while a job ran from
# --- another; hardcoding it kills that whole class of bug.
PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[smoke] cwd=$(pwd)"
echo "[smoke] host=$(hostname)  job=${SLURM_JOB_ID:-local}"

# --- Environment: Python 3.11 venv at ds6210/venv (sibling of project/).
module purge || true
module load miniforge || true
VENV="${SPECTRAL_VENV:-/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

mkdir -p slurm/logs logs telemetry

echo "[smoke] python=$(which python)  version=$(python -V)"
python -c "import torch; print(f'[smoke] torch={torch.__version__} cuda={torch.cuda.is_available()}')"
nvidia-smi | head -20 || true

# Determinism for the parity test:
#   NVIDIA_TF32_OVERRIDE=0       disables TF32 in cuBLAS/cuDNN globally
#                                (TF32 has ~1e-3 precision, non-bit-determ across calls)
#   CUBLAS_WORKSPACE_CONFIG=:4096:8
#                                pins cuBLAS workspace -> deterministic GEMM
#                                kernel selection (the remaining ~1e-5 noise)
# Both are required for parity to land below the 1e-6 tolerance.  The full-grid
# scripts intentionally leave these unset to keep TF32 on for fast training.
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SPECTRAL_DETERMINISTIC=1   # picked up by my_optimizer.py at import time
echo "[smoke] NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE}"
echo "[smoke] CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
echo "[smoke] SPECTRAL_DETERMINISTIC=${SPECTRAL_DETERMINISTIC}"

bash reproduce.sh
