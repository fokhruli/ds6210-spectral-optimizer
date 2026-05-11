#!/usr/bin/env bash
#
# smoke_task_a.slurm -- non-catastrophic check on Task A (FashionMNIST).
#
# CAPSTONE rubric requires "non-catastrophic on the others" tasks.
# This is a 3-seed smoke comparing AdamW vs Spec alpha=1.0 at one
# config on Task A.  Goal: show that the spectral correction does not
# break a well-conditioned, qualitatively different task.
#
# 1 config * 2 optimizers * 3 seeds = 6 runs.
# Task A model: residual MLP width=512, depth=16, in=784, out=10.
# Per-run cost ~3-5 min on A6000 -> ~20-30 min total.
#
# We use a generic Adam-style (lr, wd, eps) since we have not run the
# full Task A grid.  Adjust via env vars if the AdamW baseline shows a
# better config:
#     LR=3e-3   WD=1e-4   EPS=1e-8
#
# Submit:
#     STUDENT_ID=sdh4py sbatch slurm/smoke_task_a.slurm
#
#SBATCH --job-name=spec_smoke_task_a
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/smoke_task_a_%j.out
#SBATCH --error=slurm/logs/smoke_task_a_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[smoke_task_a] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

module purge || true
module load miniforge || true
VENV="${SPECTRAL_VENV:-/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -V

mkdir -p slurm/logs logs telemetry

STUDENT_ID="${STUDENT_ID:-starter}"
JOB="${SLURM_JOB_ID:-local}"
N_SEEDS="${N_SEEDS:-3}"
LR="${LR:-3e-3}"
WD="${WD:-1e-4}"
EPS="${EPS:-1e-8}"
ALPHA="${ALPHA:-1.0}"

OUT="logs/smoke_task_a_${JOB}"
echo "[smoke_task_a] STUDENT_ID=${STUDENT_ID}  N_SEEDS=${N_SEEDS}"
echo "[smoke_task_a] LR=${LR}  WD=${WD}  EPS=${EPS}  ALPHA=${ALPHA}"

# ----- AdamW arm -----
echo "[smoke_task_a] === AdamW ==="
python experiments/run_grid.py \
    --task task_a \
    --optimizers adamw \
    --grid-mode smoke \
    --smoke-lr "${LR}" --smoke-wd "${WD}" --smoke-eps "${EPS}" \
    --all-seeds --n-seeds "${N_SEEDS}" \
    --student-id "${STUDENT_ID}" \
    --max-steps 2000 --batch-size 256 \
    --out-dir "${OUT}/adamw" \
    --telemetry-suffix "_taskA"

# ----- SpectralOptimizer alpha=1.0 arm -----
echo "[smoke_task_a] === Spec alpha=${ALPHA} ==="
python experiments/run_grid.py \
    --task task_a \
    --optimizers spectral --mode production \
    --spectral-rank 8 --spectral-beta 0.95 \
    --correction-strength "${ALPHA}" \
    --grid-mode smoke \
    --smoke-lr "${LR}" --smoke-wd "${WD}" --smoke-eps "${EPS}" \
    --all-seeds --n-seeds "${N_SEEDS}" \
    --student-id "${STUDENT_ID}" \
    --max-steps 2000 --batch-size 256 \
    --out-dir "${OUT}/spec" \
    --telemetry-suffix "_taskA_a${ALPHA}"

echo "[smoke_task_a] done."
