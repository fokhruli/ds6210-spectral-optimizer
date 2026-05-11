#!/usr/bin/env bash
#
# smoke_task_c.slurm -- non-catastrophic check on Task C (teacher-student).
#
# 3-seed smoke comparing AdamW vs Spec alpha=1.0 on Task C at both
# spectrum slopes alpha_C in {0.3, 1.0}.
# (Note: alpha_C is the teacher-student spectrum slope, distinct from
# the spectral correction strength which we still pass via
# --correction-strength.)
#
# Total: 2 spectrum slopes * 2 optimizers * 3 seeds = 12 runs.
# Task C model is small (width=128, depth=8); ~30 s/run -> ~6 min total.
#
# Submit:
#     STUDENT_ID=sdh4py sbatch slurm/smoke_task_c.slurm
#
#SBATCH --job-name=spec_smoke_task_c
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/smoke_task_c_%j.out
#SBATCH --error=slurm/logs/smoke_task_c_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[smoke_task_c] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

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
SPECTRUM_ALPHAS="${SPECTRUM_ALPHAS:-0.3 1.0}"
LR="${LR:-3e-3}"
WD="${WD:-1e-4}"
EPS="${EPS:-1e-8}"
SPEC_ALPHA="${SPEC_ALPHA:-1.0}"

OUT="logs/smoke_task_c_${JOB}"
echo "[smoke_task_c] STUDENT_ID=${STUDENT_ID}  N_SEEDS=${N_SEEDS}"
echo "[smoke_task_c] LR=${LR}  WD=${WD}  EPS=${EPS}  SPEC_ALPHA=${SPEC_ALPHA}"
echo "[smoke_task_c] spectrum slopes (alpha_C) = ${SPECTRUM_ALPHAS}"

# ----- AdamW arm (covers both spectrum slopes in one run_grid call) -----
echo "[smoke_task_c] === AdamW ==="
python experiments/run_grid.py \
    --task task_c --alphas ${SPECTRUM_ALPHAS} \
    --optimizers adamw \
    --grid-mode smoke \
    --smoke-lr "${LR}" --smoke-wd "${WD}" --smoke-eps "${EPS}" \
    --all-seeds --n-seeds "${N_SEEDS}" \
    --student-id "${STUDENT_ID}" \
    --max-steps 2000 --batch-size 256 \
    --out-dir "${OUT}/adamw" \
    --telemetry-suffix "_taskC"

# ----- SpectralOptimizer alpha=1.0 arm -----
echo "[smoke_task_c] === Spec corr_strength=${SPEC_ALPHA} ==="
python experiments/run_grid.py \
    --task task_c --alphas ${SPECTRUM_ALPHAS} \
    --optimizers spectral --mode production \
    --spectral-rank 8 --spectral-beta 0.95 \
    --correction-strength "${SPEC_ALPHA}" \
    --grid-mode smoke \
    --smoke-lr "${LR}" --smoke-wd "${WD}" --smoke-eps "${EPS}" \
    --all-seeds --n-seeds "${N_SEEDS}" \
    --student-id "${STUDENT_ID}" \
    --max-steps 2000 --batch-size 256 \
    --out-dir "${OUT}/spec" \
    --telemetry-suffix "_taskC_a${SPEC_ALPHA}"

echo "[smoke_task_c] done."
