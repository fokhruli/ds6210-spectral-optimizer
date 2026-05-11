#!/usr/bin/env bash
#
# grid_spectral.slurm -- SpectralOptimizer (production mode) full course
# grid on Task B.  Same 42 configs x 2 kappas x 3 seeds = 252 runs as
# the baseline; the head-to-head comparison the rubric expects.
#
# Submit with:
#     STUDENT_ID=<your_id> CORRECTION_STRENGTH=0.5 sbatch slurm/grid_spectral.slurm
#
# Knobs you most likely want to ablate (CLI envs):
#     CORRECTION_STRENGTH (default 0.5) -- 0.0 disables (== AdamW)
#     SPECTRAL_RANK       (default 8)
#     SPECTRAL_BETA       (default 0.95)
#
#SBATCH --job-name=spec_grid_spectral
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/grid_spectral_%j.out
#SBATCH --error=slurm/logs/grid_spectral_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[grid_spectral] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

module purge || true
module load miniforge || true
VENV="${SPECTRAL_VENV:-/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -V

mkdir -p slurm/logs logs telemetry

STUDENT_ID="${STUDENT_ID:-starter}"
CORRECTION_STRENGTH="${CORRECTION_STRENGTH:-0.5}"
SPECTRAL_RANK="${SPECTRAL_RANK:-8}"
SPECTRAL_BETA="${SPECTRAL_BETA:-0.95}"

echo "[grid_spectral] STUDENT_ID=${STUDENT_ID}"
echo "[grid_spectral] correction_strength=${CORRECTION_STRENGTH}"
echo "[grid_spectral] spectral_rank=${SPECTRAL_RANK}  spectral_beta=${SPECTRAL_BETA}"

CMD=(python experiments/run_grid.py
     --task task_b
     --kappas 1e3 1e6
     --optimizers spectral
     --mode production
     --spectral-rank "${SPECTRAL_RANK}"
     --spectral-beta "${SPECTRAL_BETA}"
     --correction-strength "${CORRECTION_STRENGTH}"
     --grid-mode full
     --all-seeds
     --student-id "${STUDENT_ID}"
     --max-steps 2000
     --batch-size 256
     --out-dir "logs/grid_spectral_cs${CORRECTION_STRENGTH}_r${SPECTRAL_RANK}_${SLURM_JOB_ID:-local}")
echo "[grid_spectral] running: ${CMD[*]}"
"${CMD[@]}"
