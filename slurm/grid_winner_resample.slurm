#!/usr/bin/env bash
#
# grid_winner_resample.slurm -- Step #3 of the report plan.
#
# Re-runs the head-to-head at kappa=10^3 with N=10 SHA-derived seeds
# instead of the original 3.  Two arms:
#   (A) AdamW at its winning config (lr=3e-3, wd=1e-2, eps=1e-8)
#   (B) SpectralOptimizer at alpha=1.0 winning config (lr=3e-3, wd=0, eps=1e-8)
#
# Total: 20 runs of 2000 steps = ~15-20 min on a single A6000.
#
# Submit:
#     STUDENT_ID=<your_id> sbatch slurm/grid_winner_resample.slurm
#
# Knobs:
#     N_SEEDS  (default 10)
#
#SBATCH --job-name=spec_winner_resample
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/winner_resample_%j.out
#SBATCH --error=slurm/logs/winner_resample_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[winner_resample] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

module purge || true
module load miniforge || true
VENV="${SPECTRAL_VENV:-/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -V

mkdir -p slurm/logs logs telemetry

STUDENT_ID="${STUDENT_ID:-starter}"
N_SEEDS="${N_SEEDS:-10}"
JOB="${SLURM_JOB_ID:-local}"

OUT_ADAMW="logs/winner_adamw_${JOB}"
OUT_SPEC="logs/winner_spec_cs1.0_${JOB}"

echo "[winner_resample] STUDENT_ID=${STUDENT_ID}  N_SEEDS=${N_SEEDS}"

# ----- Arm A: AdamW at its best config -----
echo "[winner_resample] === arm A: AdamW @ (lr=3e-3, wd=1e-2, eps=1e-8) ==="
python experiments/run_grid.py \
    --task task_b --kappas 1e3 \
    --optimizers adamw \
    --grid-mode smoke \
    --smoke-lr 3e-3 --smoke-wd 1e-2 --smoke-eps 1e-8 \
    --all-seeds --n-seeds "${N_SEEDS}" \
    --student-id "${STUDENT_ID}" \
    --max-steps 2000 --batch-size 256 \
    --out-dir "${OUT_ADAMW}"

# ----- Arm B: SpectralOptimizer alpha=1.0 at its best config -----
echo "[winner_resample] === arm B: Spec a=1.0 @ (lr=3e-3, wd=0, eps=1e-8) ==="
python experiments/run_grid.py \
    --task task_b --kappas 1e3 \
    --optimizers spectral --mode production \
    --spectral-rank 8 --spectral-beta 0.95 \
    --correction-strength 1.0 \
    --grid-mode smoke \
    --smoke-lr 3e-3 --smoke-wd 0.0 --smoke-eps 1e-8 \
    --all-seeds --n-seeds "${N_SEEDS}" \
    --student-id "${STUDENT_ID}" \
    --max-steps 2000 --batch-size 256 \
    --out-dir "${OUT_SPEC}"

echo "[winner_resample] done.  CSVs:"
echo "    ${OUT_ADAMW}/grid_task_b_smoke.csv"
echo "    ${OUT_SPEC}/grid_task_b_smoke.csv"
