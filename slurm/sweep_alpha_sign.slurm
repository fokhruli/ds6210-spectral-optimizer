#!/usr/bin/env bash
#
# sweep_alpha_sign.slurm -- Negative-alpha (anti-Polyak) ablation.
#
# At the apparently-winning Spec config (lr=3e-3, wd=0, eps=1e-8,
# r=8, beta=0.95, kappa=10^3), sweep correction_strength over
#     {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0}
# with 3 SHA-derived seeds each.  Total: 7 * 3 = 21 runs.
# Per-run cost ~1.5 min on A6000 -> ~30-45 min total.
#
# Diagnostics: alpha < 0 steers AWAY from the EMA.  Geometric
# motivation: cos theta_t ~ 0.99919 in the +alpha winner means the
# EMA is tracking c_t closely, so positive steering is small in
# magnitude.  Negative alpha applies the same magnitude in the
# anti-history direction -- candidate for breaking out of zig-zag
# patterns in highly anisotropic valleys.
#
# Submit:
#     STUDENT_ID=sdh4py sbatch slurm/sweep_alpha_sign.slurm
#
#SBATCH --job-name=spec_sweep_alpha
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/sweep_alpha_%j.out
#SBATCH --error=slurm/logs/sweep_alpha_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[sweep_alpha] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

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
ALPHAS="${ALPHAS:--2.0 -1.0 -0.5 0.0 0.5 1.0 2.0}"

echo "[sweep_alpha] STUDENT_ID=${STUDENT_ID}  N_SEEDS=${N_SEEDS}"
echo "[sweep_alpha] alphas=${ALPHAS}"

for ALPHA in ${ALPHAS}; do
    OUT="logs/sweep_alpha_${JOB}/alpha_${ALPHA}"
    SUFFIX="_alpha${ALPHA}"
    echo "[sweep_alpha] === alpha = ${ALPHA} ==="
    python experiments/run_grid.py \
        --task task_b --kappas 1e3 \
        --optimizers spectral --mode production \
        --spectral-rank 8 --spectral-beta 0.95 \
        --correction-strength "${ALPHA}" \
        --grid-mode smoke \
        --smoke-lr 3e-3 --smoke-wd 0.0 --smoke-eps 1e-8 \
        --all-seeds --n-seeds "${N_SEEDS}" \
        --student-id "${STUDENT_ID}" \
        --max-steps 2000 --batch-size 256 \
        --out-dir "${OUT}" \
        --telemetry-suffix "${SUFFIX}"
done

echo "[sweep_alpha] done.  Aggregate with experiments/sweep_summary.py."
