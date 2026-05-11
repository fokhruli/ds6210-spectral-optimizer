#!/usr/bin/env bash
#
# sweep_beta.slurm -- EMA-decay ablation at the failed winner config.
#
# At alpha=1.0, kappa=10^3, lr=3e-3, wd=0, eps=1e-8, r=8: sweep the
# spectral EMA decay beta_s over {0.5, 0.9, 0.95, 0.99} with 3 seeds.
# 4 * 3 = 12 runs at ~1.5 min/run -> ~20 min total.
#
# Question: does the EMA memory length actually control performance?
#     - If all betas land within seed noise: write
#       "the EMA is not the operative mechanism."
#     - If loss is U-shaped or monotone in beta_s: the EMA mechanism
#       is real (and we have a clean ablation curve for the report).
#
# Submit:
#     STUDENT_ID=sdh4py sbatch slurm/sweep_beta.slurm
#
#SBATCH --job-name=spec_sweep_beta
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/sweep_beta_%j.out
#SBATCH --error=slurm/logs/sweep_beta_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[sweep_beta] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

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
BETAS="${BETAS:-0.5 0.9 0.95 0.99}"
ALPHA="${ALPHA:-1.0}"

echo "[sweep_beta] STUDENT_ID=${STUDENT_ID}  N_SEEDS=${N_SEEDS}  alpha=${ALPHA}"
echo "[sweep_beta] betas=${BETAS}"

for BETA in ${BETAS}; do
    OUT="logs/sweep_beta_${JOB}/beta_${BETA}"
    SUFFIX="_beta${BETA}"
    echo "[sweep_beta] === beta_s = ${BETA} ==="
    python experiments/run_grid.py \
        --task task_b --kappas 1e3 \
        --optimizers spectral --mode production \
        --spectral-rank 8 --spectral-beta "${BETA}" \
        --correction-strength "${ALPHA}" \
        --grid-mode smoke \
        --smoke-lr 3e-3 --smoke-wd 0.0 --smoke-eps 1e-8 \
        --all-seeds --n-seeds "${N_SEEDS}" \
        --student-id "${STUDENT_ID}" \
        --max-steps 2000 --batch-size 256 \
        --out-dir "${OUT}" \
        --telemetry-suffix "${SUFFIX}"
done

echo "[sweep_beta] done."
