#!/usr/bin/env bash
#
# sweep_oja_lr.slurm -- ablation over the Oja basis-adaptation rate.
#
# At basis_adapt=oja, alpha=1.0, beta=0.95, r=8, K=1, kappa=10^3,
# (lr=3e-3, wd=0, eps=1e-8): sweep oja_lr over {1e-4, 1e-3, 1e-2, 1e-1}
# with 3 SHA-derived seeds each.  4 * 3 = 12 runs.  ~20-25 min on A6000.
#
# Question: where in the basis-adaptation-rate spectrum does the
# adapted basis recover a signal vs frozen, if at all?
#     - oja_lr -> 0 : equivalent to frozen, recovers the null result.
#     - oja_lr small : slow adaptation, may take longer than 2000 steps
#       to align with principal directions.
#     - oja_lr moderate : sweet spot.
#     - oja_lr large : Oja overshoots, basis becomes noisy.
#
# Runs second.  After winner_oja_resample.slurm tells you whether
# Oja flips the headline at the default oja_lr=1e-2, this sweep
# locates the optimal rate.
#
# Submit:
#     STUDENT_ID=sdh4py sbatch slurm/sweep_oja_lr.slurm
#
#SBATCH --job-name=spec_sweep_oja_lr
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/sweep_oja_lr_%j.out
#SBATCH --error=slurm/logs/sweep_oja_lr_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[sweep_oja_lr] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

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
OJA_LRS="${OJA_LRS:-1e-4 1e-3 1e-2 1e-1}"
OJA_ORTH_INTERVAL="${OJA_ORTH_INTERVAL:-50}"
ALPHA="${ALPHA:-1.0}"

echo "[sweep_oja_lr] STUDENT_ID=${STUDENT_ID}  N_SEEDS=${N_SEEDS}"
echo "[sweep_oja_lr] alpha=${ALPHA}  K_orth=${OJA_ORTH_INTERVAL}"
echo "[sweep_oja_lr] oja_lrs=${OJA_LRS}"

for LR_OJA in ${OJA_LRS}; do
    OUT="logs/sweep_oja_lr_${JOB}/lr_${LR_OJA}"
    SUFFIX="_ojaLR${LR_OJA}"
    echo "[sweep_oja_lr] === oja_lr = ${LR_OJA} ==="
    python experiments/run_grid.py \
        --task task_b --kappas 1e3 \
        --optimizers spectral --mode production \
        --spectral-rank 8 --spectral-beta 0.95 \
        --correction-strength "${ALPHA}" \
        --basis-adapt oja \
        --oja-lr "${LR_OJA}" \
        --oja-orth-interval "${OJA_ORTH_INTERVAL}" \
        --grid-mode smoke \
        --smoke-lr 3e-3 --smoke-wd 0.0 --smoke-eps 1e-8 \
        --all-seeds --n-seeds "${N_SEEDS}" \
        --student-id "${STUDENT_ID}" \
        --max-steps 2000 --batch-size 256 \
        --out-dir "${OUT}" \
        --telemetry-suffix "${SUFFIX}"
done

echo "[sweep_oja_lr] done.  Summarize with sweep_summary.py:"
echo "  python experiments/sweep_summary.py logs/sweep_oja_lr_${JOB}"
