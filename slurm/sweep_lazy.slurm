#!/usr/bin/env bash
#
# sweep_lazy.slurm -- steering_interval ablation (cost-vs-quality).
#
# At alpha=1.0, beta=0.95, r=8, kappa=10^3, lr=3e-3, wd=0, eps=1e-8:
# sweep steering_interval K over {1, 5, 10, 20} with 3 seeds.  When K>1
# the spectral correction (basis projection + EMA + rotation) only fires
# every Kth step; intermediate steps are pure AdamW.  This addresses the
# 1.8x wall-clock penalty without requiring Triton fusion.
#
# Total: 4 * 3 = 12 runs at decreasing per-run cost.  ~25 min total.
#
# Expected reading:
#     - K=1  : control (the current null result)
#     - K=5  : 5x less spectral compute, ~50% less correction signal
#     - K=10 : ~80% less spectral compute, EMA sees 1/10 samples
#     - K=20 : ~95% less spectral compute, near-AdamW
# A flat loss curve across K confirms the spectral correction is not
# load-bearing per-step; a U-shape would say "occasional steering helps,
# every-step steering hurts."
#
# Submit:
#     STUDENT_ID=sdh4py sbatch slurm/sweep_lazy.slurm
#
#SBATCH --job-name=spec_sweep_lazy
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/sweep_lazy_%j.out
#SBATCH --error=slurm/logs/sweep_lazy_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[sweep_lazy] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

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
INTERVALS="${INTERVALS:-1 5 10 20}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-0.95}"
RANK="${RANK:-8}"

echo "[sweep_lazy] STUDENT_ID=${STUDENT_ID}  N_SEEDS=${N_SEEDS}"
echo "[sweep_lazy] alpha=${ALPHA}  beta=${BETA}  rank=${RANK}"
echo "[sweep_lazy] intervals=${INTERVALS}"

for K in ${INTERVALS}; do
    OUT="logs/sweep_lazy_${JOB}/interval_${K}"
    SUFFIX="_K${K}"
    echo "[sweep_lazy] === steering_interval = ${K} ==="
    python experiments/run_grid.py \
        --task task_b --kappas 1e3 \
        --optimizers spectral --mode production \
        --spectral-rank "${RANK}" --spectral-beta "${BETA}" \
        --correction-strength "${ALPHA}" \
        --steering-interval "${K}" \
        --grid-mode smoke \
        --smoke-lr 3e-3 --smoke-wd 0.0 --smoke-eps 1e-8 \
        --all-seeds --n-seeds "${N_SEEDS}" \
        --student-id "${STUDENT_ID}" \
        --max-steps 2000 --batch-size 256 \
        --out-dir "${OUT}" \
        --telemetry-suffix "${SUFFIX}"
done

echo "[sweep_lazy] done.  Compare per-step ms across K via ms_per_step in CSV."
