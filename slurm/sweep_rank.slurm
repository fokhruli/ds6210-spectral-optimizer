#!/usr/bin/env bash
#
# sweep_rank.slurm -- spectral_rank ablation at the failed winner config.
#
# At alpha=1.0, beta=0.95, kappa=10^3, lr=3e-3, wd=0, eps=1e-8: sweep
# spectral_rank over {2, 4, 8, 16, 32} with 3 seeds.  5 * 3 = 15 runs.
# Per-run cost grows mildly with rank; ~1.5-3 min/run -> ~30-45 min.
#
# Question: was r=8 the bottleneck?
#     - If loss is monotone in r and improves with larger r: the
#       random-subspace correction works in principle but needs more
#       expressive capacity than r=8 provides.
#     - If loss is flat across r: the random subspace itself is the
#       failure mode; rank doesn't help.  Strengthens the negative-result
#       conclusion and motivates the streaming-PCA future work.
#
# Submit:
#     STUDENT_ID=sdh4py sbatch slurm/sweep_rank.slurm
#
#SBATCH --job-name=spec_sweep_rank
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/sweep_rank_%j.out
#SBATCH --error=slurm/logs/sweep_rank_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[sweep_rank] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

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
RANKS="${RANKS:-2 4 8 16 32}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-0.95}"

echo "[sweep_rank] STUDENT_ID=${STUDENT_ID}  N_SEEDS=${N_SEEDS}"
echo "[sweep_rank] alpha=${ALPHA}  beta=${BETA}  ranks=${RANKS}"

for R in ${RANKS}; do
    OUT="logs/sweep_rank_${JOB}/rank_${R}"
    SUFFIX="_rank${R}"
    echo "[sweep_rank] === rank = ${R} ==="
    python experiments/run_grid.py \
        --task task_b --kappas 1e3 \
        --optimizers spectral --mode production \
        --spectral-rank "${R}" --spectral-beta "${BETA}" \
        --correction-strength "${ALPHA}" \
        --grid-mode smoke \
        --smoke-lr 3e-3 --smoke-wd 0.0 --smoke-eps 1e-8 \
        --all-seeds --n-seeds "${N_SEEDS}" \
        --student-id "${STUDENT_ID}" \
        --max-steps 2000 --batch-size 256 \
        --out-dir "${OUT}" \
        --telemetry-suffix "${SUFFIX}"
done

echo "[sweep_rank] done."
