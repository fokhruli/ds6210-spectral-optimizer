#!/usr/bin/env bash
#
# winner_oja_resample.slurm -- streaming-PCA basis vs frozen random,
# 10-seed paired head-to-head at the apparently-winning Spec config.
#
# Hypothesis: the n=10 null at frozen-random B (Section 5.5) leaves
# the door open that an *adapted* basis -- one that tracks the
# principal directions of the AdamW step trajectory -- could recover
# a real signal.  Oja's rule with periodic QR is the cheapest test:
# same code path, single new flag.
#
# Two arms, both at (kappa=10^3, lr=3e-3, wd=0, eps=1e-8, alpha=1.0,
# beta=0.95, r=8, K=1):
#   (A) basis_adapt=frozen     (the existing null result, 10 seeds).
#   (B) basis_adapt=oja        (oja_lr=1e-2, oja_orth_interval=50).
#
# Total: 20 runs of 2000 steps.  ~25-35 min on A6000.
#
# Submit:
#     STUDENT_ID=sdh4py sbatch slurm/winner_oja_resample.slurm
#
# Knobs:
#     N_SEEDS              (default 10)
#     OJA_LR               (default 1e-2)
#     OJA_ORTH_INTERVAL    (default 50)
#
#SBATCH --job-name=spec_winner_oja
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/winner_oja_%j.out
#SBATCH --error=slurm/logs/winner_oja_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[winner_oja] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

module purge || true
module load miniforge || true
VENV="${SPECTRAL_VENV:-/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -V

mkdir -p slurm/logs logs telemetry

STUDENT_ID="${STUDENT_ID:-starter}"
N_SEEDS="${N_SEEDS:-10}"
OJA_LR="${OJA_LR:-1e-2}"
OJA_ORTH_INTERVAL="${OJA_ORTH_INTERVAL:-50}"
JOB="${SLURM_JOB_ID:-local}"

OUT_FROZEN="logs/winner_oja_frozen_${JOB}"
OUT_OJA="logs/winner_oja_oja_${JOB}"

echo "[winner_oja] STUDENT_ID=${STUDENT_ID}  N_SEEDS=${N_SEEDS}"
echo "[winner_oja] OJA_LR=${OJA_LR}  OJA_ORTH_INTERVAL=${OJA_ORTH_INTERVAL}"

# ----- Arm A: Spec @ frozen random basis (the existing null) -----
echo "[winner_oja] === Arm A: frozen basis ==="
python experiments/run_grid.py \
    --task task_b --kappas 1e3 \
    --optimizers spectral --mode production \
    --spectral-rank 8 --spectral-beta 0.95 \
    --correction-strength 1.0 \
    --basis-adapt frozen \
    --grid-mode smoke \
    --smoke-lr 3e-3 --smoke-wd 0.0 --smoke-eps 1e-8 \
    --all-seeds --n-seeds "${N_SEEDS}" \
    --student-id "${STUDENT_ID}" \
    --max-steps 2000 --batch-size 256 \
    --out-dir "${OUT_FROZEN}" \
    --telemetry-suffix "_basisFrozen"

# ----- Arm B: Spec @ Oja-adapted basis -----
echo "[winner_oja] === Arm B: Oja-adapted basis  (lr=${OJA_LR}, K_orth=${OJA_ORTH_INTERVAL}) ==="
python experiments/run_grid.py \
    --task task_b --kappas 1e3 \
    --optimizers spectral --mode production \
    --spectral-rank 8 --spectral-beta 0.95 \
    --correction-strength 1.0 \
    --basis-adapt oja \
    --oja-lr "${OJA_LR}" \
    --oja-orth-interval "${OJA_ORTH_INTERVAL}" \
    --grid-mode smoke \
    --smoke-lr 3e-3 --smoke-wd 0.0 --smoke-eps 1e-8 \
    --all-seeds --n-seeds "${N_SEEDS}" \
    --student-id "${STUDENT_ID}" \
    --max-steps 2000 --batch-size 256 \
    --out-dir "${OUT_OJA}" \
    --telemetry-suffix "_basisOja_lr${OJA_LR}"

echo "[winner_oja] done.  Aggregate with experiments/winner_significance.py"
echo "  --adamw-csv ${OUT_FROZEN}/grid_task_b_smoke.csv"
echo "  --spec-csv  ${OUT_OJA}/grid_task_b_smoke.csv"
echo "(headers: arm A is frozen-spec acting as 'baseline', arm B is Oja-spec)"
