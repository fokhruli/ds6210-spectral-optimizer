#!/usr/bin/env bash
#
# grid_baseline.slurm -- AdamW(foreach=True) full course grid on Task B.
#
# 7 lr x 3 wd x 2 eps = 42 configs, kappas {1e3, 1e6}, 3 SHA-derived
# seeds = 252 runs.  ~2-4 h on a single A100.
#
# Submit with:
#     STUDENT_ID=<your_id> sbatch slurm/grid_baseline.slurm
#
#SBATCH --job-name=spec_grid_adamw
#SBATCH --account=ds6210
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/grid_adamw_%j.out
#SBATCH --error=slurm/logs/grid_adamw_%j.err

set -euo pipefail

PROJECT_ROOT="/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/project/spectral_harness"
cd "${PROJECT_ROOT}"
echo "[grid_adamw] cwd=$(pwd)  job=${SLURM_JOB_ID:-local}  host=$(hostname)"

module purge || true
module load miniforge || true
VENV="${SPECTRAL_VENV:-/home/sdh4py/UVA_projects/assignemnt/Neumerical_optim/ds6210/venv}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -V

mkdir -p slurm/logs logs telemetry

STUDENT_ID="${STUDENT_ID:-starter}"
echo "[grid_adamw] STUDENT_ID=${STUDENT_ID}"

# Dry-run print of the command before launching (per insights guidance).
CMD=(python experiments/run_grid.py
     --task task_b
     --kappas 1e3 1e6
     --optimizers adamw
     --grid-mode full
     --all-seeds
     --student-id "${STUDENT_ID}"
     --max-steps 2000
     --batch-size 256
     --out-dir "logs/grid_adamw_${SLURM_JOB_ID:-local}")
echo "[grid_adamw] running: ${CMD[*]}"
"${CMD[@]}"
