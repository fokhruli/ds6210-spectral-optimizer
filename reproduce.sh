#!/usr/bin/env bash
#
# reproduce.sh -- end-to-end smoke for the DS 6210 spectral starter.
#
# Designed to fit in a short GPU session (~10-20 minutes on an A100).
# Replace STUDENT_ID with your own when running for-real numbers; the
# default "starter" id derives a deterministic SHA-256 seed triple just
# for the smoke run.

set -euo pipefail

STUDENT_ID="${STUDENT_ID:-starter}"

echo "=== Environment diagnostic ==="
python -V
python - <<'PY'
import torch
try:
    import torchvision
    tv = torchvision.__version__
except Exception as exc:
    tv = f"<unimportable: {exc}>"
print(f"  torch={torch.__version__}  cuda={torch.version.cuda}  torchvision={tv}")
print(f"  cuda_available={torch.cuda.is_available()}")
PY

mkdir -p logs telemetry

echo
echo "=== Print SHA-derived seeds for student_id=${STUDENT_ID} ==="
python -m harness.seeds "${STUDENT_ID}"

echo
echo "=== Sanity 1: AdamW(foreach=True) baseline on Task B (kappa=1e3) ==="
python experiments/run_adamw_baseline.py

echo
echo "=== Sanity 2: parity test -- mode='disabled' must match AdamW ==="
python experiments/parity_check.py

echo
echo "=== Sanity 3: your optimizer (mode='debug') on Task B (kappa=1e3) ==="
python experiments/run_my_optimizer.py

echo
echo "=== Smoke grid: AdamW on Task B, kappa=1e3, single seed ==="
python experiments/run_grid.py \
  --task task_b --kappas 1e3 \
  --optimizers adamw \
  --grid-mode smoke \
  --student-id "${STUDENT_ID}" \
  --max-steps 1000

cat <<MSG

For the full course grid (7 lr x 3 wd x 2 eps = 42 configs per optimizer
per kappa) on Task B across both kappas and all 3 SHA-derived seeds, run:

    python experiments/run_grid.py \\
      --task task_b --kappas 1e3 1e6 \\
      --optimizers adamw spectral \\
      --grid-mode full --all-seeds \\
      --student-id ${STUDENT_ID}

Once your production mode is implemented, swap '--mode production' into
the spectral arm of the same call.

MSG
