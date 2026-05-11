"""Proves that ``SpectralOptimizer(mode='disabled')`` matches AdamW.

Trains two identical models for 100 steps -- one with
``torch.optim.AdamW(foreach=True)``, one with ``SpectralOptimizer``
in disabled mode -- and reports the maximum absolute difference between
their final parameter tensors.

Pass criterion (rubric): max-abs diff < 1e-6.

If this fails after you edit ``my_optimizer.py``, your ``disabled``
branch is no longer faithful to AdamW, which means ALL of your spectral
results are off by an unknown floor.  Fix this before continuing.
"""
from __future__ import annotations

import copy
import sys

import torch

torch.set_float32_matmul_precision("high")

sys.path.insert(0, ".")
from harness import ResidualMLP, batch_slice, make_task_b  # noqa: E402
from my_optimizer import SpectralOptimizer  # noqa: E402


N_STEPS = 100
BATCH_SIZE = 256
LR = 3e-3
WD = 1e-4
EPS = 1e-8
BETAS = (0.9, 0.95)
SEED = 12345


def _build_model() -> torch.nn.Module:
    torch.manual_seed(SEED)
    return ResidualMLP(in_dim=256, width=256, depth=8, out_dim=10).cuda()


def _train(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
           X: torch.Tensor, Y: torch.Tensor) -> None:
    model.train()
    loss_fn = torch.nn.functional.mse_loss
    for step in range(N_STEPS):
        xb, yb = batch_slice(X, Y, BATCH_SIZE, step)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Parity check requires CUDA.")

    X, Y = make_task_b(kappa=1e3, n_samples=4_096, d=256, seed=SEED)

    # Reference: torch.optim.AdamW(foreach=True).
    ref_model = _build_model()
    ref_opt = torch.optim.AdamW(
        ref_model.parameters(),
        lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, foreach=True,
    )
    _train(ref_model, ref_opt, X, Y)

    # Candidate: SpectralOptimizer(mode='disabled') from THE SAME init.
    cand_model = _build_model()
    cand_opt = SpectralOptimizer(
        cand_model.parameters(),
        lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, mode="disabled",
    )
    _train(cand_model, cand_opt, X, Y)

    # Compare final parameters tensor-by-tensor.
    max_diff = 0.0
    for (n_ref, p_ref), (n_cand, p_cand) in zip(
        ref_model.named_parameters(), cand_model.named_parameters(), strict=True,
    ):
        assert n_ref == n_cand, f"name mismatch: {n_ref} != {n_cand}"
        diff = (p_ref.detach() - p_cand.detach()).abs().max().item()
        if diff > max_diff:
            max_diff = diff

    tol = 1e-6
    status = "PASS" if max_diff < tol else "FAIL"
    print(f"max |theta_AdamW - theta_disabled|  = {max_diff:.3e}  ({status}, tol={tol:.0e})")
    if max_diff >= tol:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
