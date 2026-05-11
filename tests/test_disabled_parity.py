"""The one mandatory contract: ``mode='disabled'`` matches AdamW.

Skipped if no GPU is available.  Run with::

    pytest tests/test_disabled_parity.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import ResidualMLP, batch_slice, make_task_b  # noqa: E402
from my_optimizer import SpectralOptimizer  # noqa: E402


SEED = 12345
N_STEPS = 50
BATCH_SIZE = 256
LR = 3e-3
WD = 1e-4
EPS = 1e-8
BETAS = (0.9, 0.95)


def _make_model() -> torch.nn.Module:
    torch.manual_seed(SEED)
    return ResidualMLP(in_dim=256, width=256, depth=4, out_dim=10).cuda()


def _train(model: torch.nn.Module, opt: torch.optim.Optimizer,
           X: torch.Tensor, Y: torch.Tensor) -> None:
    loss_fn = torch.nn.functional.mse_loss
    model.train()
    for step in range(N_STEPS):
        xb, yb = batch_slice(X, Y, BATCH_SIZE, step)
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_disabled_mode_matches_adamw() -> None:
    """SpectralOptimizer(mode='disabled') must trace torch.optim.AdamW exactly.

    Tolerance 1e-6 max-abs over 50 steps on a small Task B run.
    """
    X, Y = make_task_b(kappa=1e3, n_samples=4_096, d=256, seed=SEED)

    ref_model = _make_model()
    ref_opt = torch.optim.AdamW(
        ref_model.parameters(),
        lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, foreach=True,
    )
    _train(ref_model, ref_opt, X, Y)

    cand_model = _make_model()
    cand_opt = SpectralOptimizer(
        cand_model.parameters(),
        lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, mode="disabled",
    )
    _train(cand_model, cand_opt, X, Y)

    max_diff = 0.0
    for (n_r, p_r), (n_c, p_c) in zip(
        ref_model.named_parameters(), cand_model.named_parameters(), strict=True,
    ):
        assert n_r == n_c, f"name mismatch: {n_r} != {n_c}"
        max_diff = max(max_diff,
                       (p_r.detach() - p_c.detach()).abs().max().item())

    assert max_diff < 1e-6, (
        f"disabled mode diverged from torch.optim.AdamW "
        f"(max-abs diff {max_diff:.3e}); your starter is no longer faithful."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_production_mode_raises_until_implemented() -> None:
    """Out of the box, production mode raises NotImplementedError."""
    p = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
    opt = SpectralOptimizer([p], lr=1e-3, mode="production")
    p.grad = torch.randn_like(p)
    with pytest.raises(NotImplementedError):
        opt.step()
