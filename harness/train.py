"""Minimal training loop for benchmarking spectral optimizers.

The loop is intentionally spartan: everything you don't see here -- LR
schedules, gradient accumulation, mixed-precision autocast, tuning
sweeps -- is YOUR responsibility to add in a wrapping script.  The
harness only provides what must be identical across all students.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .data import batch_slice
from .timing import CUDAEventTimer


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    n_steps: int,
    batch_size: int,
    loss_fn: Callable = F.mse_loss,
    log_every: int = 50,
) -> tuple[list[tuple[int, float]], CUDAEventTimer]:
    """Trains ``model`` on VRAM-resident ``(X, Y)``.

    Returns ``(log, timer)``.  ``log`` is a list of ``(step, loss_value)``
    pairs sampled every ``log_every`` steps.  The timer records
    forward/backward/step events via ``CUDAEventTimer``.
    """
    log = []
    timer = CUDAEventTimer()
    model.train()

    for step in range(n_steps):
        xb, yb = batch_slice(X, Y, batch_size, step)

        timer.start("forward")
        yp = model(xb)
        loss = loss_fn(yp, yb)
        timer.end("forward", step=step)

        timer.start("backward")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        timer.end("backward", step=step)

        timer.start("optimizer_step")
        optimizer.step()
        timer.end("optimizer_step", step=step)

        if step % log_every == 0:
            # Synchronous .item() at the log cadence is fine; it is
            # not inside the compiled hot path.
            log.append((step, float(loss.detach().item())))

    return log, timer
