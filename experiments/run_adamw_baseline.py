"""Sanity check: AdamW on Task B (kappa = 1000) with the canonical config.

Run this BEFORE editing my_optimizer.py to confirm the harness is wired
to your GPU correctly.  The canonical baseline is
``torch.optim.AdamW(foreach=True)``.  Do NOT benchmark against
``fused=True``: that variant calls a hand-written CUDA kernel and is
not the apples-to-apples comparison expected by the rubric.
"""
from __future__ import annotations

import sys

import torch

torch.set_float32_matmul_precision("high")

sys.path.insert(0, ".")
from harness import ResidualMLP, derive_seeds, make_task_b, train  # noqa: E402

STUDENT_ID = "starter"  # replace with your student id when you copy this file


def main() -> None:
    seeds = derive_seeds(STUDENT_ID)
    seed = seeds[0]
    torch.manual_seed(seed)

    X, Y = make_task_b(kappa=1e3, n_samples=20_000, d=256, seed=seed)
    model = ResidualMLP(in_dim=256, width=256, depth=24, out_dim=10).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4,
        foreach=True,
    )

    log, timer = train(model, optimizer, X, Y, n_steps=2_000, batch_size=256)
    report = timer.sync_and_dump("adamw_baseline_telemetry.json")

    print(f"student_id   = {STUDENT_ID!r}  (seeds = {seeds})")
    print(f"seed used    = {seed}")
    print(f"initial loss = {log[0][1]:.4f}")
    print(f"final loss   = {log[-1][1]:.4f}")
    fwd = report["summary"]["forward"]["mean_ms"]
    bwd = report["summary"]["backward"]["mean_ms"]
    opt = report["summary"]["optimizer_step"]["mean_ms"]
    print(f"per-step ms  fwd {fwd:.3f}  bwd {bwd:.3f}  opt {opt:.3f}")


if __name__ == "__main__":
    main()
