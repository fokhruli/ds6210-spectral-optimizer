"""Run your SpectralOptimizer on Task B (kappa = 1000) with seed[0].

By default this invokes ``mode="debug"`` so you can inspect the
diagnostics without your `production` branch crashing the run.  Switch
``mode`` to ``"production"`` once you have implemented
``_compute_spectral_correction``.

Compare ``my_optimizer_telemetry.json`` against
``adamw_baseline_telemetry.json`` (same seed, same task) to validate
both quality (final loss) and per-step CUDA time.
"""
from __future__ import annotations

import sys

import torch

torch.set_float32_matmul_precision("high")

sys.path.insert(0, ".")
from harness import ResidualMLP, derive_seeds, make_task_b, train  # noqa: E402
from my_optimizer import SpectralOptimizer  # noqa: E402

STUDENT_ID = "starter"  # replace with your student id when you copy this file
MODE = "debug"          # "disabled" | "debug" | "production"


def main() -> None:
    seeds = derive_seeds(STUDENT_ID)
    seed = seeds[0]
    torch.manual_seed(seed)

    X, Y = make_task_b(kappa=1e3, n_samples=20_000, d=256, seed=seed)
    model = ResidualMLP(in_dim=256, width=256, depth=24, out_dim=10).cuda()

    optimizer = SpectralOptimizer(
        model.parameters(),
        lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4,
        mode=MODE,
        spectral_rank=8,
        spectral_beta=0.95,
        correction_strength=0.5,
    )

    log, timer = train(model, optimizer, X, Y, n_steps=2_000, batch_size=256)
    report = timer.sync_and_dump("my_optimizer_telemetry.json")
    diag_report = optimizer.dump_diagnostics("my_optimizer_diagnostics.json")

    print(f"student_id   = {STUDENT_ID!r}  (seeds = {seeds})")
    print(f"seed used    = {seed}")
    print(f"mode         = {MODE!r}")
    print(f"initial loss = {log[0][1]:.4f}")
    print(f"final loss   = {log[-1][1]:.4f}")
    fwd = report["summary"]["forward"]["mean_ms"]
    bwd = report["summary"]["backward"]["mean_ms"]
    opt = report["summary"]["optimizer_step"]["mean_ms"]
    print(f"per-step ms  fwd {fwd:.3f}  bwd {bwd:.3f}  opt {opt:.3f}")
    if optimizer.last_diagnostics:
        print("last_diagnostics:")
        for k, v in optimizer.last_diagnostics.items():
            print(f"  {k}: {v:.4f}")
    summ = diag_report["summary"]
    print(f"diagnostic history: {diag_report['n_logged_steps']} steps logged")
    print(f"  mean_effective_rank        = {summ['mean_effective_rank']:.4f}")
    print(f"  mean_cosine_with_adamw     = {summ['mean_cosine_with_adamw']:.4f}")
    print(f"  mean_correction_norm       = {summ['mean_spectral_correction_norm']:.4e}")


if __name__ == "__main__":
    main()
