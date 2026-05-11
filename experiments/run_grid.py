"""Course-grid runner: 7 lr x 3 wd x 2 eps = 42 configs per (optimizer, task).

Single-machine, no SLURM.  One CSV per call, columns:
    regime, optimizer, seed, lr, wd, eps, step, loss, ms

Examples:

    # Smoke (one config, one seed, ~2 minutes on A100):
    python experiments/run_grid.py --task task_b --kappas 1e3 \\
        --optimizers adamw --grid-mode smoke

    # Full course grid on Task B (kappa = 1e3, 1e6) for AdamW + your method,
    # all 3 SHA-derived seeds:
    python experiments/run_grid.py --task task_b --kappas 1e3 1e6 \\
        --optimizers adamw spectral --grid-mode full --all-seeds

The grid runs sequentially.  Use --max-steps and --early-abort-frac to
cap the cost of bad configurations during exploratory tuning.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

sys.path.insert(0, ".")
from harness import (  # noqa: E402
    CUDAEventTimer,
    ResidualMLP,
    batch_slice,
    derive_seeds,
    make_task_a,
    make_task_b,
    make_task_c_teacher,
    sample_task_c,
)
from my_optimizer import SpectralOptimizer  # noqa: E402


# 7 lr x 3 wd x 2 eps = 42 configs (the course grid in §1 of the handout).
LR_GRID = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1]
WD_GRID = [0.0, 1.0e-4, 1.0e-2]
EPS_GRID = [1.0e-8, 1.0e-4]


def _scheduled_lr(base_lr: float, step: int, n_steps: int) -> float:
    """Cosine schedule with 5% linear warmup, decaying to 10% of base_lr."""
    warmup = max(1, int(0.05 * n_steps))
    if step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    denom = max(1, n_steps - warmup)
    progress = min(1.0, float(step - warmup) / float(denom))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cosine)


def _build_task(task: str, value: float | None, seed: int):
    """Returns (X, Y, model, loss_fn, regime_label)."""
    if task == "task_a":
        X, Y = make_task_a()
        model = ResidualMLP(in_dim=784, width=512, depth=16, out_dim=10).cuda()
        return X, Y, model, F.cross_entropy, "task_a_fashion_mnist"
    if task == "task_b":
        X, Y = make_task_b(kappa=value, n_samples=20_000, d=256, seed=seed)
        model = ResidualMLP(in_dim=256, width=256, depth=24, out_dim=10).cuda()
        return X, Y, model, F.mse_loss, f"task_b_kappa_{value:g}"
    if task == "task_c":
        teacher = make_task_c_teacher(alpha=value, seed=seed)
        X, Y = sample_task_c(teacher, n_samples=20_000, seed=seed)
        model = ResidualMLP(in_dim=500, width=128, depth=8, out_dim=1).cuda()
        return X, Y, model, F.mse_loss, f"task_c_alpha_{value:g}"
    raise ValueError(f"unknown task: {task}")


def _build_optimizer(name: str, model: torch.nn.Module,
                     lr: float, wd: float, eps: float,
                     args: argparse.Namespace) -> torch.optim.Optimizer:
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr, betas=(0.9, 0.95), eps=eps, weight_decay=wd,
            foreach=True,
        )
    if name == "spectral":
        return SpectralOptimizer(
            model.parameters(),
            lr=lr, betas=(0.9, 0.95), eps=eps, weight_decay=wd,
            mode=args.mode,
            spectral_rank=args.spectral_rank,
            spectral_beta=args.spectral_beta,
            correction_strength=args.correction_strength,
            steering_interval=args.steering_interval,
            basis_adapt=args.basis_adapt,
            oja_lr=args.oja_lr,
            oja_orth_interval=args.oja_orth_interval,
        )
    raise ValueError(f"unknown optimizer: {name}")


def _train_one(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               X: torch.Tensor, Y: torch.Tensor, loss_fn: Callable,
               base_lr: float, args: argparse.Namespace,
               ) -> tuple[list[dict], dict, str]:
    timer = CUDAEventTimer()
    rows: list[dict] = []
    initial_loss: float | None = None
    status = "ok"

    model.train()
    for step in range(args.max_steps):
        lr = _scheduled_lr(base_lr, step, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = batch_slice(X, Y, args.batch_size, step)

        timer.start("forward")
        pred = model(xb)
        loss = loss_fn(pred, yb)
        timer.end("forward", step=step)

        timer.start("backward")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        timer.end("backward", step=step)

        timer.start("optimizer_step")
        optimizer.step()
        timer.end("optimizer_step", step=step)

        if step % args.log_every == 0 or step == args.max_steps - 1:
            v = float(loss.detach().item())
            if initial_loss is None:
                initial_loss = v
            rows.append({"step": step, "loss": v, "lr_scheduled": lr})
            if not math.isfinite(v):
                status = "nonfinite"
                break
            if (step >= args.early_abort_step and initial_loss is not None
                    and v > args.early_abort_frac * initial_loss):
                status = "early_abort"
                break

    report = timer.sync_and_dump(args.current_telemetry_path)
    # Roll cumulative ms from the timer back into the CSV rows.
    cum: dict[int, float] = {}
    running = 0.0
    by_step: dict[int, float] = {}
    for r in report["rows"]:
        s = r["step"]
        if s is None:
            continue
        by_step[s] = by_step.get(s, 0.0) + float(r["ms"])
    for s in sorted(by_step):
        running += by_step[s]
        cum[s] = running
    for r in rows:
        r["ms"] = cum.get(int(r["step"]), "")
    return rows, report, status


def _grid(args: argparse.Namespace):
    if args.grid_mode == "smoke":
        return [(args.smoke_lr, args.smoke_wd, args.smoke_eps)]
    return [(lr, wd, eps) for lr in LR_GRID for wd in WD_GRID for eps in EPS_GRID]


def _regimes(args: argparse.Namespace):
    if args.task == "task_a":
        yield "task_a", None
    elif args.task == "task_b":
        for k in args.kappas:
            yield "task_b", k
    elif args.task == "task_c":
        for a in args.alphas:
            yield "task_c", a
    else:
        raise ValueError(args.task)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["task_a", "task_b", "task_c"], default="task_b")
    p.add_argument("--kappas", nargs="+", type=float, default=[1.0e3])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.3, 1.0])
    p.add_argument("--optimizers", nargs="+",
                   choices=["adamw", "spectral"], default=["adamw"])
    p.add_argument("--grid-mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--smoke-lr", type=float, default=3.0e-3)
    p.add_argument("--smoke-wd", type=float, default=1.0e-4)
    p.add_argument("--smoke-eps", type=float, default=1.0e-8)
    p.add_argument("--all-seeds", action="store_true",
                   help="Run all SHA-derived seeds (default: just seeds[0])")
    p.add_argument("--n-seeds", type=int, default=3,
                   help="How many SHA-derived seeds to use when --all-seeds "
                        "is set.  Default 3 (the rubric anchor); for "
                        "single-config power studies set to 8-10. The first "
                        "3 values are bit-identical to the original triple.")
    p.add_argument("--student-id", default="starter",
                   help="Used to derive the SHA-256 seed triple")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--early-abort-step", type=int, default=400)
    p.add_argument("--early-abort-frac", type=float, default=2.0)
    # Spectral hyperparameters (ignored unless --optimizers includes spectral).
    p.add_argument("--mode", choices=["disabled", "debug", "production"],
                   default="disabled")
    p.add_argument("--spectral-rank", type=int, default=8)
    p.add_argument("--spectral-beta", type=float, default=0.95)
    p.add_argument("--correction-strength", type=float, default=0.0,
                   help="Scalar gate on the spectral correction. "
                        "Negative values steer AWAY from the EMA "
                        "(anti-Polyak / exploration).")
    p.add_argument("--steering-interval", type=int, default=1,
                   help="Apply the spectral correction every K steps "
                        "(K=1 is the default per-step behavior; K>1 "
                        "skips the basis algebra on intermediate steps "
                        "and emits AdamW for them).")
    p.add_argument("--basis-adapt", choices=["frozen", "oja"],
                   default="frozen",
                   help="frozen: random Gaussian frame fixed at first step "
                        "(the original method).  oja: streaming PCA via "
                        "Oja's rule with periodic QR re-orthonormalization "
                        "(adaptive basis ablation).")
    p.add_argument("--oja-lr", type=float, default=1e-2,
                   help="Oja learning rate (basis adaptation rate).  "
                        "Ignored unless --basis-adapt=oja.")
    p.add_argument("--oja-orth-interval", type=int, default=50,
                   help="Re-orthonormalize the basis every K Oja steps.")
    p.add_argument("--out-dir", default="logs")
    p.add_argument("--telemetry-suffix", default="",
                   help="Appended to the per-run telemetry filename stem. "
                        "Use this to keep telemetry files unique across "
                        "ablation sweeps that vary a parameter not "
                        "encoded in the stem (alpha, beta, rank, ...).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("telemetry").mkdir(exist_ok=True)
    seeds = derive_seeds(args.student_id, n=args.n_seeds)
    if not args.all_seeds:
        seeds = seeds[:1]

    csv_path = out_dir / f"grid_{args.task}_{args.grid_mode}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "regime", "optimizer", "seed", "lr", "wd", "eps",
            "step", "loss", "lr_scheduled", "ms", "status",
        ])
        writer.writeheader()

        for regime_name, regime_value in _regimes(args):
            for opt_name in args.optimizers:
                for lr, wd, eps in _grid(args):
                    for sidx, seed in enumerate(seeds):
                        torch.manual_seed(seed)
                        X, Y, model, loss_fn, label = _build_task(
                            args.task, regime_value, seed,
                        )
                        opt = _build_optimizer(opt_name, model, lr, wd, eps, args)
                        stem = (
                            f"{label}_{opt_name}_lr{lr:g}_wd{wd:g}_"
                            f"eps{eps:g}_seed{sidx}{args.telemetry_suffix}"
                        )
                        args.current_telemetry_path = str(
                            Path("telemetry") / f"{stem}.json"
                        )
                        rows, _, status = _train_one(
                            model, opt, X, Y, loss_fn, lr, args,
                        )
                        # Persist spectral diagnostics next to the timing
                        # telemetry.  Only the SpectralOptimizer has them;
                        # AdamW runs skip this block.
                        if hasattr(opt, "dump_diagnostics"):
                            diag_path = Path("telemetry") / f"{stem}_diag.json"
                            opt.dump_diagnostics(diag_path)
                        for r in rows:
                            writer.writerow({
                                "regime": label, "optimizer": opt_name,
                                "seed": int(seed),
                                "lr": lr, "wd": wd, "eps": eps,
                                **r, "status": status,
                            })
                        fh.flush()
                        final = rows[-1]["loss"] if rows else float("nan")
                        print(f"  [{label} {opt_name:8s} lr={lr:g} wd={wd:g} "
                              f"eps={eps:g} seed{sidx}] final={final:.4g} "
                              f"status={status}", flush=True)
                        del opt, model, X, Y
                        torch.cuda.empty_cache()

    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
