"""Student starter template for the Spectral Optimizer Capstone (DS 6210).

This file is YOURS to edit.  Do not edit anything in `harness/`.

Three modes (§2 of the handout):

    disabled   -- exactly AdamW (already implemented; do not break parity)
    debug      -- applies AdamW but logs spectral diagnostics each step
    production -- YOUR custom update rule (raises NotImplementedError until
                  you fill in `_compute_spectral_correction`)

A pure-Python AdamW reference is built into the disabled/debug branches
so that `parity_check.py` can prove your starter matches torch.optim.AdamW
to floating-point tolerance.  Once you fill in `production` you must
keep parity for `disabled` -- the parity test is the only behavioral
guard the grader runs against the harness.

A safe trajectory template (from §2 of the handout) is sketched in
`_compute_spectral_correction`.  You are free to use it, replace it,
or invent something entirely different -- as long as the optimizer is
honestly per-tensor and respects the rules in `docs/problem.tex` §1.
"""
from __future__ import annotations

import math
import os
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import torch


# When SPECTRAL_DETERMINISTIC=1 is exported (set by slurm/smoke.slurm only),
# enable bit-deterministic kernels so parity_check.py can hit the 1e-6
# tolerance.  TF32 + non-deterministic cuBLAS GEMM otherwise leaves ~1e-5
# of run-to-run noise even when the optimizer arithmetic is bit-exact (CPU
# parity passes at 6e-8).  Grid scripts intentionally leave this unset to
# keep TF32 on for fast training.
if os.environ.get("SPECTRAL_DETERMINISTIC", "0") == "1":
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


Mode = Literal["disabled", "debug", "production"]


class SpectralOptimizer(torch.optim.Optimizer):
    """A 3-mode skeleton for a per-tensor low-rank spectral optimizer.

    Args:
        params: parameters or parameter groups (same shape as torch.optim.AdamW).
        lr, betas, eps, weight_decay: standard AdamW hyperparameters.
        mode: one of {"disabled", "debug", "production"}.
        spectral_rank: per-tensor rank of the spectral subspace (default 8).
            Used only by `debug` and `production`.  Must be <= min(weight.shape).
        spectral_beta: EMA decay for whatever moment you maintain in the
            spectral subspace.  The starter does not consume this; it is
            here so your `production` branch has a hyperparameter to tune.
        correction_strength: starter scalar gate on your spectral
            correction.  `disabled`/`debug` ignore it.

    The optimizer maintains, for each 2-D parameter:
        state["step"]   : int
        state["m"]      : first moment, shape == p.shape
        state["v"]      : second moment, shape == p.shape
        state["basis"]  : optional per-tensor orthonormal frame
                          (shape (r, d_flat)), allocated lazily on first
                          `debug`/`production` step.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        mode: Mode = "disabled",
        spectral_rank: int = 8,
        spectral_beta: float = 0.95,
        correction_strength: float = 0.0,
        steering_interval: int = 1,
        basis_adapt: str = "frozen",
        oja_lr: float = 1e-2,
        oja_orth_interval: int = 50,
    ) -> None:
        if mode not in ("disabled", "debug", "production"):
            raise ValueError(
                f"mode must be one of disabled/debug/production, got {mode!r}",
            )
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if spectral_rank < 1:
            raise ValueError(f"spectral_rank must be >= 1, got {spectral_rank}")
        if steering_interval < 1:
            raise ValueError(
                f"steering_interval must be >= 1, got {steering_interval}",
            )
        if basis_adapt not in ("frozen", "oja"):
            raise ValueError(
                f"basis_adapt must be one of 'frozen' or 'oja', "
                f"got {basis_adapt!r}",
            )
        if oja_orth_interval < 1:
            raise ValueError(
                f"oja_orth_interval must be >= 1, got {oja_orth_interval}",
            )

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            mode=mode, spectral_rank=spectral_rank,
            spectral_beta=spectral_beta,
            correction_strength=correction_strength,
            steering_interval=steering_interval,
            basis_adapt=basis_adapt,
            oja_lr=oja_lr,
            oja_orth_interval=oja_orth_interval,
        )
        super().__init__(params, defaults)
        # Per-step diagnostics from `debug` mode are appended here for
        # student inspection.  Cleared on every `step()` call.
        self.last_diagnostics: dict[str, float] = {}
        # Append-only history: every step where last_diagnostics is
        # non-empty gets one row {step, effective_rank, cosine_with_adamw,
        # spectral_correction_norm}.  Persisted by the experiment runner
        # via `dump_diagnostics()` so the report can prove the controller
        # is not silent (README self-audit checklist).
        self._diag_history: list[dict] = []
        self._step_counter: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Bit-exact fast path for `disabled` mode: dispatch to torch's
        # _foreach_* kernels in the same order as torch.optim.AdamW
        # (foreach=True).  Without this, a per-tensor loop produces ~1e-5
        # divergence on GPU even with TF32 off + deterministic algorithms,
        # because the fused vs non-fused kernels round FMAs differently.
        if all(g["mode"] == "disabled" for g in self.param_groups):
            self._step_disabled_foreach()
            self.last_diagnostics = {}
            self._step_counter += 1
            return loss

        # Per-step diagnostics get aggregated across all tensors and
        # then averaged before being stored on `last_diagnostics`.
        diag_acc = {"effective_rank": 0.0, "cosine_with_adamw": 0.0,
                    "spectral_correction_norm": 0.0, "n_tensors": 0}

        for group in self.param_groups:
            mode = group["mode"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                # AdamW direction is needed by all three modes.
                d_t, m_hat, v_hat = _adamw_direction(p, state, group)

                if mode == "disabled":
                    update = d_t
                elif mode == "debug":
                    update = d_t
                    if p.dim() == 2:
                        diag = self._diagnostics_only(p, d_t, group, state)
                        for k in ("effective_rank", "cosine_with_adamw",
                                  "spectral_correction_norm"):
                            diag_acc[k] += diag.get(k, 0.0)
                        diag_acc["n_tensors"] += 1
                else:  # production
                    update = d_t
                    if p.dim() == 2:
                        correction, diag = self._trial_correction(
                            p, d_t, group, state, update_state=True,
                        )
                        update = update + correction
                        for k in ("effective_rank", "cosine_with_adamw",
                                  "spectral_correction_norm"):
                            diag_acc[k] += diag.get(k, 0.0)
                        diag_acc["n_tensors"] += 1

                # AdamW step: theta <- theta - lr * (update + wd * theta)
                lr = group["lr"]
                wd = group["weight_decay"]
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

        if diag_acc["n_tensors"] > 0:
            n = diag_acc.pop("n_tensors")
            self.last_diagnostics = {k: v / n for k, v in diag_acc.items()}
        else:
            self.last_diagnostics = {}

        self._step_counter += 1
        if self.last_diagnostics:
            self._diag_history.append(
                {"step": self._step_counter, **self.last_diagnostics},
            )

        return loss

    def dump_diagnostics(self, path: str | Path) -> dict:
        """Persist the spectral diagnostic history to a JSON file.

        Writes one row per logged step with keys
            step, effective_rank, cosine_with_adamw, spectral_correction_norm
        plus a `summary` block (n, mean over the run) and the controller
        settings.  Required by the README self-audit checklist:
        proves the controller is not silent.
        """
        rows = list(self._diag_history)
        n = len(rows)

        def _mean(key: str) -> float:
            if not rows:
                return float("nan")
            return float(sum(r[key] for r in rows) / n)

        # Controller settings come from group 0 (matches how every grid run
        # is configured — one group, one set of spectral hparams).
        g0 = self.param_groups[0] if self.param_groups else {}
        out = {
            "n_logged_steps": n,
            "controller": {
                "mode": g0.get("mode"),
                "spectral_rank": g0.get("spectral_rank"),
                "spectral_beta": g0.get("spectral_beta"),
                "correction_strength": g0.get("correction_strength"),
                "steering_interval": g0.get("steering_interval", 1),
                "basis_adapt": g0.get("basis_adapt", "frozen"),
                "oja_lr": g0.get("oja_lr", 0.0),
                "oja_orth_interval": g0.get("oja_orth_interval", 0),
            },
            "summary": {
                "mean_effective_rank": _mean("effective_rank"),
                "mean_cosine_with_adamw": _mean("cosine_with_adamw"),
                "mean_spectral_correction_norm":
                    _mean("spectral_correction_norm"),
            },
            "rows": rows,
        }
        Path(path).write_text(json.dumps(out, indent=2))
        return out

    def _step_disabled_foreach(self) -> None:
        """Bit-exact replication of torch's _multi_tensor_adamw step.

        Calls torch._foreach_* in the same order as torch.optim.adamw.
        _multi_tensor_adamw so the disabled-mode parity check passes within
        1e-6 on GPU.  Per-tensor loops with the same math give ~1e-5 noise
        because fused foreach kernels and per-tensor kernels compute FMAs
        in different rounding order.
        """
        for group in self.param_groups:
            params: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            steps: list[int] = []

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(state["m"])
                exp_avg_sqs.append(state["v"])
                steps.append(state["step"])

            if not params:
                continue

            if wd != 0.0:
                torch._foreach_mul_(params, 1.0 - lr * wd)

            torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)

            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1.0 - beta2)

            bias_correction1 = [1.0 - beta1 ** t for t in steps]
            bias_correction2_sqrt = [(1.0 - beta2 ** t) ** 0.5 for t in steps]
            step_size = [(-lr) / bc1 for bc1 in bias_correction1]

            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denom, bias_correction2_sqrt)
            torch._foreach_add_(denom, eps)

            torch._foreach_addcdiv_(params, exp_avgs, denom, step_size)

    # ------------------------------------------------------------------ #
    # Student extension point
    # ------------------------------------------------------------------ #

    def _compute_spectral_correction(
        self,
        p: torch.nn.Parameter,
        d_t: torch.Tensor,
        group: dict,
        state: dict,
    ) -> torch.Tensor:
        """Additive spectral correction to the AdamW update (production mode).

        Trajectory-EMA angular steering: project the normalized AdamW
        direction onto a frozen per-tensor orthonormal frame, smooth the
        coefficients with an EMA, and rotate this step's direction toward
        the smoothed coefficients.  AdamW step magnitude is preserved
        exactly; only the direction is rotated.

        Hypothesis: under severe anisotropy (Task B, large kappa) AdamW's
        step direction is dominated by short-horizon gradient noise along
        a few high-curvature axes.  Smoothing the projected coefficients
        removes that high-frequency component without flattening across
        layers (state is per-tensor, rank <= spectral_rank).

        Reduces to AdamW exactly when ``correction_strength == 0``.
        """
        correction, _ = self._trial_correction(
            p, d_t, group, state, update_state=True,
        )
        return correction

    def _trial_correction(
        self,
        p: torch.nn.Parameter,
        d_t: torch.Tensor,
        group: dict,
        state: dict,
        *,
        update_state: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the angular-steering correction and its diagnostics.

        Shared by `production` (update_state=True) and `_diagnostics_only`
        (update_state=False).  Returns ``(correction, diag)`` where
        ``correction`` has the same shape as ``d_t`` and ``diag`` exposes
        effective rank, cosine with AdamW, and correction norm.
        """
        # Fast path: at correction_strength == 0 the correction is
        # identically zero, so skip the basis allocation, projection,
        # EMA update, and renormalization entirely.  This (a) makes the
        # alpha=0 path bit-exactly AdamW (no eps' leak in the second
        # norm divide) and (b) eliminates the ~1.8x wall-clock penalty
        # that otherwise applies at alpha=0.
        if group["correction_strength"] == 0.0:
            return (
                torch.zeros_like(d_t),
                {"effective_rank": 0.0,
                 "cosine_with_adamw": 1.0,
                 "spectral_correction_norm": 0.0},
            )

        """ # Lazy steering: when steering_interval > 1, only do the spectral
        # work every K steps; intermediate steps emit pure AdamW.  Tests
        # whether occasional angular correction is sufficient (cost win)
        # without sacrificing quality.  EMA coefficients are sub-sampled
        # in time at rate 1/K -- this is intentional, since recomputing
        # the basis projection every step would defeat the cost win. """

        interval = group.get("steering_interval", 1)
        if interval > 1:
            sstep = state.get("step", 0)
            # state["step"] is 1-indexed (incremented in _adamw_direction),
            # so step==1 fires the correction; step==2..K-1 skip; step==K
            # fires; etc.  Phase chosen so the very first production step
            # also computes a correction (otherwise EMA never bootstraps).
            if (sstep - 1) % interval != 0:
                return (
                    torch.zeros_like(d_t),
                    {"effective_rank": 0.0,
                     "cosine_with_adamw": 1.0,
                     "spectral_correction_norm": 0.0},
                )

        d_flat = d_t.reshape(-1)
        norm = d_flat.norm() + 1e-12
        x_t = d_flat / norm

        r = min(group["spectral_rank"], p.numel())
        basis = state.get("basis")
        if basis is None or basis.shape != (r, p.numel()):
            if "basis_seed" not in state:
                state["basis_seed"] = self._next_basis_seed()
            basis = _orthonormal_rows(
                r, p.numel(),
                device=p.device, dtype=p.dtype,
                generator_seed=state["basis_seed"],
            )
            state["basis"] = basis

        c_t = basis @ x_t

        """ # Streaming-PCA basis adaptation (Oja's rule).  When enabled,
        # rotate B toward the principal directions of the unit-direction
        # stream {x_t}.  Update applied after the projection but before
        # the lift, so this step still uses B_t for both directions
        # (consistency).  Periodic QR keeps B B^T close to I_r so
        # Theorem 1 (magnitude preservation) holds exactly and the
        # rotation-angle formula in Proposition 1 holds approximately. """
        
        if update_state and group.get("basis_adapt", "frozen") == "oja":
            oja_lr = float(group.get("oja_lr", 1e-2))
            # Oja update: each row b_i gets a Hebbian increment
            # b_i <- b_i + eta * (b_i^T x_t) * x_t.  In matrix form,
            # B <- B + eta * c_t @ x_t^T.  c_t is (r,), x_t is (d,);
            # torch.outer(c_t, x_t) is the (r,d) outer product.
            basis.add_(torch.outer(c_t, x_t), alpha=oja_lr)
            ostep = int(state.get("oja_step", 0)) + 1
            state["oja_step"] = ostep
            orth_K = int(group.get("oja_orth_interval", 50))
            if ostep % orth_K == 0:
                # Re-orthonormalize via reduced QR on B^T.
                q, _ = torch.linalg.qr(basis.T, mode="reduced")
                basis_new = q.T.contiguous()
                state["basis"] = basis_new
                basis = basis_new

        beta_s = group["spectral_beta"]
        s_prev = state.get("c_ema")
        if s_prev is None:
            s_prev = torch.zeros_like(c_t)
        s_new = beta_s * s_prev + (1.0 - beta_s) * c_t
        spectral_step = state.get("spectral_step", 0) + 1
        s_hat = s_new / (1.0 - beta_s ** spectral_step)
        if update_state:
            state["c_ema"] = s_new
            state["spectral_step"] = spectral_step

        alpha = group["correction_strength"]
        phi = alpha * (s_hat - c_t)

        raw = x_t + basis.T @ phi
        y_t = raw / (raw.norm() + 1e-12)
        correction_flat = norm * y_t - d_flat
        correction = correction_flat.reshape_as(d_t)

        update_flat = d_flat + correction_flat
        cosine = (
            update_flat.dot(d_flat)
            / (update_flat.norm() * d_flat.norm() + 1e-12)
        ).item()
        probs = c_t.pow(2) / c_t.pow(2).sum().clamp(min=1e-12)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        diag = {
            "effective_rank": float(entropy.exp().item()),
            "cosine_with_adamw": float(cosine),
            "spectral_correction_norm": float(correction_flat.norm().item()),
        }
        return correction, diag

    def _next_basis_seed(self) -> int:
        seed = getattr(self, "_basis_counter", 0)
        self._basis_counter = seed + 1
        return seed

    # ------------------------------------------------------------------ #
    # Diagnostics  (used by debug mode and figures)
    # ------------------------------------------------------------------ #

    def _diagnostics_only(
        self,
        p: torch.nn.Parameter,
        d_t: torch.Tensor,
        group: dict,
        state: dict,
    ) -> dict[str, float]:
        """Compute spectral diagnostics for `p` WITHOUT mutating the update.

        The EMA state IS mutated so the trial correction reflects what
        production mode would actually emit; if we kept update_state=False
        the EMA would always start from zero and the correction would be
        identically zero, which is the silent-controller false positive.
        Only the parameter `p` is left unchanged (debug mode applies plain
        AdamW; the correction is computed-but-not-applied).

        Returns a dict with at least:
            effective_rank       : entropy-based rank of the projected
                                   coefficients (1 .. spectral_rank).
            cosine_with_adamw    : cosine between the would-be corrected
                                   update and the raw AdamW direction;
                                   1.0 means a silent controller.
            spectral_correction_norm
                                 : Frobenius norm of the trial correction.
        """
        with torch.no_grad():
            _, diag = self._trial_correction(
                p, d_t, group, state, update_state=True,
            )
            return diag


# ---------------------------------------------------------------------- #
# Free functions: AdamW reference + scaffolding helpers.
# ---------------------------------------------------------------------- #


def _adamw_direction(
    p: torch.nn.Parameter,
    state: dict,
    group: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the AdamW direction matching torch's foreach AdamW bit-exactly.

    The mathematical identity ``m_hat / (sqrt(v_hat) + eps) ==
    m / (bc1 * (sqrt(v)/sqrt(bc2) + eps))`` only holds in real arithmetic;
    in float32 the two orderings differ by ~1 ULP per step and accumulate
    past the 1e-6 parity tolerance over 100 steps.  We mirror torch's
    foreach=True implementation in `_multi_tensor_adamw`:

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.sqrt() / bc2_sqrt + eps
        update = -lr/bc1 * exp_avg / denom

    Returns ``(d_t, m_hat, v_hat)`` with ``update = -lr * d_t``.  d_t has
    the same shape as ``p``.
    """
    grad = p.grad
    beta1, beta2 = group["betas"]
    eps = group["eps"]

    state["step"] += 1
    t = state["step"]
    state["m"].lerp_(grad, 1.0 - beta1)
    state["v"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    bias1 = 1.0 - beta1 ** t
    bias2 = 1.0 - beta2 ** t
    bc2_sqrt = bias2 ** 0.5

    denom = state["v"].sqrt().div_(bc2_sqrt).add_(eps)
    d_t = state["m"] / (bias1 * denom)

    m_hat = state["m"] / bias1
    v_hat = state["v"] / bias2
    return d_t, m_hat, v_hat


def _orthonormal_rows(
    r: int, d: int,
    *, device, dtype, generator_seed: int = 0,
) -> torch.Tensor:
    """Return a frozen orthonormal frame B in R^{r x d}, B @ B.T = I_r.

    Uses a CPU torch.Generator for reproducibility, then moves to GPU.
    """
    g = torch.Generator(device="cpu").manual_seed(generator_seed)
    raw = torch.randn(r, d, generator=g)
    # QR on a tall matrix (d >= r); take the Q rows.
    q, _ = torch.linalg.qr(raw.T, mode="reduced")
    return q.T.to(device=device, dtype=dtype).contiguous()


def _starter_low_rank_template(
    d_t: torch.Tensor,
    basis: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of the §2 angular-correction template.

    Given AdamW direction ``d_t`` (shape == weight), an orthonormal
    frame ``basis`` of shape (r, d_flat), and a coefficient vector
    ``phi`` of shape (r,) chosen by the student, returns the additive
    correction such that

        d_t + correction = ||d_t|| * y_t,    y_t = (x_t + B.T @ phi) / ||...||

    with ``x_t = d_t / ||d_t||``.  Step magnitude is preserved exactly.

    Cut and paste this into your `_compute_spectral_correction` once you
    have decided how to build ``phi`` from the trajectory state.
    """
    d_flat = d_t.reshape(-1)
    norm = d_flat.norm() + 1e-12
    x_t = d_flat / norm
    raw = x_t + basis.T @ phi
    y_t = raw / (raw.norm() + 1e-12)
    correction_flat = norm * y_t - d_flat
    return correction_flat.reshape_as(d_t)
