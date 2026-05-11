# Spectral Optimizer Capstone — Starter Harness (DS 6210)

A small, read-only scaffolding that lets you focus on the one file that
matters: `my_optimizer.py`. Everything in `harness/` is fixed for the
class so all student results are comparable.

> The full assignment sheet (problem statement, rubric, rules) is
> distributed separately by the instructor.

## Layout

    my_optimizer.py                ← your work lives here (and only here)
    harness/
      model.py                     pre-normalized residual MLP backbone
      data.py                      Task A (FashionMNIST), B (kappa-sweep), C (teacher)
      timing.py                    CUDAEventTimer (replaces time.time)
      seeds.py                     SHA-256 seed derivation
      train.py                     minimal training loop
    experiments/
      run_adamw_baseline.py        AdamW(foreach=True) sanity — RUN THIS FIRST
      run_my_optimizer.py          your method, single seed, Task B
      parity_check.py              proves your `mode='disabled'` matches AdamW
      run_grid.py                  small 7×3×2 = 42-config grid runner
    tests/
      test_disabled_parity.py      pytest sanity for the disabled branch
    reproduce.sh                   end-to-end smoke (fits in a short GPU session)
    requirements.txt

## Quickstart (≤ 5 minutes once CUDA is up)

```bash
pip install -r requirements.txt
python experiments/run_adamw_baseline.py     # confirms GPU + harness wired
python experiments/parity_check.py           # proves disabled-mode == AdamW
# now edit my_optimizer.py
python experiments/run_my_optimizer.py       # your method, 1 seed, Task B
```

Both `run_*` scripts emit a `*_telemetry.json` with CUDA-event traces.

## What you must do

The starter `my_optimizer.py` ships with three modes:

| mode         | behavior                                                  |
| ------------ | --------------------------------------------------------- |
| `disabled`   | exactly AdamW (already implemented; do not break parity)  |
| `debug`      | applies AdamW but logs spectral diagnostics each step     |
| `production` | YOUR custom update rule (raises `NotImplementedError`)    |

Your job is to fill in `production`. Before doing so:

1. Run `parity_check.py` and confirm `disabled` matches `torch.optim.AdamW`
   to within 1e-7 max-abs over 100 steps.
2. Run `debug` for a few steps and look at the diagnostics that get
   logged (effective rank, cosine-with-AdamW). These are your sanity
   meters — if your `production` mode silently emits zero correction,
   the diagnostics will show it.
3. Pre-register three predictions (where you'll win, where AdamW wins,
   where you'll fail). Write them in the report before running the grid.

## A safe template (from §2 of the handout)

For a single 2-D weight tensor with AdamW direction `d_t = m_hat / (sqrt(v_hat) + eps)`,
let `x_t = d_t / ||d_t||`, maintain a per-tensor orthonormal basis
`B_t ∈ R^{r×d}`, compute coefficients `c_t = B_t @ x_t`, choose
`Φ_t ∈ R^r`, and emit

    y_t = (x_t + B_t.T @ Φ_t) / ||x_t + B_t.T @ Φ_t||
    Δθ_t = -η_t * ||d_t|| * y_t

This preserves AdamW's step magnitude and rotates only the direction.
The realized angle is

    tan(θ_t) = ||Φ_perp,t|| / (1 + <Φ_t, c_t>)

If `tan(θ_t) ≈ 0` over your whole run, the correction is a no-op and you
should report that — not claim a mechanism.

## Course grid

Run only the configurations actually needed for your hypothesis:

```bash
# Smoke (1 config, 1 seed, ~2 minutes on A100):
python experiments/run_grid.py --task task_b --kappas 1e3 \
    --optimizers adamw spectral --grid-mode smoke

# Course grid (7 lr × 3 wd × 2 eps = 42 configs per optimizer per task,
# 3 SHA-derived seeds, ~30-90 min depending on task):
python experiments/run_grid.py --task task_b --kappas 1e3 1e6 \
    --optimizers adamw spectral --grid-mode full --all-seeds
```

CSV columns: `regime,optimizer,seed,lr,wd,eps,step,loss,ms`. One row
per logged step. Aggregate yourself; the grader only requires the raw
CSVs and the figure-generating script.

## Self-audit checklist (before submitting)

- [ ] `my_optimizer.SpectralOptimizer` subclasses `torch.optim.Optimizer`
- [ ] `mode='disabled'` is bit/tolerance-equivalent to AdamW (parity test passes)
- [ ] Spectral state is **per-tensor** — no `torch.cat`/flatten across layers
- [ ] No `torch.linalg.svd` or `eigh` on matrices of dimension > 256
- [ ] AdamW baseline uses `foreach=True` (not `fused=True`)
- [ ] All data tensors live in VRAM **before** the timer starts
- [ ] Three SHA-derived seeds (`derive_seeds(...)`) anchor every plot
- [ ] Wall-clock claims backed by `CUDAEventTimer.sync_and_dump(...)`
- [ ] At least one diagnostic log proves your controller is **not silent**
- [ ] `reproduce.sh` runs end-to-end in a short GPU session
- [ ] Three pre-registered predictions in the report (one win, one loss, one failure)

## Common traps

- **Silent controller** — the most common pitfall. If your effective rank
  is ≤ 1 or your realized angle is below noise, you have not implemented
  a different optimizer; you have implemented AdamW with extra overhead.
- **Boundary-grid win** — best lr is the largest or smallest tested
  value. Extend the grid and re-run.
- **Mixed CSVs** — if you run multiple jobs, include `optimizer` and
  `seed` columns; the runner already does this.
- **Bad timing** — never use `time.time()` for per-step claims. Use the
  `CUDAEventTimer`.
- **Layer flattening** — a spectral trick that "works" only because it
  mixes unrelated tensors does not work.

## What is NOT included on purpose

- No SLURM batch scripts. Run locally; if you need a cluster, write your
  own one-liner.
- No production reference optimizer. Your starter mode is `disabled`;
  the spectral correction is the assignment.
- No 40-script research-grade orchestration. The four scripts in
  `experiments/` cover the entire deliverable surface.
