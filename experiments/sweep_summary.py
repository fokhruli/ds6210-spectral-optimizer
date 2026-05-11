"""Summarize a 1-D sweep over a single hyperparameter.

After running any of the slurm/sweep_*.slurm scripts, point this at the
parent directory and it will:
    1. group runs by the swept value (extracted from the subdir name),
    2. extract final-step loss per (value, seed),
    3. report mean +- std plus per-step ms,
    4. flag any value that beats the n=10 winner-resample AdamW baseline
       (1.4975e-4 +- 1.06e-5) with more than 2 sigma.

Usage:
    python experiments/sweep_summary.py logs/sweep_alpha_12345
    python experiments/sweep_summary.py logs/sweep_beta_12346
    python experiments/sweep_summary.py logs/sweep_rank_12347
    python experiments/sweep_summary.py logs/sweep_lazy_12348

For Task A / Task C smokes the directory layout is different (adamw/
and spec/ subdirs); this script is for the 1-D parameter sweeps only.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

# n=10 AdamW baseline at the winner config (Section 5.5 of the report).
ADAMW_BASELINE_MEAN = 1.4975e-4
ADAMW_BASELINE_STD  = 1.06e-5
ADAMW_BASELINE_N    = 10


def _final_losses(csv_path: Path) -> dict[int, float]:
    """seed -> final-step loss"""
    by_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            by_seed[int(r["seed"])].append(
                (int(r["step"]), float(r["loss"])))
    return {s: sorted(rows)[-1][1] for s, rows in by_seed.items()}


def _ms_per_step(csv_path: Path) -> float | None:
    """Mean of (cumulative_ms / max_step) across seeds in this CSV."""
    by_seed: dict[int, tuple[int, float]] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"]); step = int(r["step"])
            ms = float(r["ms"]) if r["ms"] not in ("", None) else 0.0
            cur = by_seed.get(seed, (-1, 0.0))
            if step > cur[0]:
                by_seed[seed] = (step, ms)
    rates = [(ms / s) for s, ms in by_seed.values() if s > 0]
    return statistics.mean(rates) if rates else None


def _parse_value(subdir: str) -> str:
    """Extract the swept value from a subdir name like 'alpha_-1.0'."""
    m = re.match(r"^[a-zA-Z]+_(.+)$", subdir)
    return m.group(1) if m else subdir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sweep_dir", type=Path,
                   help="Parent dir containing one subdir per swept value.")
    args = p.parse_args()

    if not args.sweep_dir.is_dir():
        raise SystemExit(f"not a directory: {args.sweep_dir}")

    rows = []
    for sub in sorted(args.sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        # The runner writes one CSV per sweep iteration:
        # logs/sweep_*/<value>/grid_task_b_smoke.csv
        csv_paths = list(sub.glob("grid_*_smoke.csv"))
        if not csv_paths:
            print(f"[skip] no smoke CSV in {sub}")
            continue
        if len(csv_paths) > 1:
            print(f"[warn] multiple CSVs in {sub}; using {csv_paths[0].name}")
        path = csv_paths[0]
        value = _parse_value(sub.name)
        finals = list(_final_losses(path).values())
        if not finals:
            continue
        m = statistics.mean(finals)
        sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
        ms = _ms_per_step(path)
        rows.append({
            "value": value, "n": len(finals),
            "mean": m, "std": sd, "ms_per_step": ms,
            "delta_vs_baseline": m - ADAMW_BASELINE_MEAN,
        })

    if not rows:
        raise SystemExit("no sweep CSVs found.")

    # Try numeric sort, fall back to string sort.
    def _key(r):
        try: return (0, float(r["value"]))
        except ValueError: return (1, r["value"])
    rows.sort(key=_key)

    print(f"\nSweep summary for {args.sweep_dir}")
    print(f"AdamW baseline (n=10): {ADAMW_BASELINE_MEAN:.4e} +- "
          f"{ADAMW_BASELINE_STD:.2e}\n")
    print(f"{'value':>10s}  {'n':>3s}  {'mean':>11s}  {'std':>11s}  "
          f"{'ms/step':>9s}  {'delta_vs_AdamW':>14s}  flag")
    print("-" * 78)
    for r in rows:
        # flag: '*' if 2-sigma below the AdamW baseline.
        # Use pooled std for a rough Welch-style decision threshold.
        n = r["n"]
        if n >= 2 and r["std"] > 0:
            pooled_se = (r["std"]**2 / n + ADAMW_BASELINE_STD**2 / ADAMW_BASELINE_N) ** 0.5
            z = (ADAMW_BASELINE_MEAN - r["mean"]) / pooled_se
        else:
            z = 0.0
        flag = "WIN" if z > 2.0 else ("near" if z > 1.0 else "")
        ms_str = f"{r['ms_per_step']:>9.2f}" if r["ms_per_step"] else "       --"
        print(f"{r['value']:>10s}  {n:>3d}  {r['mean']:>11.4e}  "
              f"{r['std']:>11.2e}  {ms_str}  {r['delta_vs_baseline']:>+14.3e}  "
              f"{flag}")
    print()

    promote = [r for r in rows
               if r["n"] >= 2 and r["std"] > 0
               and (ADAMW_BASELINE_MEAN - r["mean"]) /
                   ((r["std"]**2 / r["n"] + ADAMW_BASELINE_STD**2 / ADAMW_BASELINE_N) ** 0.5) > 2.0]
    if promote:
        print("Recommend promoting these values to a 10-seed paired resample:")
        for r in promote:
            print(f"  value = {r['value']}")
    else:
        print("No value clears the 2-sigma threshold; report as a flat ablation curve.")


if __name__ == "__main__":
    main()
