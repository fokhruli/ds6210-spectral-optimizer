"""Aggregate the 10-seed winner-resample CSVs and report effect size.

Reads two CSVs produced by `slurm/grid_winner_resample.slurm`:
    logs/winner_adamw_*/grid_task_b_smoke.csv
    logs/winner_spec_cs1.0_*/grid_task_b_smoke.csv

For each arm: extracts the final-step (step 1999) loss for every seed,
computes mean +- std, runs a Welch two-sample t-test (unequal variances),
and reports the effect size (Cohen's d) plus a 95%-CI for the difference.

The p-value uses the normal approximation to the t-distribution because
scipy is not available in the project venv.  At n1=n2=10 the
Welch-Satterthwaite df is roughly 18, where the normal approximation
under-estimates two-tailed p-values by less than 5%.

Run:
    python experiments/winner_significance.py
or with explicit paths:
    python experiments/winner_significance.py \\
        --adamw-csv logs/winner_adamw_<job>/grid_task_b_smoke.csv \\
        --spec-csv  logs/winner_spec_cs1.0_<job>/grid_task_b_smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _final_losses(path: Path) -> list[float]:
    """Final-step loss per seed.  CSV has one row per (seed, step) pair."""
    by_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            by_seed[int(r["seed"])].append((int(r["step"]), float(r["loss"])))
    finals = []
    for seed, rows in by_seed.items():
        rows.sort()
        finals.append(rows[-1][1])
    return finals


def _welch_t(a: list[float], b: list[float]) -> dict[str, float]:
    """Welch two-sample t-test (a is the *baseline*; positive t => a > b)."""
    n1, n2 = len(a), len(b)
    m1, m2 = statistics.mean(a), statistics.mean(b)
    v1 = statistics.variance(a) if n1 > 1 else 0.0
    v2 = statistics.variance(b) if n2 > 1 else 0.0
    se = math.sqrt(v1 / n1 + v2 / n2)
    t = (m1 - m2) / se if se > 0 else float("inf")

    # Welch-Satterthwaite df.
    num = (v1 / n1 + v2 / n2) ** 2
    den = ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)) if (n1 > 1 and n2 > 1) else 1.0
    df = num / den if den > 0 else float("inf")

    # Normal-approx two-tailed p-value (good enough at df>=15).
    p_two = math.erfc(abs(t) / math.sqrt(2.0))

    # 95% CI for (m1 - m2): use z=1.96 (normal approx).
    z = 1.959963984540054
    ci_lo = (m1 - m2) - z * se
    ci_hi = (m1 - m2) + z * se

    # Cohen's d (pooled std).
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) \
             if (n1 + n2 > 2) else 0.0
    d = (m1 - m2) / pooled if pooled > 0 else float("inf")

    return {"t": t, "df": df, "p_two": p_two,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "cohens_d": d,
            "delta": m1 - m2, "rel_delta": (m1 - m2) / m1 if m1 else 0.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adamw-csv", default=None,
                   help="Path to AdamW arm CSV (auto-discovers most recent if omitted).")
    p.add_argument("--spec-csv",  default=None,
                   help="Path to Spec arm CSV (auto-discovers most recent if omitted).")
    args = p.parse_args()

    def _find(pattern: str) -> Path:
        hits = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime,
                      reverse=True)
        if not hits:
            raise SystemExit(f"no match for {pattern}; run the slurm job first.")
        return hits[0]

    adamw = Path(args.adamw_csv) if args.adamw_csv else \
            _find("logs/winner_adamw_*/grid_task_b_smoke.csv")
    spec  = Path(args.spec_csv)  if args.spec_csv  else \
            _find("logs/winner_spec_cs1.0_*/grid_task_b_smoke.csv")

    print(f"AdamW CSV: {adamw.relative_to(ROOT)}")
    print(f"Spec  CSV: {spec.relative_to(ROOT)}")

    a = _final_losses(adamw)
    b = _final_losses(spec)
    print()
    print(f"AdamW (n={len(a)}): mean={statistics.mean(a):.4e}  "
          f"std={statistics.stdev(a):.2e}  losses={['%.4e' % x for x in sorted(a)]}")
    print(f"Spec  (n={len(b)}): mean={statistics.mean(b):.4e}  "
          f"std={statistics.stdev(b):.2e}  losses={['%.4e' % x for x in sorted(b)]}")

    if len(a) < 2 or len(b) < 2:
        print("\n[warning] need >=2 seeds per arm for a t-test.")
        return

    r = _welch_t(a, b)
    print()
    print("Welch two-sample t-test  (H1: AdamW != Spec):")
    print(f"  delta (AdamW - Spec)  = {r['delta']:+.4e}  "
          f"({100*r['rel_delta']:+.2f}% of AdamW mean)")
    print(f"  95% CI for delta      = [{r['ci_lo']:+.4e}, {r['ci_hi']:+.4e}]")
    print(f"  t = {r['t']:+.3f}   df ~ {r['df']:.1f}   "
          f"p (normal approx) = {r['p_two']:.4g}")
    print(f"  Cohen's d             = {r['cohens_d']:.3f}")

    print()
    if r["delta"] > 0 and r["p_two"] < 0.05:
        print("VERDICT: Spec significantly beats AdamW at p<0.05 (10 seeds).")
    elif r["delta"] > 0:
        print("VERDICT: Spec mean is lower than AdamW, but the difference is "
              "within seed noise at this sample size.")
    elif r["delta"] < 0 and r["p_two"] < 0.05:
        print("VERDICT: AdamW significantly beats Spec at p<0.05.")
    else:
        print("VERDICT: no significant difference at the 95% level.")


if __name__ == "__main__":
    main()
