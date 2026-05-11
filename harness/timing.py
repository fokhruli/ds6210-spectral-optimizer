"""CUDA-event-based telemetry.  ``time.time()`` is banned (§7 of the handout).

Usage:

    timer = CUDAEventTimer()
    timer.start("forward"); out = model(x); timer.end("forward", step=s)
    timer.start("backward"); loss.backward(); timer.end("backward", step=s)
    timer.start("optimizer_step"); opt.step(); timer.end("optimizer_step", step=s)
    ...
    timer.sync_and_dump("telemetry.json")

The JSON artifact contains:
    * an ``nvidia-smi`` header (GPU model, driver, memory)
    * torch and CUDA versions
    * per-label summary (n, total_ms, mean_ms)
    * per-step row log

The grader reconciles ``sum(per-step rows)`` vs total wall-clock and
rejects claims that disagree by more than 5%.
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import torch


class CUDAEventTimer:
    def __init__(self) -> None:
        self._active: dict[str, torch.cuda.Event] = {}
        self._log: list[dict] = []

    def start(self, label: str) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._active[label] = ev

    def end(self, label: str, step: int | None = None) -> None:
        start = self._active.pop(label)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._log.append(
            {"label": label, "start": start, "end": end, "step": step}
        )

    def sync_and_dump(self, path: str | Path) -> dict:
        torch.cuda.synchronize()
        rows = []
        by_label: dict[str, list[float]] = defaultdict(list)
        for rec in self._log:
            ms = rec["start"].elapsed_time(rec["end"])
            by_label[rec["label"]].append(ms)
            rows.append({"step": rec["step"], "label": rec["label"], "ms": ms})
        summary = {
            k: {
                "n": len(v),
                "total_ms": float(sum(v)),
                "mean_ms": float(sum(v) / len(v)),
            }
            for k, v in by_label.items()
        }
        out = {
            "nvidia_smi": _nvidia_smi_header(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "summary": summary,
            "rows": rows,
        }
        Path(path).write_text(json.dumps(out, indent=2))
        return out


def _nvidia_smi_header() -> dict:
    try:
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True, capture_output=True, text=True,
        )
        parts = [x.strip() for x in cp.stdout.strip().split(",")]
        return {"gpu": parts[0], "driver": parts[1], "memory_total": parts[2]}
    except Exception as e:
        return {"error": str(e)}
