"""Spectral Optimizer Capstone -- read-only harness.

Do not edit any file in this package. All student work lives in
`my_optimizer.py` at the top level of the repository.
"""
from .data import (
    batch_slice,
    make_task_a,
    make_task_b,
    make_task_c_teacher,
    sample_task_c,
)
from .model import ResidualBlock, ResidualMLP, RMSNorm
from .seeds import derive_seed, derive_seeds
from .timing import CUDAEventTimer
from .train import train

__all__ = [
    "ResidualMLP", "ResidualBlock", "RMSNorm",
    "make_task_a", "make_task_b", "make_task_c_teacher", "sample_task_c",
    "batch_slice",
    "CUDAEventTimer",
    "derive_seed", "derive_seeds",
    "train",
]
