"""Pre-normalized residual MLP backbone -- §6 of the handout.

    x <- x + W_2 . GELU(W_1 . RMSNorm(x))

with He fan-in init on W_1 and zero init on W_2 (fixup-style residual
stability).  Architecture is fixed; the optimizer is the only variable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.g * x / rms


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = RMSNorm(width)
        self.w1 = nn.Linear(width, width, bias=False)
        self.w2 = nn.Linear(width, width, bias=False)
        nn.init.kaiming_normal_(self.w1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w2(F.gelu(self.w1(self.norm(x))))


class ResidualMLP(nn.Module):
    """Input -> Linear(embed) -> [ResidualBlock x depth] -> RMSNorm -> Linear(head).

    Canonical per-task configurations (§6 of the handout):
        Task A (Fashion-MNIST)   : in=784, width=512, depth=16, out=10
        Task B (kappa-sweep)     : in=256, width=256, depth=24, out=10
        Task C (teacher-student) : in=500, width=128, depth=8,  out=1
    """

    def __init__(self, in_dim: int, width: int, depth: int, out_dim: int):
        super().__init__()
        self.embed = nn.Linear(in_dim, width, bias=False)
        nn.init.kaiming_normal_(self.embed.weight, mode="fan_in", nonlinearity="relu")
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(depth)])
        self.final_norm = RMSNorm(width)
        self.head = nn.Linear(width, out_dim, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))
