"""VRAM-resident data generators for Tasks A, B, C.

Critical rule (§7 of the handout):
    ALL datasets are pre-allocated on the GPU BEFORE the CUDA-event
    timer starts.  Batch slicing must be a pure GPU-to-GPU operation.
    PCIe transfers from a standard torch.utils.data.DataLoader will
    dominate step time and invalidate your wall-clock claim.
"""
from __future__ import annotations

from pathlib import Path

import torch


# --------------------------------------------------------------------- #
#  Task A  --  FashionMNIST classification                              #
# --------------------------------------------------------------------- #
def make_task_a(
    n_samples: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    root: str = "data",
) -> tuple[torch.Tensor, torch.Tensor]:
    """FashionMNIST training set, normalized and pushed entirely to GPU.

    Returns (X, Y) where X has shape (N, 784) and Y has shape (N,) with
    integer class labels suitable for ``F.cross_entropy``.

    Mean/std (0.2860, 0.3530) are the FashionMNIST training-set
    statistics.  Pass ``n_samples`` to truncate for a faster smoke run.
    """
    try:
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:
        raise RuntimeError(
            "Task A requires torchvision; `pip install torchvision`",
        ) from exc

    train = FashionMNIST(root=Path(root), train=True, download=True)
    X = train.data.to(device=device, dtype=dtype).flatten(1) / 255.0
    X = (X - 0.2860) / 0.3530
    Y = train.targets.to(device=device, dtype=torch.long)
    if n_samples is not None:
        X, Y = X[:n_samples], Y[:n_samples]
    return X, Y


# --------------------------------------------------------------------- #
#  Task B  --  ill-conditioned synthetic regression (kappa-sweep)       #
# --------------------------------------------------------------------- #
def make_task_b(
    kappa: float,
    n_samples: int = 20_000,
    d: int = 256,
    out_dim: int = 10,
    seed: int = 0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotated-anisotropy regression.

    x_n ~ N(0, Q Sigma Q^T) with sigma_i = kappa^{-(i-1)/(d-1)}.
    y_n = B x_n + zeta_n,  B ~ N(0, 0.01),  zeta_n ~ N(0, 0.01^2).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    sigma = torch.tensor(
        [kappa ** (-(i / (d - 1))) for i in range(d)], dtype=torch.float64,
    ).to(torch.float32)
    L = Q @ torch.diag(sigma.sqrt())
    z = torch.randn(n_samples, d, generator=g)
    X = (z @ L.T).to(device=device, dtype=dtype)
    B = (torch.randn(out_dim, d, generator=g) * 0.1).to(device=device, dtype=dtype)
    noise = (torch.randn(n_samples, out_dim, generator=g) * 0.01).to(
        device=device, dtype=dtype,
    )
    Y = X @ B.T + noise
    return X, Y


# --------------------------------------------------------------------- #
#  Task C  --  teacher-student with prescribed teacher spectrum         #
# --------------------------------------------------------------------- #
def make_task_c_teacher(
    alpha: float = 0.3,
    K: int = 4,
    d: int = 500,
    seed: int = 0,
    device: str = "cuda",
) -> torch.nn.Module:
    """2-hidden-layer Erf MLP with W_1 = U diag(exp(-alpha * i)) V^T.

    K = 4 hidden units, input dim d = 500.  The exponential singular-value
    decay controls how much of the relevant subspace is concentrated in
    the first few directions.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(K, K, generator=g))
    V, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    s = torch.tensor([torch.exp(torch.tensor(-alpha * i)).item() for i in range(K)])
    W1 = U @ torch.diag(s) @ V[:, :K].T
    w2 = torch.randn(K, generator=g)

    class Teacher(torch.nn.Module):
        def __init__(self, W1, w2):
            super().__init__()
            self.register_buffer("W1", W1)
            self.register_buffer("w2", w2)

        def forward(self, x):
            return torch.erf(x @ self.W1.T) @ self.w2

    return Teacher(W1.to(device=device), w2.to(device=device))


def sample_task_c(
    teacher: torch.nn.Module,
    n_samples: int,
    d: int = 500,
    noise: float = 0.01,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(n_samples, d, generator=g).to(device=device)
    with torch.no_grad():
        Y = teacher(X) + noise * torch.randn(n_samples, generator=g).to(device=device)
    return X, Y.unsqueeze(-1)


# --------------------------------------------------------------------- #
#  GPU-native batch slicing                                             #
# --------------------------------------------------------------------- #
def batch_slice(
    X: torch.Tensor, Y: torch.Tensor, batch_size: int, step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap-around batch slicing that stays entirely on the GPU.

    No DataLoader, no PCIe.
    """
    n = X.shape[0]
    i = (step * batch_size) % n
    if i + batch_size <= n:
        return X[i:i + batch_size], Y[i:i + batch_size]
    head_xb, head_yb = X[i:], Y[i:]
    tail = batch_size - head_xb.shape[0]
    return torch.cat([head_xb, X[:tail]], 0), torch.cat([head_yb, Y[:tail]], 0)
