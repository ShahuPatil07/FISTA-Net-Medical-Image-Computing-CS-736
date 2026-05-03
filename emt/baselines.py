import math
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLASSICAL


def laplacian_regularization(
    b:   np.ndarray,
    A:   np.ndarray,
    L:   np.ndarray,
    lam: float = CLASSICAL["lap_reg_lam"],
) -> np.ndarray:
    n = int(round(A.shape[1] ** 0.5))
    return np.linalg.solve(
        A.T @ A + lam * L.T @ L,
        A.T @ b,
    ).reshape(n, n).astype(np.float32)


def fista_tv_emt(
    b:       np.ndarray,
    A:       np.ndarray,
    x0:      np.ndarray,
    n_iters: int   = CLASSICAL["fista_tv_emt_iters"],
    lam:     float = CLASSICAL["fista_tv_emt_lam"],
) -> np.ndarray:
    H, W = x0.shape
    L_lip = np.linalg.norm(A, ord=2) ** 2 + 1e-6
    mu    = 1.0 / L_lip
    x     = x0.flatten().astype(np.float64)
    y     = x.copy(); t = 1.0; xp = x.copy()

    for _ in range(n_iters):
        r   = y - mu * (A.T @ (A @ y - b.astype(np.float64)))
        r2d = r.reshape(H, W)

        dx  = np.diff(r2d, axis=1, append=r2d[:, -1:])
        dy  = np.diff(r2d, axis=0, append=r2d[-1:, :])
        mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        dx /= mag; dy /= mag
        div = dx - np.roll(dx, 1, axis=1) + dy - np.roll(dy, 1, axis=0)
        x   = (r2d - lam * mu * div).flatten()

        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        y     = x + ((t - 1.0) / t_new) * (x - xp)
        xp, t = x.copy(), t_new

    return x.reshape(H, W).astype(np.float32)
