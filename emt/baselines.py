"""
emt/baselines.py
================
Classical (non-learned) EMT reconstruction baselines.

laplacian_regularization  — closed-form Tikhonov / Laplacian reg solution
fista_tv_emt              — iterative TV minimisation with explicit A matrix
"""

import math
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLASSICAL


def laplacian_regularization(
    b:          np.ndarray,
    A:          np.ndarray,
    image_size: int   = 64,
    lam:        float = CLASSICAL["lap_reg_lam"],
) -> np.ndarray:
    """
    Solve:  min ||Ax - b||^2 + lam * ||L x||^2
    closed-form:  x* = (A^T A + lam L^T L)^{-1} A^T b

    Parameters
    ----------
    b          : (M,)     measurement vector
    A          : (M, N)   sensitivity matrix  (N = image_size^2)
    image_size : square image side length
    lam        : regularisation weight

    Returns
    -------
    x : (image_size, image_size)
    """
    from scipy.sparse import diags

    n  = image_size
    D1 = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
    I  = np.eye(n)
    L  = np.kron(D1, I) + np.kron(I, D1)
    return np.linalg.solve(
        A.T @ A + lam * L.T @ L,
        A.T @ b,
    ).reshape(n, n).astype(np.float32)


def fista_tv_emt(
    b:          np.ndarray,
    A:          np.ndarray,
    x0:         np.ndarray,
    image_size: int   = 64,
    n_iters:    int   = CLASSICAL["fista_tv_emt_iters"],
    lam:        float = CLASSICAL["fista_tv_emt_lam"],
) -> np.ndarray:
    """
    TV-regularised reconstruction for EMT (explicit A matrix).
    Uses FISTA with isotropic TV proximal step.

    Parameters
    ----------
    b, A   : measurement vector and sensitivity matrix
    x0     : (image_size, image_size) initial image (e.g. from Laplacian reg)
    n_iters, lam : FISTA iterations and TV weight

    Returns
    -------
    x : (image_size, image_size) float32
    """
    L  = np.linalg.norm(A, ord=2) ** 2 + 1e-6
    mu = 1.0 / L
    x  = x0.flatten().astype(np.float64)
    y  = x.copy(); t = 1.0; xp = x.copy()

    for _ in range(n_iters):
        r = y - mu * (A.T @ (A @ y - b.astype(np.float64)))
        r2d = r.reshape(image_size, image_size)

        # Isotropic TV proximal
        dx  = np.diff(r2d, axis=1, append=r2d[:, -1:])
        dy  = np.diff(r2d, axis=0, append=r2d[-1:, :])
        mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        dx /= mag; dy /= mag
        div = dx - np.roll(dx, 1, axis=1) + dy - np.roll(dy, 1, axis=0)
        x   = (r2d - lam * mu * div).flatten()

        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        y     = x + ((t - 1.0) / t_new) * (x - xp)
        xp, t = x.copy(), t_new

    return x.reshape(image_size, image_size).astype(np.float32)
