"""
ct/baselines.py
===============
Classical (non-learned) CT reconstruction baselines.

FBP       — already returned by MayoCTDataset as the 'fbp' tensor; no extra code needed.
FISTA-TV  — iterative TV minimisation using Radon/iRadon operators.
"""

import numpy as np
from skimage.transform import radon as sk_radon, iradon as sk_iradon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CT, CLASSICAL


def fista_tv_ct(
    fbp_img:  np.ndarray,
    sino_np:  np.ndarray,
    angles:   np.ndarray,
    n_iters:  int   = CLASSICAL["fista_tv_ct_iters"],
    lam:      float = CLASSICAL["fista_tv_ct_lam"],
    step:     float = CLASSICAL["fista_tv_ct_step"],
) -> np.ndarray:
    """
    Total-Variation regularised CT reconstruction via FISTA.
    All operations are in numpy — runs per-image (not batched).

    Parameters
    ----------
    fbp_img : H×W float32  —  FBP initialisation (from MayoCTDataset)
    sino_np : H×n_views float32  —  sparse sinogram
    angles  : (n_views,) float  —  projection angles in degrees
    n_iters : number of FISTA outer iterations
    lam     : TV regularisation weight
    step    : gradient step size (μ)

    Returns
    -------
    x : H×W float32  —  TV-reconstructed image in [0, 1]
    """
    mu    = step
    x     = fbp_img.copy().astype(np.float64)
    y     = x.copy()
    t     = 1.0
    x_prev = x.copy()

    def forward(img):
        return sk_radon(img, theta=angles, circle=True)

    def adjoint(sino):
        return sk_iradon(sino, theta=angles, filter_name="ramp", circle=True)

    for _ in range(n_iters):
        # Gradient of data fidelity: A^T (A y - b)
        grad = adjoint(forward(y) - sino_np.astype(np.float64))
        r    = y - mu * grad

        # Isotropic TV proximal step (forward-difference approximation)
        dx  = np.diff(r, axis=1, append=r[:, -1:])
        dy  = np.diff(r, axis=0, append=r[-1:, :])
        mag = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        dx /= mag
        dy /= mag
        div = (dx - np.roll(dx, 1, axis=1) + dy - np.roll(dy, 1, axis=0))
        x   = np.clip(r - lam * mu * div, 0.0, 1.0)

        # FISTA momentum
        t_new  = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        y      = x + ((t - 1.0) / t_new) * (x - x_prev)
        x_prev = x.copy()
        t      = t_new

    return x.astype(np.float32)
