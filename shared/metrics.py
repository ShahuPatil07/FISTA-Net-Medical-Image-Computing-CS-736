"""
shared/metrics.py
=================
Evaluation metrics and table printing utilities used by both CT and EMT.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity  as _ssim


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute PSNR, SSIM, RMSE for a batch or single image pair.

    Parameters
    ----------
    pred, gt : numpy arrays of shape (H, W) or (B, H, W)

    Returns
    -------
    {'PSNR': float, 'SSIM': float, 'RMSE': float}  — averaged over batch
    """
    if pred.ndim == 2:
        pred = pred[np.newaxis]
        gt   = gt[np.newaxis]

    psnrs, ssims, rmses = [], [], []
    for p, g in zip(pred, gt):
        dr = max(float(g.max() - g.min()), 1e-8)
        psnrs.append(_psnr(g, p, data_range=dr))
        ssims.append(_ssim(g, p, data_range=dr))
        rmses.append(float(np.sqrt(np.mean((p - g) ** 2))))

    return {
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "RMSE": float(np.mean(rmses)),
    }


def print_results_table(
    results: dict,
    title: str = "Reconstruction Comparison",
    highlight: str = "FISTA-Net",
) -> None:
    """
    Print a paper-style comparison table.

    Parameters
    ----------
    results   : {method_name: {'PSNR': list, 'SSIM': list, 'RMSE': list}}
    title     : table header string
    highlight : method name to mark with ◄ (proposed)
    """
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Method':<18} {'PSNR (dB)':>16} {'SSIM':>16} {'RMSE':>12}")
    print("─" * 72)
    for method, mets in results.items():
        vals = {k: (np.mean(v), np.std(v)) for k, v in mets.items()}
        star = " ◄" if method == highlight else "  "
        print(
            f"  {method:<18}"
            f"  {vals['PSNR'][0]:>7.3f} ± {vals['PSNR'][1]:<5.3f}"
            f"  {vals['SSIM'][0]:>6.4f} ± {vals['SSIM'][1]:<6.4f}"
            f"  {vals['RMSE'][0]:>6.4f} ± {vals['RMSE'][1]:<5.4f}"
            f"{star}"
        )
    print(sep)
    print("  ◄ Proposed  |  PSNR/SSIM ↑ higher is better  |  RMSE ↓ lower is better")
    print(sep)


def save_results_csv(results: dict, path) -> None:
    """Save per-sample metric arrays to a CSV file."""
    import csv
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    max_len = max(len(v["PSNR"]) for v in results.values())

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["sample"]
        for m in methods:
            header += [f"{m}_PSNR", f"{m}_SSIM", f"{m}_RMSE"]
        w.writerow(header)
        for i in range(max_len):
            row = [i]
            for m in methods:
                d = results[m]
                row += [
                    d["PSNR"][i] if i < len(d["PSNR"]) else "",
                    d["SSIM"][i] if i < len(d["SSIM"]) else "",
                    d["RMSE"][i] if i < len(d["RMSE"]) else "",
                ]
            w.writerow(row)
    print(f"Results saved → {path}")


def save_results_summary_csv(results: dict, path) -> None:
    """Save mean±std summary table to CSV."""
    import csv
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "PSNR_mean", "PSNR_std", "SSIM_mean", "SSIM_std", "RMSE_mean", "RMSE_std"])
        for method, mets in results.items():
            w.writerow([
                method,
                np.mean(mets["PSNR"]), np.std(mets["PSNR"]),
                np.mean(mets["SSIM"]), np.std(mets["SSIM"]),
                np.mean(mets["RMSE"]), np.std(mets["RMSE"]),
            ])
    print(f"Summary saved → {path}")
