"""
ct/evaluate_all.py
==================
Load saved model weights and evaluate ALL methods on the CT test set.
Saves tables (CSV) and figures (PNG) to ct/results/.

This script requires that you have already trained the three deep models:
    fistanet_ct_ep*_psnr*.pth
    istanet_ct_ep*_psnr*.pth
    fbpconvnet_ct_ep*_psnr*.pth

Usage
-----
    python ct/evaluate_all.py                         # auto-selects best checkpoints
    python ct/evaluate_all.py --fista path/to/ckpt.pth --ista ...  --fbpc ...
    python ct/evaluate_all.py --n_display 4 --fista_tv_iters 100
"""

import argparse, sys, os, copy
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config          import (CT, FISTA_NET, ISTA_NET, FBPCONVNET, CLASSICAL, EVAL,
                              DEVICE, CT_WEIGHTS_DIR, CT_RESULTS_DIR, CT_DATA_DIR)
from ct.dataset      import build_ct_loaders, RadonOperator
from ct.baselines    import fista_tv_ct
from ct.train_fistanet  import run_fista_ct
from ct.train_istanet   import run_ista_ct
from shared.models   import FISTANet, ISTANet, FBPConvNet
from shared.metrics  import (compute_metrics, print_results_table,
                              save_results_csv, save_results_summary_csv)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_best_checkpoint(weights_dir: Path, prefix: str) -> Path:
    """Find the checkpoint with highest PSNR for a given model prefix."""
    candidates = sorted(weights_dir.glob(f"{prefix}_ct_*.pth"))
    if not candidates:
        script = {"fistanet": "train_fistanet", "istanet": "train_istanet",
                  "fbpconvnet": "train_fbpconvnet"}.get(prefix, f"train_{prefix}")
        raise FileNotFoundError(
            f"No checkpoint found for '{prefix}' in {weights_dir}.\n"
            f"Run  python ct/{script}.py  first."
        )
    # Parse psnr from filename: {model}_ct_ep{epoch}_psnr{xx.xx}.pth
    def psnr_from_name(p: Path) -> float:
        try:
            return float(p.stem.split("_psnr")[-1])
        except Exception:
            return -1.0
    best = max(candidates, key=psnr_from_name)
    print(f"  Auto-selected {prefix}: {best.name}  (PSNR={psnr_from_name(best):.2f} dB)")
    return best


def load_fistanet(ckpt_path: Path, eff_size: int, device: str) -> FISTANet:
    model = FISTANet(torch.eye(1), n_stages=FISTA_NET["n_stages"],
                     n_filters=FISTA_NET["n_filters"], image_size=eff_size).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_istanet(ckpt_path: Path, eff_size: int, device: str) -> ISTANet:
    model = ISTANet(torch.eye(1), n_stages=ISTA_NET["n_stages"],
                    n_filters=ISTA_NET["n_filters"], image_size=eff_size).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_fbpconvnet(ckpt_path: Path, device: str) -> FBPConvNet:
    model = FBPConvNet(base_ch=FBPCONVNET["base_ch"]).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(fista_model, ista_model, fbpc_model, radon_op,
                 test_loader, angles, args, device):
    """
    Returns results dict:  method → {'PSNR': [], 'SSIM': [], 'RMSE': []}
    """
    methods = ["FBP", "FISTA-TV", "ISTA-Net", "FBPConvNet", "FISTA-Net"]
    res     = {m: {"PSNR": [], "SSIM": [], "RMSE": []} for m in methods}

    for fbp, sino, gt in tqdm(test_loader, desc="Evaluating"):
        fbp_np  = fbp.squeeze(1).numpy()
        sino_np = sino.squeeze(1).numpy()
        gt_np   = gt.squeeze(1).numpy()

        # FBP ──────────────────────────────────────────────────────────────────
        for p, g in zip(fbp_np, gt_np):
            m = compute_metrics(p, g)
            for k in ("PSNR", "SSIM", "RMSE"):
                res["FBP"][k].append(m[k])

        # FISTA-TV ─────────────────────────────────────────────────────────────
        for p_fbp, s_np, g in zip(fbp_np, sino_np, gt_np):
            tv = fista_tv_ct(p_fbp, s_np, angles,
                             n_iters=args.fista_tv_iters,
                             lam=CLASSICAL["fista_tv_ct_lam"],
                             step=CLASSICAL["fista_tv_ct_step"])
            m  = compute_metrics(tv, g)
            for k in ("PSNR", "SSIM", "RMSE"):
                res["FISTA-TV"][k].append(m[k])

        # ISTA-Net ─────────────────────────────────────────────────────────────
        with torch.no_grad():
            x_ista, _ = run_ista_ct(ista_model, fbp, sino, radon_op, device)
        for p, g in zip(x_ista.squeeze(1).cpu().numpy(), gt_np):
            m = compute_metrics(p, g)
            for k in ("PSNR", "SSIM", "RMSE"):
                res["ISTA-Net"][k].append(m[k])

        # FBPConvNet ───────────────────────────────────────────────────────────
        with torch.no_grad():
            fbpc_pred = fbpc_model(fbp.to(device)).squeeze(1).cpu().numpy()
        for p, g in zip(fbpc_pred, gt_np):
            m = compute_metrics(p, g)
            for k in ("PSNR", "SSIM", "RMSE"):
                res["FBPConvNet"][k].append(m[k])

        # FISTA-Net ────────────────────────────────────────────────────────────
        with torch.no_grad():
            x_fista, _ = run_fista_ct(fista_model, fbp, sino, radon_op, device)
        for p, g in zip(x_fista.squeeze(1).cpu().numpy(), gt_np):
            m = compute_metrics(p, g)
            for k in ("PSNR", "SSIM", "RMSE"):
                res["FISTA-Net"][k].append(m[k])

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_figure(display_rows, save_path):
    methods = ["FBP", "FISTA-TV", "ISTA-Net", "FBPConvNet", "FISTA-Net", "GT"]
    N, M    = len(display_rows), len(methods)
    fig, axes = plt.subplots(N, M, figsize=(3.5 * M, 3.5 * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    for col, name in enumerate(methods):
        axes[0, col].set_title(name, fontsize=11,
                               fontweight="bold" if name == "FISTA-Net" else "normal",
                               color="red"   if name == "FISTA-Net" else "black")
    for row, sample in enumerate(display_rows):
        vmin, vmax = sample["GT"].min(), sample["GT"].max()
        for col, name in enumerate(methods):
            ax  = axes[row, col]
            img = sample[name]
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if name not in ("GT",):
                m = sample["metrics"][name]
                ax.set_xlabel(
                    f"PSNR:{m['PSNR']:.2f} SSIM:{m['SSIM']:.3f}",
                    fontsize=7, labelpad=2)

    plt.suptitle("CT Reconstruction — All Methods (Mayo Clinic, 60-view)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def save_error_map_figure(display_rows, save_path):
    methods = ["FBP", "FISTA-TV", "ISTA-Net", "FBPConvNet", "FISTA-Net"]
    sample  = display_rows[0]
    gt_img  = sample["GT"]
    err_max = max(np.abs(sample[m] - gt_img).max() for m in methods)

    fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8))
    vmin, vmax = gt_img.min(), gt_img.max()

    for col, mname in enumerate(methods):
        pred    = sample[mname]
        err_map = np.abs(pred - gt_img)
        met     = sample["metrics"][mname]

        axes[0, col].imshow(pred,    cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(mname, fontsize=11,
                               fontweight="bold" if mname == "FISTA-Net" else "normal",
                               color="red" if mname == "FISTA-Net" else "black")
        axes[0, col].set_xlabel(
            f"PSNR:{met['PSNR']:.2f}dB  SSIM:{met['SSIM']:.4f}", fontsize=8)
        axes[0, col].axis("off")

        im = axes[1, col].imshow(err_map, cmap="hot", vmin=0, vmax=err_max)
        axes[1, col].set_title(f"|Error| max={err_map.max():.4f}", fontsize=9)
        axes[1, col].axis("off")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("Reconstruction", fontsize=10)
    axes[1, 0].set_ylabel("|Error Map|",    fontsize=10)
    plt.suptitle("Reconstruction vs Error Maps — CT (Mayo Clinic, 60-view)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def save_bar_chart(results, save_path):
    methods = list(results.keys())
    colors  = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    metrics = [("PSNR", "PSNR (dB)", True), ("SSIM", "SSIM", True), ("RMSE", "RMSE", False)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, ylabel, higher) in zip(axes, metrics):
        means = [np.mean(results[m][metric]) for m in methods]
        stds  = [np.std( results[m][metric]) for m in methods]
        bars  = ax.bar(methods, means, yerr=stds, capsize=5,
                       color=colors, edgecolor="k", linewidth=0.8, alpha=0.88)
        bars[-1].set_edgecolor("darkred"); bars[-1].set_linewidth(2)
        ax.set_title(ylabel, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
        ax.text(0.98, 0.98, "↑ higher" if higher else "↓ lower",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
        ax.grid(axis="y", alpha=0.3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("CT Reconstruction Metrics — Mayo Clinic 60-view Sparse CT", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def save_learned_params_figure(fista_model, save_path):
    mus, thetas, rhos = fista_model.get_learned_params()
    stages = list(range(1, len(mus) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, vals, style) in zip(
        axes,
        [("Step size μ", mus, "b-o"), ("Threshold θ", thetas, "r-s"), ("Momentum ρ", rhos, "g-^")]
    ):
        ax.plot(stages, vals, style, markersize=7, linewidth=2)
        ax.set_xlabel("Stage k"); ax.set_ylabel(label)
        ax.set_title(f"Learned {label}"); ax.set_xticks(stages); ax.grid(alpha=0.4)
    plt.suptitle("FISTA-Net Learned Parameter Schedules (CT)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = DEVICE
    CT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (CT_RESULTS_DIR / "figures").mkdir(exist_ok=True)
    (CT_RESULTS_DIR / "tables").mkdir(exist_ok=True)

    # ── Test loader (full slices, batch=1) ────────────────────────────────────
    token = args.box_token or CT["box_token"] or None
    _, _, test_loader = build_ct_loaders(
        box_token  = token,
        data_root  = CT_DATA_DIR,
        n_views    = args.n_views,
        patch_size = None,
        batch_size = 1,
    )

    # Effective image size from first test batch
    fbp0, _, _ = next(iter(test_loader))
    eff_size   = fbp0.shape[-1]
    angles     = np.linspace(0.0, 180.0, args.n_views, endpoint=False)
    radon_op   = RadonOperator(image_size=eff_size, n_views=args.n_views)

    # ── Load checkpoints ──────────────────────────────────────────────────────
    print("\nLoading model checkpoints …")
    fista_ckpt = Path(args.fista) if args.fista else find_best_checkpoint(CT_WEIGHTS_DIR, "fistanet")
    ista_ckpt  = Path(args.ista)  if args.ista  else find_best_checkpoint(CT_WEIGHTS_DIR, "istanet")
    fbpc_ckpt  = Path(args.fbpc)  if args.fbpc  else find_best_checkpoint(CT_WEIGHTS_DIR, "fbpconvnet")

    fista_model = load_fistanet(fista_ckpt, eff_size, device)
    ista_model  = load_istanet( ista_ckpt,  eff_size, device)
    fbpc_model  = load_fbpconvnet(fbpc_ckpt, device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nRunning evaluation …")
    results = evaluate_all(fista_model, ista_model, fbpc_model,
                           radon_op, test_loader, angles, args, device)

    # ── Print table ───────────────────────────────────────────────────────────
    print_results_table(results, title=f"CT Sparse-View Reconstruction ({args.n_views} views, Mayo Clinic)")

    # ── Save tables ───────────────────────────────────────────────────────────
    save_results_csv(results,         CT_RESULTS_DIR / "tables" / "ct_per_sample.csv")
    save_results_summary_csv(results, CT_RESULTS_DIR / "tables" / "ct_summary.csv")

    # ── Collect display samples ───────────────────────────────────────────────
    display_rows = []
    with torch.no_grad():
        for fbp, sino, gt in test_loader:
            if len(display_rows) >= args.n_display:
                break
            fbp_np  = fbp.squeeze(1).numpy()
            sino_np = sino.squeeze(1).numpy()
            gt_np   = gt.squeeze(1).numpy()

            x_fista, _ = run_fista_ct(fista_model, fbp, sino, radon_op, device)
            x_ista,  _ = run_ista_ct( ista_model,  fbp, sino, radon_op, device)
            fbpc_pred  = fbpc_model(fbp.to(device)).squeeze(1).cpu().numpy()

            for i in range(fbp.shape[0]):
                if len(display_rows) >= args.n_display:
                    break
                fbp_i  = fbp_np[i];  sino_i = sino_np[i]; gt_i = gt_np[i]
                tv_i   = fista_tv_ct(fbp_i, sino_i, angles, n_iters=args.fista_tv_iters)
                dr     = max(gt_i.max() - gt_i.min(), 1e-8)
                row    = {
                    "FBP":       fbp_i,
                    "FISTA-TV":  tv_i,
                    "ISTA-Net":  x_ista[i, 0].cpu().numpy(),
                    "FBPConvNet":fbpc_pred[i],
                    "FISTA-Net": x_fista[i, 0].cpu().numpy(),
                    "GT":        gt_i,
                }
                row["metrics"] = {
                    m: compute_metrics(row[m], gt_i)
                    for m in ["FBP", "FISTA-TV", "ISTA-Net", "FBPConvNet", "FISTA-Net"]
                }
                display_rows.append(row)

    # ── Save figures ──────────────────────────────────────────────────────────
    fig_dir = CT_RESULTS_DIR / "figures"
    save_comparison_figure(display_rows, fig_dir / "ct_comparison_all_methods.png")
    save_error_map_figure( display_rows, fig_dir / "ct_error_maps.png")
    save_bar_chart(results,              fig_dir / "ct_metrics_barchart.png")
    save_learned_params_figure(fista_model, fig_dir / "ct_learned_params.png")

    # ── Final summary ─────────────────────────────────────────────────────────
    means = {m: {k: float(np.mean(v)) for k, v in mets.items()} for m, mets in results.items()}
    best_psnr_m = max(means, key=lambda m: means[m]["PSNR"])
    delta = means["FISTA-Net"]["PSNR"] - means["FBP"]["PSNR"]
    print(f"\n  Best PSNR : {best_psnr_m} ({means[best_psnr_m]['PSNR']:.3f} dB)")
    print(f"  FISTA-Net gain over FBP: ΔPSNR = {delta:+.3f} dB")
    print(f"\nAll results saved to {CT_RESULTS_DIR}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all CT methods and save results")
    p.add_argument("--fista",           type=str,   default=None,
                   help="Path to FISTA-Net checkpoint (auto-selects best if omitted)")
    p.add_argument("--ista",            type=str,   default=None)
    p.add_argument("--fbpc",            type=str,   default=None)
    p.add_argument("--n_views",         type=int,   default=CT["n_views"])
    p.add_argument("--n_display",       type=int,   default=EVAL["n_display"])
    p.add_argument("--fista_tv_iters",  type=int,   default=CLASSICAL["fista_tv_ct_iters"])
    p.add_argument("--box_token",       type=str,   default="",
                   help="Box developer token (uses cached slices if already fetched)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
