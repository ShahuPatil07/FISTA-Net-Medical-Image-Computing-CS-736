import argparse, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config         import (EMT, FISTA_NET, ISTA_NET, FBPCONVNET, CLASSICAL, EVAL,
                             DEVICE, EMT_WEIGHTS_DIR, EMT_RESULTS_DIR, EMT_DATASET_DIR)
from emt.dataset    import build_emt_loaders, load_sensitivity_matrix, load_laplacian
from emt.baselines  import laplacian_regularization, fista_tv_emt
from shared.models  import FISTANet, ISTANet, FBPConvNet
from shared.metrics import (compute_metrics, print_results_table,
                             save_results_csv, save_results_summary_csv)


def find_best_checkpoint(weights_dir: Path, prefix: str) -> Path:
    candidates = sorted(weights_dir.glob(f"{prefix}_emt_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint for '{prefix}' in {weights_dir}")
    def psnr_from(p):
        try: return float(p.stem.split("_psnr")[-1])
        except: return -1.0
    best = max(candidates, key=psnr_from)
    print(f"  Auto-selected {prefix}: {best.name}")
    return best


def load_fistanet(ckpt_path, A_tensor, img_size, device):
    model = FISTANet(A_tensor, n_stages=FISTA_NET["n_stages"],
                     n_filters=FISTA_NET["n_filters"], image_size=img_size).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return model


def load_istanet(ckpt_path, A_tensor, img_size, device):
    model = ISTANet(A_tensor, n_stages=ISTA_NET["n_stages"],
                    n_filters=ISTA_NET["n_filters"], image_size=img_size).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return model


def load_fbpconvnet(ckpt_path, device):
    model = FBPConvNet(base_ch=FBPCONVNET["base_ch"]).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return model


def evaluate_all(fista_model, ista_model, fbpc_model,
                 test_loader, A_np, L_np, img_size, device):
    methods = ["Lap.Reg", "FISTA-TV", "ISTA-Net", "FBPConvNet", "FISTA-Net"]
    res     = {m: {"PSNR": [], "SSIM": [], "RMSE": []} for m in methods}

    for b_batch, x0_batch, gt_batch in tqdm(test_loader, desc="Evaluating EMT"):
        gt_np = gt_batch.squeeze(1).numpy()
        b_np  = b_batch.numpy()

        for i in range(b_batch.shape[0]):
            b_i  = b_np[i]; gt_i = gt_np[i]
            x_lap = laplacian_regularization(b_i, A_np, L_np)
            x_tv  = fista_tv_emt(b_i, A_np, x_lap)
            for name, pred in [("Lap.Reg", x_lap), ("FISTA-TV", x_tv)]:
                m = compute_metrics(pred, gt_i)
                for k in ("PSNR", "SSIM", "RMSE"):
                    res[name][k].append(m[k])

        b_t  = b_batch.to(device)
        x0_t = x0_batch.to(device)

        with torch.no_grad():
            ista_out,  _ = ista_model(b_t, x0_t)
            fista_out, _ = fista_model(b_t, x0_t)
            fbpc_out     = fbpc_model(x0_t)

        for name, pred_t in [("ISTA-Net", ista_out), ("FBPConvNet", fbpc_out), ("FISTA-Net", fista_out)]:
            pred_np = pred_t.squeeze(1).cpu().numpy()
            for p, g in zip(pred_np, gt_np):
                m = compute_metrics(p, g)
                for k in ("PSNR", "SSIM", "RMSE"):
                    res[name][k].append(m[k])

    return res


def save_emt_comparison_figure(display_rows, save_path):
    methods   = ["Lap.Reg", "FISTA-TV", "ISTA-Net", "FBPConvNet", "FISTA-Net", "GT"]
    N, M      = len(display_rows), len(methods)
    fig, axes = plt.subplots(N, M, figsize=(3.5 * M, 3.5 * N))
    if N == 1: axes = axes[np.newaxis, :]

    for col, name in enumerate(methods):
        axes[0, col].set_title(name, fontsize=11,
                               fontweight="bold" if name == "FISTA-Net" else "normal",
                               color="red" if name == "FISTA-Net" else "black")
    for row, sample in enumerate(display_rows):
        vmin, vmax = sample["GT"].min(), sample["GT"].max()
        for col, name in enumerate(methods):
            ax = axes[row, col]
            ax.imshow(sample[name], cmap="hot", vmin=vmin, vmax=vmax)
            ax.axis("off")
            if name != "GT":
                m = sample["metrics"][name]
                ax.set_xlabel(f"PSNR:{m['PSNR']:.2f} SSIM:{m['SSIM']:.3f}", fontsize=7)

    plt.suptitle("EMT Reconstruction — All Methods", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


def run(args):
    device = DEVICE
    EMT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (EMT_RESULTS_DIR / "figures").mkdir(exist_ok=True)
    (EMT_RESULTS_DIR / "tables").mkdir(exist_ok=True)

    print(f"Test split: {args.test_split}")
    _, _, test_loader = build_emt_loaders(
        data_dir    = EMT_DATASET_DIR,
        batch_size  = EMT["batch_size"],
        test_split  = args.test_split,
    )

    A_np     = load_sensitivity_matrix(EMT_DATASET_DIR)
    L_np     = load_laplacian(EMT_DATASET_DIR)
    A_tensor = torch.tensor(A_np, dtype=torch.float32)
    img_size = EMT["img_size"]
    print(f"A: {A_np.shape}  L: {L_np.shape}")

    print("\nLoading model checkpoints …")
    fista_ckpt = Path(args.fista) if args.fista else find_best_checkpoint(EMT_WEIGHTS_DIR, "fistanet")
    ista_ckpt  = Path(args.ista)  if args.ista  else find_best_checkpoint(EMT_WEIGHTS_DIR, "istanet")
    fbpc_ckpt  = Path(args.fbpc)  if args.fbpc  else find_best_checkpoint(EMT_WEIGHTS_DIR, "fbpconvnet")

    fista_model = load_fistanet(fista_ckpt, A_tensor, img_size, device)
    ista_model  = load_istanet( ista_ckpt,  A_tensor, img_size, device)
    fbpc_model  = load_fbpconvnet(fbpc_ckpt, device)

    print("\nRunning evaluation …")
    results = evaluate_all(fista_model, ista_model, fbpc_model,
                           test_loader, A_np, L_np, img_size, device)

    tag = args.test_split
    print_results_table(results, title=f"EMT Reconstruction ({tag})")
    save_results_csv(results,
                     EMT_RESULTS_DIR / "tables" / f"emt_{tag}_per_sample.csv")
    save_results_summary_csv(results,
                     EMT_RESULTS_DIR / "tables" / f"emt_{tag}_summary.csv")

    display_rows = []
    with torch.no_grad():
        for b_batch, x0_batch, gt_batch in test_loader:
            if len(display_rows) >= args.n_display: break
            b_np  = b_batch.numpy()
            gt_np = gt_batch.squeeze(1).numpy()
            b_t   = b_batch.to(device)
            x0_t  = x0_batch.to(device)
            fista_pred = fista_model(b_t, x0_t)[0].squeeze(1).cpu().numpy()
            ista_pred  = ista_model( b_t, x0_t)[0].squeeze(1).cpu().numpy()
            fbpc_pred  = fbpc_model(x0_t).squeeze(1).cpu().numpy()
            for i in range(b_batch.shape[0]):
                if len(display_rows) >= args.n_display: break
                b_i   = b_np[i]; gt_i = gt_np[i]
                x_lap = laplacian_regularization(b_i, A_np, L_np)
                x_tv  = fista_tv_emt(b_i, A_np, x_lap)
                row = {
                    "Lap.Reg":    x_lap,
                    "FISTA-TV":   x_tv,
                    "ISTA-Net":   ista_pred[i],
                    "FBPConvNet": fbpc_pred[i],
                    "FISTA-Net":  fista_pred[i],
                    "GT":         gt_i,
                }
                row["metrics"] = {m: compute_metrics(row[m], gt_i)
                                   for m in ["Lap.Reg","FISTA-TV","ISTA-Net","FBPConvNet","FISTA-Net"]}
                display_rows.append(row)

    fig_dir = EMT_RESULTS_DIR / "figures"
    save_emt_comparison_figure(
        display_rows,
        fig_dir / f"emt_comparison_{tag}.png",
    )
    print(f"\nAll results saved to {EMT_RESULTS_DIR}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all EMT methods")
    p.add_argument("--fista",       type=str, default=None)
    p.add_argument("--ista",        type=str, default=None)
    p.add_argument("--fbpc",        type=str, default=None)
    p.add_argument("--test_split",  type=str, default="test1",
                   choices=["test1", "test2"],
                   help="Which test set to evaluate on (test1 or test2)")
    p.add_argument("--n_display",   type=int, default=EVAL["n_display"])
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
