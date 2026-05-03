"""
emt/train_fbpconvnet.py
=======================
Train FBPConvNet (U-Net) on the pre-generated EMT dataset.

For EMT, FBPConvNet takes the pre-computed initial estimate x0 (stored in
the dataset) as input and refines it — analogous to post-processing the
FBP reconstruction in CT.

Usage
-----
    python emt/train_fbpconvnet.py
    python emt/train_fbpconvnet.py --n_epochs 50 --lr_net 5e-5

Saves to: emt/weights/fbpconvnet_emt_ep{epoch:03d}_psnr{psnr:.2f}.pth
"""

import argparse, sys, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config          import EMT, FBPCONVNET, TRAIN, DEVICE, EMT_WEIGHTS_DIR, EMT_DATASET_DIR, weight_name
from emt.dataset     import build_emt_loaders
from shared.models   import FBPConvNet
from shared.metrics  import compute_metrics


def train(args):
    device      = DEVICE
    weights_dir = EMT_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Epochs : {args.n_epochs}")
    print(f"LR     : {args.lr_net}")

    train_loader, val_loader, _ = build_emt_loaders(
        data_dir   = EMT_DATASET_DIR,
        batch_size = args.batch_size,
    )

    model = FBPConvNet(base_ch=FBPCONVNET["base_ch"]).to(device)
    print(f"FBPConvNet EMT params: {model.n_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_net)

    history   = {"train_loss": [], "val_psnr": []}
    best_psnr = -float("inf")
    best_state = None

    for epoch in range(1, args.n_epochs + 1):
        model.train(); ep_loss = 0.0
        for b, x0, x_gt in tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}", leave=False):
            x0, x_gt = x0.to(device), x_gt.to(device)
            pred = model(x0)   # refine the initial estimate
            loss = F.mse_loss(pred, x_gt)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            optimizer.step(); ep_loss += loss.item()

        model.eval(); val_psnrs = []
        with torch.no_grad():
            for b, x0, x_gt in val_loader:
                x0, x_gt = x0.to(device), x_gt.to(device)
                pred = model(x0)
                for p, g in zip(pred.squeeze(1).cpu().numpy(),
                                x_gt.squeeze(1).cpu().numpy()):
                    val_psnrs.append(compute_metrics(p, g)["PSNR"])

        avg_vp = float(np.mean(val_psnrs))
        history["train_loss"].append(ep_loss / len(train_loader))
        history["val_psnr"].append(avg_vp)
        print(f"Epoch {epoch:3d} | Loss: {ep_loss/len(train_loader):.5f} | Val PSNR: {avg_vp:.2f} dB")

        if avg_vp > best_psnr:
            best_psnr  = avg_vp
            best_state = copy.deepcopy(model.state_dict())

        if epoch % TRAIN["save_every"] == 0 or epoch == args.n_epochs:
            ckpt = weights_dir / weight_name("fbpconvnet", "emt", epoch, avg_vp)
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_psnr": avg_vp, "history": history}, ckpt)
            print(f"  Saved → {ckpt.name}")

    model.load_state_dict(best_state)
    best_ckpt = weights_dir / weight_name("fbpconvnet", "emt", args.n_epochs, best_psnr)
    torch.save({"epoch": args.n_epochs, "state_dict": model.state_dict(),
                "val_psnr": best_psnr, "history": history}, best_ckpt)
    print(f"\nDone. Best Val PSNR: {best_psnr:.2f} dB → {best_ckpt}")
    return model, history


def parse_args():
    p = argparse.ArgumentParser(description="Train FBPConvNet on EMT data")
    p.add_argument("--n_epochs",   type=int,   default=TRAIN["n_epochs_emt"])
    p.add_argument("--lr_net",     type=float, default=TRAIN["lr_net_emt"])
    p.add_argument("--img_size",   type=int,   default=EMT["img_size"])
    p.add_argument("--batch_size", type=int,   default=EMT["batch_size"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
