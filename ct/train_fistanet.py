"""
ct/train_fistanet.py
====================
Train FISTA-Net on the Mayo Clinic sparse-view CT dataset.

Usage
-----
    python ct/train_fistanet.py                          # use config.py defaults
    python ct/train_fistanet.py --n_epochs 20            # override epochs
    python ct/train_fistanet.py --n_epochs 20 --lr_net 5e-5 --patch_size 64

Saves checkpoints to:
    ct/weights/fistanet_ct_ep{epoch:03d}_psnr{psnr:.2f}.pth
"""

import argparse, os, sys, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config          import CT, FISTA_NET, TRAIN, DEVICE, CT_WEIGHTS_DIR, CT_DATA_DIR, BOX_TOKEN, weight_name
from ct.dataset      import build_ct_loaders, RadonOperator
from shared.models   import FISTANet
from shared.metrics  import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Loss  (Eq. 14 of paper)
# ─────────────────────────────────────────────────────────────────────────────

def fista_net_loss(x_final, intermediates, x_gt, prox_net,
                   lambda1=TRAIN["lambda1"], lambda2=TRAIN["lambda2"]):
    """L_total = L_mse + λ1·L_sym + λ2·L_spa"""
    L_mse = F.mse_loss(x_final, x_gt)
    L_sym = L_spa = torch.tensor(0.0, device=x_final.device)
    for x_k, z_k in intermediates:
        L_sym += F.mse_loss(prox_net.decode(prox_net.encode(x_k)), x_k)
        L_spa += z_k.abs().mean()
    n = len(intermediates)
    return L_mse + lambda1 * L_sym / n + lambda2 * L_spa / n, L_mse, L_sym / n, L_spa / n


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch — inference helper (used by val loop and callers)
# ─────────────────────────────────────────────────────────────────────────────

def run_fista_ct(model, fbp, sino, radon_op, device):
    """Forward pass using RadonOperator (CT-specific)."""
    fbp    = fbp.to(device)
    sino   = sino.to(device)
    x_prev = y = fbp
    ints   = []
    for k in range(1, model.Ns + 1):
        mu    = model._mu(k)
        theta = model._theta(k)
        rho   = model._rho(k)
        Ay    = radon_op.batch_forward(y).to(device)
        r     = y - mu * radon_op.batch_adjoint(Ay - sino).to(device)
        x, z  = model.prox_net(r, theta)
        y     = x + rho * (x - x_prev)
        x_prev = x
        ints.append((x, z))
    return x, ints


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device      = DEVICE
    weights_dir = CT_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    eff_size = args.patch_size or CT["image_size"]
    radon_op = RadonOperator(image_size=eff_size, n_views=args.n_views)

    print(f"Device       : {device}")
    print(f"Image size   : {eff_size}×{eff_size}  (patch_size={args.patch_size})")
    print(f"Views        : {args.n_views}")
    print(f"Epochs       : {args.n_epochs}")
    print(f"LR net/params: {args.lr_net} / {args.lr_params}")

    # ── Data ──────────────────────────────────────────────────────────────────
    token = args.box_token or BOX_TOKEN or None
    train_loader, val_loader, _ = build_ct_loaders(
        box_token   = token,
        data_root   = CT_DATA_DIR,
        n_views     = args.n_views,
        patch_size  = args.patch_size,
        batch_size  = args.batch_size,
        num_workers = CT["num_workers"],
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    model = FISTANet(torch.eye(1), n_stages=FISTA_NET["n_stages"],
                     n_filters=FISTA_NET["n_filters"], image_size=eff_size).to(device)
    print(f"FISTA-Net params: {model.n_parameters():,}")

    # ── Optimiser  (two param groups: network vs scalar params) ───────────────
    optimizer = torch.optim.Adam([
        {"params": list(model.prox_net.parameters()), "lr": args.lr_net},
        {"params": [model.w1, model.c1, model.w2, model.c2, model.w3, model.c3],
         "lr": args.lr_params},
    ])

    history    = {"train_loss": [], "val_psnr": []}
    best_psnr  = -float("inf")
    best_state = None

    for epoch in range(1, args.n_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        ep_loss = 0.0
        for fbp, sino, gt in tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}", leave=False):
            gt   = gt.to(device)
            x, ints = run_fista_ct(model, fbp, sino, radon_op, device)
            loss, *_ = fista_net_loss(x, ints, gt, model.prox_net,
                                      args.lambda1, args.lambda2)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            optimizer.step()
            ep_loss += loss.item()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_psnrs = []
        with torch.no_grad():
            for fbp, sino, gt in val_loader:
                x, _ = run_fista_ct(model, fbp, sino, radon_op, device)
                for p, g in zip(x.squeeze(1).cpu().numpy(),
                                gt.squeeze(1).numpy()):
                    m = compute_metrics(p, g)
                    val_psnrs.append(m["PSNR"])

        avg_vp = float(np.mean(val_psnrs))
        history["train_loss"].append(ep_loss / len(train_loader))
        history["val_psnr"].append(avg_vp)
        print(f"Epoch {epoch:3d} | Loss: {ep_loss/len(train_loader):.5f} | Val PSNR: {avg_vp:.2f} dB")

        if avg_vp > best_psnr:
            best_psnr  = avg_vp
            best_state = copy.deepcopy(model.state_dict())

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % TRAIN["save_every"] == 0 or epoch == args.n_epochs:
            ckpt = weights_dir / weight_name("fistanet", "ct", epoch, avg_vp)
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_psnr": avg_vp, "history": history}, ckpt)
            print(f"  Saved checkpoint → {ckpt.name}")

    # ── Save best ─────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    best_ckpt = weights_dir / weight_name("fistanet", "ct", args.n_epochs, best_psnr)
    torch.save({"epoch": args.n_epochs, "state_dict": model.state_dict(),
                "val_psnr": best_psnr, "history": history}, best_ckpt)
    print(f"\nTraining done.  Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Best weights   → {best_ckpt}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train FISTA-Net on Mayo Clinic CT")
    p.add_argument("--n_epochs",   type=int,   default=TRAIN["n_epochs_ct"])
    p.add_argument("--lr_net",     type=float, default=TRAIN["lr_net_ct"])
    p.add_argument("--lr_params",  type=float, default=TRAIN["lr_params_ct"])
    p.add_argument("--lambda1",    type=float, default=TRAIN["lambda1"])
    p.add_argument("--lambda2",    type=float, default=TRAIN["lambda2"])
    p.add_argument("--n_views",    type=int,   default=CT["n_views"])
    p.add_argument("--batch_size", type=int,   default=CT["batch_size"])
    p.add_argument("--patch_size", type=int,   default=CT["patch_size"])
    p.add_argument("--box_token",  type=str,   default="",
                   help="Box developer token (streams slices from Box — no full download)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
