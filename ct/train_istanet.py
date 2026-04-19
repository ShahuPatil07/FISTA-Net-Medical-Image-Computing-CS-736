"""
ct/train_istanet.py
===================
Train ISTA-Net (no-momentum baseline) on Mayo Clinic sparse-view CT.

Usage
-----
    python ct/train_istanet.py
    python ct/train_istanet.py --n_epochs 20 --lr_net 5e-5

Saves checkpoints to:
    ct/weights/istanet_ct_ep{epoch:03d}_psnr{psnr:.2f}.pth
"""

import argparse, os, sys, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config          import CT, ISTA_NET, TRAIN, DEVICE, CT_WEIGHTS_DIR, CT_DATA_DIR, BOX_TOKEN, weight_name
from ct.dataset      import build_ct_loaders, RadonOperator
from ct.train_fistanet import fista_net_loss
from shared.models   import ISTANet
from shared.metrics  import compute_metrics


def run_ista_ct(model, fbp, sino, radon_op, device):
    fbp    = fbp.to(device)
    sino   = sino.to(device)
    x      = fbp
    ints   = []
    for k in range(model.Ns):
        mu    = model.mus[k]
        theta = model.thetas[k]
        Ax    = radon_op.batch_forward(x).to(device)
        r     = x - mu * radon_op.batch_adjoint(Ax - sino).to(device)
        x, z  = model.prox_net(r, theta)
        ints.append((x, z))
    return x, ints


def train(args):
    device      = DEVICE
    weights_dir = CT_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    eff_size = args.patch_size or CT["image_size"]
    radon_op = RadonOperator(image_size=eff_size, n_views=args.n_views)

    print(f"Device    : {device}")
    print(f"Patch size: {eff_size}×{eff_size}")
    print(f"Epochs    : {args.n_epochs}")

    token = args.box_token or BOX_TOKEN or None
    train_loader, val_loader, _ = build_ct_loaders(
        box_token  = token,
        data_root  = CT_DATA_DIR,
        n_views    = args.n_views,
        patch_size = args.patch_size,
        batch_size = args.batch_size,
    )

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    model = ISTANet(torch.eye(1), n_stages=ISTA_NET["n_stages"],
                    n_filters=ISTA_NET["n_filters"], image_size=eff_size).to(device)
    print(f"ISTA-Net params: {model.n_parameters():,}")

    optimizer = torch.optim.Adam([
        {"params": list(model.prox_net.parameters()), "lr": args.lr_net},
        {"params": list(model.mus) + list(model.thetas), "lr": args.lr_params},
    ])

    history    = {"train_loss": [], "val_psnr": []}
    best_psnr  = -float("inf")
    best_state = None

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        ep_loss = 0.0
        for fbp, sino, gt in tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}", leave=False):
            gt = gt.to(device)
            x, ints = run_ista_ct(model, fbp, sino, radon_op, device)
            loss, *_ = fista_net_loss(x, ints, gt, model.prox_net,
                                      args.lambda1, args.lambda2)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            optimizer.step()
            ep_loss += loss.item()

        model.eval()
        val_psnrs = []
        with torch.no_grad():
            for fbp, sino, gt in val_loader:
                x, _ = run_ista_ct(model, fbp, sino, radon_op, device)
                for p, g in zip(x.squeeze(1).cpu().numpy(), gt.squeeze(1).numpy()):
                    val_psnrs.append(compute_metrics(p, g)["PSNR"])

        avg_vp = float(np.mean(val_psnrs))
        history["train_loss"].append(ep_loss / len(train_loader))
        history["val_psnr"].append(avg_vp)
        print(f"Epoch {epoch:3d} | Loss: {ep_loss/len(train_loader):.5f} | Val PSNR: {avg_vp:.2f} dB")

        if avg_vp > best_psnr:
            best_psnr  = avg_vp
            best_state = copy.deepcopy(model.state_dict())

        if epoch % TRAIN["save_every"] == 0 or epoch == args.n_epochs:
            ckpt = weights_dir / weight_name("istanet", "ct", epoch, avg_vp)
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_psnr": avg_vp, "history": history}, ckpt)
            print(f"  Saved → {ckpt.name}")

    model.load_state_dict(best_state)
    best_ckpt = weights_dir / weight_name("istanet", "ct", args.n_epochs, best_psnr)
    torch.save({"epoch": args.n_epochs, "state_dict": model.state_dict(),
                "val_psnr": best_psnr, "history": history}, best_ckpt)
    print(f"\nDone. Best Val PSNR: {best_psnr:.2f} dB → {best_ckpt}")
    return model, history


def parse_args():
    p = argparse.ArgumentParser(description="Train ISTA-Net on Mayo Clinic CT")
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
