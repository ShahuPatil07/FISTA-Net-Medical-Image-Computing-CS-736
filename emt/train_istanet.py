import argparse, sys, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config          import EMT, ISTA_NET, TRAIN, DEVICE, EMT_WEIGHTS_DIR, EMT_DATASET_DIR, weight_name
from emt.dataset     import build_emt_loaders, load_sensitivity_matrix
from emt.train_fistanet import fista_net_loss
from shared.models   import ISTANet
from shared.metrics  import compute_metrics


def train(args):
    device      = DEVICE
    weights_dir = EMT_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_emt_loaders(
        data_dir   = EMT_DATASET_DIR,
        batch_size = args.batch_size,
    )

    A_np     = load_sensitivity_matrix(EMT_DATASET_DIR)
    A_tensor = torch.tensor(A_np, dtype=torch.float32)
    print(f"A shape: {A_np.shape}")

    model = ISTANet(
        A_tensor,
        n_stages   = ISTA_NET["n_stages"],
        n_filters  = ISTA_NET["n_filters"],
        image_size = args.img_size,
    ).to(device)
    print(f"ISTA-Net EMT params: {model.n_parameters():,}")

    optimizer = torch.optim.Adam([
        {"params": list(model.prox_net.parameters()), "lr": args.lr_net},
        {"params": list(model.mus) + list(model.thetas), "lr": args.lr_params},
    ])

    history   = {"train_loss": [], "val_psnr": []}
    best_psnr = -float("inf")
    best_state = None

    for epoch in range(1, args.n_epochs + 1):
        model.train(); ep_loss = 0.0
        for b, x0, x_gt in tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}", leave=False):
            b, x0, x_gt = b.to(device), x0.to(device), x_gt.to(device)
            x_final, ints = model(b, x0)
            loss, *_ = fista_net_loss(x_final, ints, x_gt, model.prox_net,
                                      args.lambda1, args.lambda2)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            optimizer.step(); ep_loss += loss.item()

        model.eval(); val_psnrs = []
        with torch.no_grad():
            for b, x0, x_gt in val_loader:
                b, x0, x_gt = b.to(device), x0.to(device), x_gt.to(device)
                x_final, _ = model(b, x0)
                for p, g in zip(x_final.squeeze(1).cpu().numpy(),
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
            ckpt = weights_dir / weight_name("istanet", "emt", epoch, avg_vp)
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_psnr": avg_vp, "history": history}, ckpt)
            print(f"  Saved → {ckpt.name}")

    model.load_state_dict(best_state)
    best_ckpt = weights_dir / weight_name("istanet", "emt", args.n_epochs, best_psnr)
    torch.save({"epoch": args.n_epochs, "state_dict": model.state_dict(),
                "val_psnr": best_psnr, "history": history}, best_ckpt)
    print(f"\nDone. Best Val PSNR: {best_psnr:.2f} dB → {best_ckpt}")
    return model, history


def parse_args():
    p = argparse.ArgumentParser(description="Train ISTA-Net on EMT data")
    p.add_argument("--n_epochs",   type=int,   default=TRAIN["n_epochs_emt"])
    p.add_argument("--lr_net",     type=float, default=TRAIN["lr_net_emt"])
    p.add_argument("--lr_params",  type=float, default=TRAIN["lr_params_emt"])
    p.add_argument("--lambda1",    type=float, default=TRAIN["lambda1"])
    p.add_argument("--lambda2",    type=float, default=TRAIN["lambda2"])
    p.add_argument("--img_size",   type=int,   default=EMT["img_size"])
    p.add_argument("--batch_size", type=int,   default=EMT["batch_size"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
