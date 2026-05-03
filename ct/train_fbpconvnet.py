import argparse, os, sys, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config          import CT, FBPCONVNET, TRAIN, DEVICE, CT_WEIGHTS_DIR, CT_DATA_DIR, weight_name
from ct.dataset      import build_ct_loaders
from shared.models   import FBPConvNet
from shared.metrics  import compute_metrics


def train(args):
    device      = DEVICE
    weights_dir = CT_WEIGHTS_DIR
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Epochs : {args.n_epochs}")
    print(f"LR     : {args.lr_net}")

    token = args.box_token or CT["box_token"] or None
    train_loader, val_loader, _ = build_ct_loaders(
        box_token   = token,
        data_root   = CT_DATA_DIR,
        n_views     = args.n_views,
        patch_size  = args.patch_size,
        batch_size  = args.batch_size,
        num_workers = CT["num_workers"],
    )

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    model = FBPConvNet(base_ch=FBPCONVNET["base_ch"]).to(device)
    print(f"FBPConvNet params: {model.n_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_net)

    history    = {"train_loss": [], "val_psnr": []}
    best_psnr  = -float("inf")
    best_state = None

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        ep_loss = 0.0
        for fbp, sino, gt in tqdm(train_loader, desc=f"Epoch {epoch}/{args.n_epochs}", leave=False):
            fbp, gt = fbp.to(device), gt.to(device)
            pred    = model(fbp)
            loss    = F.mse_loss(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            optimizer.step()
            ep_loss += loss.item()

        model.eval()
        val_psnrs = []
        with torch.no_grad():
            for fbp, sino, gt in val_loader:
                fbp, gt = fbp.to(device), gt.to(device)
                pred    = model(fbp)
                for p, g in zip(pred.squeeze(1).cpu().numpy(), gt.squeeze(1).cpu().numpy()):
                    val_psnrs.append(compute_metrics(p, g)["PSNR"])

        avg_vp = float(np.mean(val_psnrs))
        history["train_loss"].append(ep_loss / len(train_loader))
        history["val_psnr"].append(avg_vp)
        print(f"Epoch {epoch:3d} | Loss: {ep_loss/len(train_loader):.5f} | Val PSNR: {avg_vp:.2f} dB")

        if avg_vp > best_psnr:
            best_psnr  = avg_vp
            best_state = copy.deepcopy(model.state_dict())

        if epoch % TRAIN["save_every"] == 0 or epoch == args.n_epochs:
            ckpt = weights_dir / weight_name("fbpconvnet", "ct", epoch, avg_vp)
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_psnr": avg_vp, "history": history}, ckpt)
            print(f"  Saved → {ckpt.name}")

    model.load_state_dict(best_state)
    best_ckpt = weights_dir / weight_name("fbpconvnet", "ct", args.n_epochs, best_psnr)
    torch.save({"epoch": args.n_epochs, "state_dict": model.state_dict(),
                "val_psnr": best_psnr, "history": history}, best_ckpt)
    print(f"\nDone. Best Val PSNR: {best_psnr:.2f} dB → {best_ckpt}")
    return model, history


def parse_args():
    p = argparse.ArgumentParser(description="Train FBPConvNet on Mayo Clinic CT")
    p.add_argument("--n_epochs",   type=int,   default=TRAIN["n_epochs_ct"])
    p.add_argument("--lr_net",     type=float, default=TRAIN["lr_net_ct"])
    p.add_argument("--n_views",    type=int,   default=CT["n_views"])
    p.add_argument("--batch_size", type=int,   default=CT["batch_size"])
    p.add_argument("--patch_size", type=int,   default=CT["patch_size"])
    p.add_argument("--box_token",  type=str,   default="",
                   help="Box developer token (streams slices from Box — no full download)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
