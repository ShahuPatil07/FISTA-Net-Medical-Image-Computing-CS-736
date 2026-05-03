# FISTA-Net — CS 736 Medical Image Computing

Replication of **FISTA-Net: Learning A Fast Iterative Shrinkage Thresholding Network for Inverse Problems in Imaging** (Xiang et al., IEEE TMI 2021) for two modalities: sparse-view CT and Electrical Impedance Tomography (EMT).

---

## Project Structure

```
├── config.py                        master config — all hyperparameters and paths
├── shared/
│   ├── models.py                    FISTANet, ISTANet, FBPConvNet, ProximalMappingNetwork
│   └── metrics.py                   compute_metrics, print_results_table, CSV exports
├── ct/
│   ├── dataset.py                   BoxCTDataset, MayoCTDataset, RadonOperator, build_ct_loaders
│   ├── baselines.py                 fista_tv_ct (classical TV baseline)
│   ├── train_fistanet.py            train FISTA-Net on Mayo CT
│   ├── train_istanet.py             train ISTA-Net on Mayo CT
│   ├── train_fbpconvnet.py          train FBPConvNet on Mayo CT
│   ├── evaluate_all.py              evaluate all CT methods → tables + figures
│   ├── weights/                     saved checkpoints (auto-created)
│   ├── results/figures/             PNG comparison figures
│   ├── results/tables/              CSV metric tables
│   └── notebooks/                   01_CT_EDA.ipynb, 02_CT_Evaluate_All.ipynb
└── emt/
    ├── FISTA_Net_EMT_Dataset/       pre-generated .pt dataset files
    ├── dataset.py                   EMTDataset, build_emt_loaders, load_sensitivity_matrix
    ├── baselines.py                 laplacian_regularization, fista_tv_emt
    ├── train_fistanet.py            train FISTA-Net on EMT
    ├── train_istanet.py             train ISTA-Net on EMT
    ├── train_fbpconvnet.py          train FBPConvNet on EMT
    ├── evaluate_all.py              evaluate all EMT methods → tables + figures
    ├── weights/
    └── results/
```

---

## config.py

Single source of truth for all hyperparameters. Every training and evaluation script imports from here. To override a value for one run without editing the file, pass CLI flags (e.g. `--n_epochs 20`).

### Paths

| Variable | Default | Purpose |
|----------|---------|---------|
| `ROOT_DIR` | repo root | base for all relative paths |
| `CT_DATA_DIR` | `ct_cache/full_3mm/full_3mm` | local DICOM directory (used by MayoCTDataset) |
| `CT_WEIGHTS_DIR` | `ct/weights` | where CT checkpoints are saved |
| `CT_RESULTS_DIR` | `ct/results` | where CT figures and tables are saved |
| `EMT_DATASET_DIR` | `emt/FISTA_Net_EMT_Dataset` | pre-generated .pt files for EMT |
| `EMT_WEIGHTS_DIR` | `emt/weights` | EMT checkpoints |
| `EMT_RESULTS_DIR` | `emt/results` | EMT results |
| `DEVICE` | `"cuda"` or `"cpu"` | auto-detected |

### CT dict

Controls the CT data pipeline and dataset splits.

| Key | Value | Meaning |
|-----|-------|---------|
| `box_token` | string | Box developer token for streaming DICOM files; leave `""` for local disk mode |
| `train_patients` | list of 8 IDs | Mayo Clinic patient IDs used for training |
| `val_patients` | `["L333"]` | single patient for validation |
| `test_patients` | `["L506"]` | single patient for test |
| `slices_per_patient_train` | 30 | slices used per training patient (30×8 = 240 total) |
| `slices_per_patient_val` | 10 | slices per val patient |
| `slices_per_patient_test` | 10 | slices per test patient |
| `n_views` | 60 | number of equi-spaced Radon projection angles |
| `image_size` | 512 | full DICOM slice resolution |
| `patch_size` | 128 | random crop size during training; test uses full 512×512 |
| `win_min` / `win_max` | -140 / 260 HU | HU window mapped to [0, 1] |
| `batch_size` | 2 | training batch size |
| `num_workers` | 2 | DataLoader workers |
| `pin_memory` | True | pinned memory for GPU transfer |

### EMT dict

Controls the EMT data pipeline.

| Key | Value | Meaning |
|-----|-------|---------|
| `n_coils` | 8 | electrodes on the boundary |
| `n_meas` | 64 | total measurements (8 excitations × 8 per excitation) |
| `img_size` | 64 | output image resolution (64×64) |
| `domain_r` | 0.90 | circular domain radius |
| `sigma_bg` / `sigma_obj` | 1.0 / 3.0 S/m | background and object conductivity |
| `obj_r_min` / `obj_r_max` | 0.12 / 0.28 | range of object radii |
| `n_train` / `n_val` / `n_test` | 1920 / 480 / 1200 | dataset sizes per split |
| `train_subset` | 640 | use only first 640 training samples to reduce runtime (≈1/3 of full set) |
| `noise_db` | 30 | SNR in dB for measurement noise |
| `batch_size` | 16 | training batch size |

### FISTA_NET dict

Architecture and initialisation for the proposed method.

| Key | Value | Meaning |
|-----|-------|---------|
| `n_stages` | 7 | number of unrolled FISTA iterations |
| `n_filters` | 32 | feature channels in ProximalMappingNetwork |
| `init_w1`, `init_c1` | -0.2, 0.1 | initial affine parameters for step-size schedule μ_k |
| `init_w2`, `init_c2` | -0.5, -2.0 | initial parameters for threshold schedule θ_k |
| `init_w3`, `init_c3` | 0.5, 0.0 | initial parameters for momentum schedule ρ_k |

### ISTA_NET dict

| Key | Value | Meaning |
|-----|-------|---------|
| `n_stages` | 7 | same as FISTA-Net for fair comparison |
| `n_filters` | 32 | same ProxNet architecture |
| `init_mu` | 0.1 | initial per-stage step size |
| `init_theta` | 0.1 | initial per-stage threshold |

### FBPCONVNET dict

| Key | Value | Meaning |
|-----|-------|---------|
| `base_ch` | 32 | feature channels at the first encoder level of the U-Net |

### TRAIN dict

| Key | Value | Meaning |
|-----|-------|---------|
| `n_epochs_ct` | 20 | CT training epochs |
| `lr_net_ct` | 1e-4 | learning rate for ProxNet / U-Net weights |
| `lr_params_ct` | 1e-3 | learning rate for FISTA-Net scalar params (μ, θ, ρ) |
| `n_epochs_emt` | 50 | EMT training epochs |
| `lr_net_emt` | 1e-4 | EMT ProxNet learning rate |
| `lr_params_emt` | 1e-3 | EMT scalar param learning rate |
| `lambda1` | 0.01 | weight for symmetry loss L_sym (Eq. 14) |
| `lambda2` | 0.001 | weight for sparsity loss L_spa (Eq. 14) |
| `grad_clip` | 1.0 | gradient norm clipping threshold |
| `save_every` | 5 | checkpoint save frequency in epochs |

### CLASSICAL dict

| Key | Value | Meaning |
|-----|-------|---------|
| `fista_tv_ct_iters` | 100 | FISTA-TV outer iterations for CT |
| `fista_tv_ct_lam` | 0.005 | TV regularisation weight for CT |
| `fista_tv_ct_step` | 0.001 | gradient step size μ for CT FISTA-TV |
| `fista_tv_emt_iters` | 100 | FISTA-TV iterations for EMT |
| `fista_tv_emt_lam` | 0.001 | TV weight for EMT |
| `lap_reg_lam` | 0.001 | Laplacian regularisation weight for EMT |

### EVAL dict

| Key | Value | Meaning |
|-----|-------|---------|
| `n_display` | 3 | number of test samples shown in comparison figures |
| `figure_dpi` | 150 | output figure resolution |

### weight_name(model, modality, epoch, psnr)

Returns a standardised checkpoint filename:
```
{model}_{modality}_ep{epoch:03d}_psnr{psnr:.2f}.pth
```
Examples: `fistanet_ct_ep020_psnr28.45.pth`, `istanet_emt_ep050_psnr22.34.pth`

---

## shared/models.py

All neural network architectures shared between CT and EMT pipelines.

### ProximalMappingNetwork

4-layer encoder / 4-layer decoder CNN that implements the learned proximal operator. Weights are shared across all unrolling stages; only the threshold θ varies per stage.

- **Encoder**: 4 × Conv2d(3×3, no bias) with ReLU. The first layer projects the 1-channel input to `n_filters` channels with no activation (direct projection).
- **Decoder**: 3 × Conv2d(3×3) + ReLU, then a final Conv2d to collapse back to 1 channel.
- **Forward** `(r, theta)` → `(x_out, z_thresh)`:
  - Encodes `r` to latent `z`
  - Applies element-wise soft-thresholding: `sign(z) * max(|z| - theta, 0)`
  - Decodes `z_thresh` and adds residual skip connection with ReLU: `relu(decode(z_thresh) + r)`
  - Returns both `x_out` (proximal output) and `z_thresh` (for the sparsity loss L_spa)

### FISTANet

Unrolled FISTA with learned per-stage parameters μ_k (step size), θ_k (threshold), ρ_k (momentum). All three schedules are parameterised as monotone functions of stage index k through learnable affine + softplus transforms (paper Section III-C).

**Constructor** takes:
- `A_matrix` — sensitivity matrix. For EMT this is the (64, 4096) pixel-space matrix. For CT a dummy 1×1 eye is passed; the actual Radon forward/adjoint are applied in the training loop.
- `n_stages` — number of unrolled iterations (default 7)
- `n_filters` — ProxNet filter count
- `image_size` — spatial dimension of the image

**Parameters**: 6 learnable scalars (w1, c1, w2, c2, w3, c3) plus all ProxNet weights.

**Forward** `(b, x0)` → `(x, intermediates)`:
- Runs n_stages iterations of FISTA: gradient step using A and W_tilde (= A^T), proximal step via ProxNet, momentum update.
- Returns the final reconstruction and a list of (x_k, z_k) intermediate outputs needed for the training loss.
- This forward pass is used directly for EMT. For CT, the training loop bypasses it and calls `prox_net` + `RadonOperator` directly so that Radon transforms replace the explicit A multiplication.

**get_learned_params()**: returns three lists (mus, thetas, rhos) of the learned parameter values at each stage, used for visualisation.

### ISTANet

Same as FISTANet but without the momentum update. Each stage has its own independent scalar μ_k and θ_k (stored as ParameterLists rather than a shared functional schedule). Same number of stages as FISTANet for fair comparison.

### FBPConvNet

Standard 4-level U-Net applied as a post-processor. Takes a filtered back-projection or initial estimate image and outputs a refined reconstruction. Uses residual output: `pred = UNet(input) + input`. Each encoder/decoder block is two Conv2d(3×3) + BatchNorm + ReLU pairs. Encoder uses MaxPool2d(2) for downsampling; decoder uses ConvTranspose2d(2×2) for upsampling with skip connections from the encoder.

---

## shared/metrics.py

### compute_metrics(pred, gt)

Accepts numpy arrays of shape (H, W) or (B, H, W). For batches, returns the mean over samples.

Returns a dict with three keys:
- `PSNR` — peak signal-to-noise ratio in dB, computed with `data_range = gt.max() - gt.min()`
- `SSIM` — structural similarity index
- `RMSE` — root mean square error

### print_results_table(results, title, highlight)

Prints a formatted comparison table to stdout. `results` is a dict mapping method name to `{'PSNR': list, 'SSIM': list, 'RMSE': list}`. The `highlight` method (default `"FISTA-Net"`) is marked with ◄.

### save_results_csv(results, path)

Saves per-sample metrics for every method as a CSV. One row per test sample, one column group per method.

### save_results_summary_csv(results, path)

Saves mean ± std summary table as CSV. One row per method.

---

## ct/dataset.py

### _BoxSeekableStream

Internal class that wraps Box API HTTP range requests so Python's `zipfile` module can seek inside a remote zip without downloading the entire file. Follows the Box 302 redirect once to get the CDN URL, then issues `Range: bytes=X-Y` requests for each read. Handles token expiry (401) by refreshing the CDN URL.

### load_ima_file(path)

Reads a single `.IMA` DICOM file using pydicom, applies RescaleSlope/RescaleIntercept to get HU values, and maps the HU range `[win_min, win_max]` linearly to `[0, 1]`.

### make_sparse_sinogram(img, n_views)

Given a full-dose image and a view count, computes a sparse sinogram using skimage `radon` at `n_views` equi-spaced angles from 0° to 180°, then reconstructs an FBP initialisation using `iradon` with ramp filter. Returns `(sinogram, fbp_init)`.

### BoxCTDataset

Fetches only the needed `.IMA` slices from the Box zip using HTTP range requests — no full zip download. On first use, caches individual DICOM files to `ct_cache/slices/{patient_id}/`. Subsequent epochs read entirely from local cache; the Box token is not needed once all slices are cached.

Each call to `__getitem__` returns `(fbp, sinogram, gt)` as float32 tensors of shape `(1, H, W)`. During training, a random `patch_size × patch_size` crop is applied to all three arrays consistently. During test (patch_size=None), full 512×512 slices are returned.

### MayoCTDataset

Reads `.IMA` files from a local directory structure: `{data_root}/{patient_id}/full_3mm/*.IMA`. Use this when the data is already extracted to disk. Identical `__getitem__` interface to BoxCTDataset.

### build_ct_loaders(box_token, data_root, n_views, patch_size, batch_size, num_workers)

Convenience function that builds train, val, and test DataLoaders. Automatically picks BoxCTDataset if a token is available (checking the argument first, then `CT["box_token"]` from config), otherwise falls back to MayoCTDataset. The test loader always uses `patch_size=None` (full slices) and `batch_size=1`.

### RadonOperator

Thin numpy/skimage wrapper for batched Radon forward (`radon`) and adjoint (`iradon`) operations. Used inside the CT training loops in place of an explicit matrix A, since storing the full 512²×60 Radon matrix in memory is impractical. All operations detach from autograd; only the ProxNet weights and FISTA-Net scalar parameters are differentiated through the training loss.

---

## ct/baselines.py

### fista_tv_ct(fbp_img, sino_np, angles, n_iters, lam, step)

Classical TV-regularised CT reconstruction via FISTA. All operations are in numpy; runs per-image (not batched).

Algorithm:
1. Initialise x from the FBP image
2. At each iteration: compute gradient of data fidelity `A^T(Ax - b)` using skimage radon/iradon, take a gradient step, apply isotropic TV proximal operator (forward-difference divergence), then update the FISTA momentum variable

Returns a float32 image in [0, 1].

---

## ct/train_fistanet.py

### fista_net_loss(x_final, intermediates, x_gt, prox_net, lambda1, lambda2)

Implements the combined training loss from Eq. 14 of the paper:

`L_total = L_mse + λ1 * L_sym + λ2 * L_spa`

- `L_mse` — MSE between final reconstruction and ground truth
- `L_sym` — symmetry loss: for each stage, MSE between `decode(encode(x_k))` and `x_k` (enforces the encoder/decoder to be approximate inverses)
- `L_spa` — sparsity loss: mean absolute value of latent features z_k after soft-thresholding

Both L_sym and L_spa are averaged over all stages.

### run_fista_ct(model, fbp, sino, radon_op, device)

CT-specific FISTA forward pass that replaces the matrix-vector products with RadonOperator calls. Runs model.Ns stages manually rather than calling `model.forward()`, since the explicit A matrix stored in the model is a dummy 1×1 eye for CT. Returns `(x_final, intermediates)` in the same format as `FISTANet.forward`.

### train(args)

Full training loop:
- Builds CT data loaders (Box streaming or local disk based on token)
- Creates FISTANet with a dummy eye matrix (RadonOperator handles the physics)
- Uses two Adam parameter groups: one for ProxNet weights at `lr_net`, one for the six scalar parameters (w1,c1,w2,c2,w3,c3) at `lr_params`
- Saves a checkpoint every `save_every` epochs and at the final epoch
- Saves the best-PSNR model separately at the end

**CLI flags**: `--n_epochs`, `--lr_net`, `--lr_params`, `--lambda1`, `--lambda2`, `--n_views`, `--batch_size`, `--patch_size`, `--box_token`

---

## ct/train_istanet.py

### run_ista_ct(model, fbp, sino, radon_op, device)

Same structure as `run_fista_ct` but without the momentum update. Uses per-stage `model.mus[k]` and `model.thetas[k]` instead of the functional schedules.

### train(args)

Identical training loop to FISTA-Net CT but with ISTANet and two Adam parameter groups: ProxNet weights and the ParameterList of per-stage mus/thetas.

Reuses `fista_net_loss` from `ct.train_fistanet` — the loss formulation is the same for both models.

---

## ct/train_fbpconvnet.py

### train(args)

Trains FBPConvNet (U-Net) as a post-processor. Input is the FBP reconstruction `fbp`; target is the ground-truth full-dose image `gt`. Loss is plain MSE. Single Adam optimizer over all U-Net parameters. Same checkpoint saving logic as the other training scripts.

**CLI flags**: `--n_epochs`, `--lr_net`, `--n_views`, `--batch_size`, `--patch_size`, `--box_token`

---

## ct/evaluate_all.py

### find_best_checkpoint(weights_dir, prefix)

Scans the weights directory for `{prefix}_ct_*.pth` files and returns the one with the highest PSNR value parsed from the filename. Raises FileNotFoundError with a helpful message if no checkpoint exists.

### load_fistanet / load_istanet / load_fbpconvnet

Load a model from a checkpoint path, restore state dict, set to eval mode.

### evaluate_all(...)

Runs all five methods on the test set in a single pass:
1. FBP — uses the fbp tensor directly from the DataLoader
2. FISTA-TV — calls `fista_tv_ct` per image in numpy
3. ISTA-Net — calls `run_ista_ct`
4. FBPConvNet — calls `fbpc_model(fbp)`
5. FISTA-Net — calls `run_fista_ct`

Returns a results dict: `{method: {'PSNR': list, 'SSIM': list, 'RMSE': list}}`.

### Figures saved

| File | Content |
|------|---------|
| `ct_comparison_all_methods.png` | Side-by-side reconstruction images for n_display test slices |
| `ct_error_maps.png` | Reconstruction images + absolute error maps (hot colormap) for first test slice |
| `ct_metrics_barchart.png` | Bar chart of mean ± std for PSNR, SSIM, RMSE across all methods |
| `ct_learned_params.png` | Line plots of learned μ_k, θ_k, ρ_k schedules across the 7 stages |

### run(args)

Full evaluation pipeline: loads test data, loads checkpoints (auto-selects best by PSNR if paths not specified), runs evaluation, prints table, saves CSVs and all four figures.

**CLI flags**: `--fista`, `--ista`, `--fbpc` (checkpoint paths), `--n_views`, `--n_display`, `--fista_tv_iters`, `--box_token`

---

## emt/dataset.py

### EMTDataset

Loads one split from the pre-generated `.pt` files in `FISTA_Net_EMT_Dataset/`. Each file contains a dict with three tensors:

| Key | Shape | Content |
|-----|-------|---------|
| `measurements` | (N, 64) | noisy differential voltage measurements |
| `x0` | (N, 1, 64, 64) | initial estimate (backprojection or Laplacian reg output) |
| `phantoms` | (N, 1, 64, 64) | ground-truth conductivity contrast images |

Split names: `"train"` → `train.pt`, `"val"` → `val.pt`, `"test1"` → `test_set1.pt`, `"test2"` → `test_set2.pt`.

`__getitem__` returns `(measurements, x0, phantoms)` — all tensors.

### build_emt_loaders(data_dir, batch_size, num_workers, test_split, n_train)

Builds train, val, and test DataLoaders. If `n_train` is set (default 640 from config), wraps the training dataset in a `Subset` to use only the first `n_train` samples — val and test are always evaluated in full. `test_split` selects between `"test1"` and `"test2"`.

### load_sensitivity_matrix(data_dir)

Loads `sensitivity_matrix_A.npy` from the dataset directory. Shape: (64, 4096) — 64 measurements × 4096 pixels (64×64 image flattened). This is passed as `A_matrix` to FISTANet and ISTANet.

### load_laplacian(data_dir)

Loads `laplacian_matrix_L.npy`. Shape: (4096, 4096). Used by the Laplacian regularisation baseline.

---

## emt/baselines.py

### laplacian_regularization(b, A, L, lam)

Closed-form Tikhonov solution: `x* = (A^T A + λ L^T L)^{-1} A^T b`. Solved with `np.linalg.solve`. Returns a (sqrt(N), sqrt(N)) float32 image.

### fista_tv_emt(b, A, x0, n_iters, lam)

TV-regularised EMT reconstruction using explicit A matrix (no Radon — just matrix-vector products). Uses FISTA with isotropic TV proximal step. Step size `mu = 1 / (||A||_2^2 + epsilon)` set by Lipschitz constant. Returns a (H, W) float32 image.

---

## emt/train_fistanet.py

### fista_net_loss

Identical implementation to the CT version — same combined MSE + symmetry + sparsity loss.

### train(args)

Loads EMT data loaders and sensitivity matrix A. Passes A directly to FISTANet (no dummy eye — the explicit matrix multiplication handles the forward model). Same two-group Adam optimizer and checkpoint logic as CT.

**CLI flags**: `--n_epochs`, `--lr_net`, `--lr_params`, `--lambda1`, `--lambda2`, `--img_size`, `--batch_size`

---

## emt/train_istanet.py

Trains ISTANet on EMT. Reuses `fista_net_loss` from `emt.train_fistanet`. Passes sensitivity matrix A to ISTANet. Same training loop and checkpoint logic.

---

## emt/train_fbpconvnet.py

Trains FBPConvNet on EMT. Uses the pre-computed initial estimate `x0` from the dataset as input (analogous to the FBP image in CT). MSE loss against ground-truth phantoms.

---

## emt/evaluate_all.py

Evaluates five methods on the EMT test set: Lap.Reg, FISTA-TV, ISTA-Net, FBPConvNet, FISTA-Net.

Classical baselines (Lap.Reg and FISTA-TV) run per-sample in numpy. Deep models run in batched PyTorch with `torch.no_grad()`.

Saves:
- `emt_{test_split}_per_sample.csv` — per-sample metrics
- `emt_{test_split}_summary.csv` — mean ± std summary
- `emt_comparison_{test_split}.png` — side-by-side visual comparison (hot colormap)

**CLI flags**: `--fista`, `--ista`, `--fbpc`, `--test_split` (`test1` or `test2`), `--n_display`

---

## How to Run

### CT Workflow

```bash
# Set Box token in config.py (or pass --box_token each time)
# CT["box_token"] = "your_token_here"

# Train all three models
python ct/train_fistanet.py
python ct/train_istanet.py
python ct/train_fbpconvnet.py

# Evaluate and save results
python ct/evaluate_all.py
```

Override any hyperparameter from CLI:
```bash
python ct/train_fistanet.py --n_epochs 30 --lr_net 5e-5 --patch_size 64
python ct/evaluate_all.py --n_display 5 --fista_tv_iters 200
```

### EMT Workflow

```bash
# No data download needed — .pt files already in emt/FISTA_Net_EMT_Dataset/

python emt/train_fistanet.py
python emt/train_istanet.py
python emt/train_fbpconvnet.py

python emt/evaluate_all.py                   # evaluates on test_set1
python emt/evaluate_all.py --test_split test2
```

### Checkpoint Naming

All checkpoints follow the pattern:
```
{model}_{modality}_ep{epoch:03d}_psnr{psnr:.2f}.pth
```

`evaluate_all.py` automatically selects the checkpoint with the highest PSNR in the filename for each model when no explicit path is given.

---

## Methods Compared

| Method | Type | Notes |
|--------|------|-------|
| FBP | Classical | Filtered back-projection (CT) / direct backprojection (EMT) |
| FISTA-TV | Classical | TV-regularised iterative reconstruction |
| Lap.Reg | Classical | Laplacian regularisation — EMT only |
| ISTA-Net | Deep unrolled | Unrolled ISTA without momentum, same stage count as FISTA-Net |
| FBPConvNet | Deep post-processor | U-Net applied to FBP/initial estimate |
| FISTA-Net | Deep unrolled | Proposed — unrolled FISTA with learned μ, θ, ρ schedules |

---

## Metrics

| Metric | Meaning | Better |
|--------|---------|--------|
| PSNR (dB) | Peak Signal-to-Noise Ratio | higher |
| SSIM | Structural Similarity Index | higher |
| RMSE | Root Mean Square Error | lower |

---

## Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-image scipy tqdm pydicom requests
```

---

## Paper Reference

> Xiang, J., Dong, Y., & Yang, Y. (2021). FISTA-Net: Learning A Fast Iterative Shrinkage Thresholding Network for Inverse Problems in Imaging. *IEEE Transactions on Medical Imaging*, 40(5), 1329–1339.
