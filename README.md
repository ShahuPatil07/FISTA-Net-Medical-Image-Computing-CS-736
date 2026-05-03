# FISTA-Net Replication — CS 736 Project

Replication of **FISTA-Net: Learning A Fast Iterative Shrinkage Thresholding Network
for Inverse Problems in Imaging** (Xiang et al., 2021).

Two modalities are implemented completely separately:
- **CT** — Mayo Clinic sparse-view CT (Section IV-B of paper)  ← ready to run
- **EMT** — Electrical Impedance Tomography (Section IV-A)     ← data gen in progress

---

## Project Structure

```
CS 736/
│
├── config.py                    ← MASTER CONFIG — edit paths & hyperparams here
├── README.md
│
├── shared/                      ← Model architectures shared by CT and EMT
│   ├── models.py                ← FISTANet, ISTANet, FBPConvNet, ProximalMappingNetwork
│   └── metrics.py               ← compute_metrics, print_results_table, save_*_csv
│
├── ct/                          ← Everything CT (Mayo Clinic)
│   ├── dataset.py               ← MayoCTDataset, RadonOperator, build_ct_loaders
│   ├── baselines.py             ← fista_tv_ct (classical TV baseline)
│   ├── train_fistanet.py        ← Train FISTA-Net  →  ct/weights/fistanet_ct_ep010_psnr28.45.pth
│   ├── train_istanet.py         ← Train ISTA-Net   →  ct/weights/istanet_ct_ep010_psnr27.12.pth
│   ├── train_fbpconvnet.py      ← Train FBPConvNet →  ct/weights/fbpconvnet_ct_ep010_psnr26.89.pth
│   ├── evaluate_all.py          ← Load all weights → generate tables + figures
│   ├── weights/                 ← Saved checkpoints (auto-created during training)
│   ├── results/
│   │   ├── figures/             ← ct_comparison_all_methods.png, ct_error_maps.png, …
│   │   └── tables/              ← ct_summary.csv, ct_per_sample.csv
│   └── notebooks/
│       ├── 01_CT_EDA.ipynb      ← Data exploration & visualisation
│       └── 02_CT_Evaluate_All.ipynb  ← Interactive evaluation after training
│
└── emt/                         ← Everything EMT
    ├── FISTA_Net_EMT_Dataset/   ← Pre-generated .pt dataset files
    ├── dataset.py               ← EMTDataset (.pt loader), build_emt_loaders
    ├── baselines.py             ← laplacian_regularization, fista_tv_emt
    ├── train_fistanet.py        ← Train FISTA-Net on EMT
    ├── train_istanet.py         ← Train ISTA-Net  on EMT
    ├── train_fbpconvnet.py      ← Train FBPConvNet on EMT
    ├── evaluate_all.py          ← Evaluate all EMT methods
    ├── weights/                 ← (auto-created)
    ├── results/
    │   ├── figures/
    │   └── tables/
    └── notebooks/
        └── 01_EMT_EDA.ipynb     ← EMT data exploration
```

---

## Step 0 — Configuration

No path editing required.  CT data is downloaded from Box automatically.

**`config.py`** is the only file you need for hyperparameters, but the default
paths are already wired correctly — don't change them unless you move the data.

All training hyperparameters, dataset splits, and output paths live in `config.py`.
No need to hunt through individual scripts.

---

## CT Workflow (Section IV-B)

### 1. Set Your Box Token

In `config.py` (or pass `--box_token` on the CLI):
```python
BOX_TOKEN = "YOUR_BOX_DEV_TOKEN"   # regenerate every 60 min at developer.box.com
```

**No full zip download.** `BoxCTDataset` uses HTTP range requests to read only the
configured slices (30 train + 10 val + 10 test per patient) directly from the Box zip.
Individual DICOM files are cached to `ct_cache/slices/` (~a few MB) — after the
first run the token is no longer needed.

### 2. Verify Data

```bash
jupyter notebook ct/notebooks/01_CT_EDA.ipynb
```

Fetches a small sample via Box streaming, shows sample slices, sinograms, FBP quality
at different views, and computes baseline FBP metrics on the test set.

### 2. Train Models

Run each script independently (or in parallel on different GPUs):

```bash
python ct/train_fistanet.py         # proposed method
python ct/train_istanet.py          # baseline: no momentum
python ct/train_fbpconvnet.py       # baseline: U-Net post-processor
```

Override any hyperparameter from CLI:
```bash
python ct/train_fistanet.py --n_epochs 20 --lr_net 5e-5 --patch_size 64
# Pass token on CLI if not set in config.py:
python ct/train_fistanet.py --box_token YOUR_TOKEN
```

Checkpoints are saved to `ct/weights/` with the naming convention:
```
{model}_{modality}_ep{epoch:03d}_psnr{val_psnr:.2f}.pth
```
e.g. `fistanet_ct_ep010_psnr28.45.pth`

Every `TRAIN["save_every"]` epochs + the best epoch are always saved.

### 3. Evaluate All Methods

```bash
python ct/evaluate_all.py
```

This script:
- Auto-selects the best checkpoint for each model (highest PSNR filename)
- Runs FBP, FISTA-TV (classical), ISTA-Net, FBPConvNet, FISTA-Net on the test set
- Prints a paper-style comparison table to stdout
- Saves to `ct/results/`:
  - `tables/ct_summary.csv` — mean ± std per method
  - `tables/ct_per_sample.csv` — per-slice metrics
  - `figures/ct_comparison_all_methods.png`
  - `figures/ct_error_maps.png`
  - `figures/ct_metrics_barchart.png`
  - `figures/ct_learned_params.png`

Or interactively:
```bash
jupyter notebook ct/notebooks/02_CT_Evaluate_All.ipynb
```

---

## EMT Workflow (Section IV-A)

Pre-generated dataset (`.pt` files) is in `emt/FISTA_Net_EMT_Dataset/`.
No data generation step is needed.

### 1. EDA

```bash
jupyter notebook emt/notebooks/01_EMT_EDA.ipynb
```

### 2. Train

```bash
python emt/train_fistanet.py
python emt/train_istanet.py
python emt/train_fbpconvnet.py
```

### 4. Evaluate

```bash
python emt/evaluate_all.py
```

---

## Methods Compared

| Method      | Type          | Notes |
|-------------|---------------|-------|
| FBP         | Classical     | Filtered back-projection (CT) / direct (EMT) |
| FISTA-TV    | Classical     | TV-regularised iterative reconstruction |
| Lap.Reg     | Classical     | Laplacian regularisation (EMT only) |
| ISTA-Net    | Deep (unrolled) | Unrolled ISTA, no momentum — same #stages as FISTA-Net |
| FBPConvNet  | Deep (post-proc) | U-Net applied to FBP/initial estimate |
| **FISTA-Net** | **Deep (unrolled)** | **Proposed — FISTA unrolling with learned μ, θ, ρ** |

---

## Metrics

| Metric | Meaning | Better |
|--------|---------|--------|
| PSNR (dB) | Peak Signal-to-Noise Ratio | ↑ higher |
| SSIM      | Structural Similarity Index | ↑ higher |
| RMSE      | Root Mean Square Error      | ↓ lower  |

---

## Weight Naming Convention

```
{model}_{modality}_ep{epoch:03d}_psnr{val_psnr:.2f}.pth
```

Examples:
```
ct/weights/fistanet_ct_ep010_psnr28.45.pth   ← FISTA-Net CT epoch 10
ct/weights/istanet_ct_ep010_psnr27.12.pth    ← ISTA-Net  CT epoch 10
ct/weights/fbpconvnet_ct_ep005_psnr26.89.pth ← FBPConvNet CT epoch 5 (periodic save)
emt/weights/fistanet_emt_ep100_psnr22.34.pth ← FISTA-Net EMT epoch 100
```

`evaluate_all.py` automatically picks the file with the **highest PSNR** in the filename.

---

## Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-image scipy tqdm pydicom
```

---

## Paper Reference

> Xiang, J., Dong, Y., & Yang, Y. (2021).
> **FISTA-Net: Learning A Fast Iterative Shrinkage Thresholding Network for Inverse Problems in Imaging.**
> *IEEE Transactions on Medical Imaging*, 40(5), 1329–1339.
