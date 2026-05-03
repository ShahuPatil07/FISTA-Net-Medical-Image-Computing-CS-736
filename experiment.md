# CT Experiment Guide — FISTA-Net Replication (Section IV-B)

This file explains how to run every CT experiment end-to-end, what each
hyperparameter does, and how to reproduce the paper's results.

---

## 0. One-time setup

```bash
pip install torch torchvision numpy matplotlib scikit-image scipy tqdm pydicom requests
```

All parameters live in **`config.py`** at the project root.  Edit it once;
every script reads from it automatically.

---

## 1. Set your Box token (once per environment)

Open `config.py` and paste your token:

```python
CT = dict(
    box_token = "YOUR_BOX_DEV_TOKEN_HERE",   # ← regenerate at developer.box.com
    ...
)
```

Token expires every 60 minutes. On **first run**, the dataset loader fetches
only the slices it needs (30 train + 10 val + 10 test per patient) via HTTP
range requests from the Box zip — **no full download**. Slices are cached to
`ct_cache/slices/` and the token is never needed again for that environment.

You can also pass the token per-run without editing `config.py`:
```bash
python ct/train_fistanet.py --box_token YOUR_TOKEN
```

---

## 2. Data parameters (all in `config.py → CT = dict(...)`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `box_token` | `""` | Box dev token. Leave `""` if data already on disk. |
| `train_patients` | 8 patients | Mayo Clinic patient IDs used for training |
| `val_patients` | `["L333"]` | 1 patient for validation |
| `test_patients` | `["L506"]` | 1 patient for final evaluation |
| `slices_per_patient_train` | 30 | Slices fetched per train patient (240 total) |
| `slices_per_patient_val` | 10 | Slices fetched for val patient |
| `slices_per_patient_test` | 10 | Slices fetched for test patient |
| `n_views` | 60 | Number of equi-spaced sparse-view projections (paper: 60) |
| `patch_size` | 128 | Random crop size during training; test uses full 512×512 |
| `batch_size` | 2 | Training batch size (reduce if OOM) |
| `win_min / win_max` | -140 / 260 | HU window → normalised [0, 1] |

---

## 3. Model parameters (all in `config.py`)

### FISTA-Net (`FISTA_NET = dict(...)`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_stages` | 7 | Unrolled FISTA iterations (paper: 7) |
| `n_filters` | 32 | Conv channels in ProximalMappingNetwork |
| `init_w1 / init_c1` | -0.5 / -2.0 | Initial μ schedule weights (softplus param) |
| `init_w2 / init_c2` | -0.2 / -1.0 | Initial θ schedule weights |
| `init_w3 / init_c3` | 1.0 / 0.0 | Initial ρ schedule weights |

### ISTA-Net (`ISTA_NET = dict(...)`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_stages` | 7 | Same as FISTA-Net for fair comparison |
| `init_mu` | 0.1 | Per-stage step-size initialisation |
| `init_theta` | 0.1 | Per-stage threshold initialisation |

### FBPConvNet (`FBPCONVNET = dict(...)`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `base_ch` | 32 | Feature channels at first U-Net encoder level |

---

## 4. Training parameters (`TRAIN = dict(...)`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_epochs_ct` | 10 | Training epochs |
| `lr_net_ct` | 1e-4 | ProxNet / U-Net learning rate |
| `lr_params_ct` | 1e-3 | FISTA-Net scalar params (μ, θ, ρ) learning rate |
| `lambda1` | 0.01 | Symmetry loss weight L_sym (Eq. 14) |
| `lambda2` | 0.001 | Sparsity loss weight L_spa (Eq. 14) |
| `grad_clip` | 1.0 | Gradient norm clip |
| `save_every` | 5 | Save checkpoint every N epochs + always save best |

---

## 5. Classical baseline parameters (`CLASSICAL = dict(...)`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fista_tv_ct_iters` | 100 | FISTA-TV outer iterations |
| `fista_tv_ct_lam` | 0.005 | TV regularisation weight |
| `fista_tv_ct_step` | 0.001 | Gradient step size μ |

---

## 6. Run training

Run each script independently (or in parallel on separate GPUs):

```bash
# Proposed method
python ct/train_fistanet.py

# Baselines
python ct/train_istanet.py
python ct/train_fbpconvnet.py
```

Override any parameter from the CLI without editing `config.py`:
```bash
python ct/train_fistanet.py --n_epochs 20 --lr_net 5e-5 --patch_size 64
python ct/train_fistanet.py --box_token YOUR_TOKEN --n_epochs 10
python ct/train_fbpconvnet.py --n_epochs 20 --lr_net 1e-4
```

All three scripts accept: `--n_epochs`, `--n_views`, `--batch_size`,
`--patch_size`, `--box_token`.  FISTA-Net and ISTA-Net also accept
`--lr_net`, `--lr_params`, `--lambda1`, `--lambda2`.

Checkpoints are saved to `ct/weights/` with the naming convention:
```
{model}_ct_ep{epoch:03d}_psnr{val_psnr:.2f}.pth
```
Saved at every `save_every` epochs and always at the best validation PSNR.

---

## 7. Run evaluation

After all three models are trained:

```bash
python ct/evaluate_all.py
```

This will:
- Auto-select the best checkpoint for each model (highest PSNR in filename)
- Run all 5 methods on the test set: FBP, FISTA-TV, ISTA-Net, FBPConvNet, FISTA-Net
- Print a paper-style comparison table to stdout
- Save all results to `ct/results/`:

```
ct/results/
  tables/
    ct_summary.csv      ← mean ± std per method
    ct_per_sample.csv   ← per-slice metrics for all methods
  figures/
    ct_comparison_all_methods.png   ← visual reconstruction comparison
    ct_error_maps.png               ← |error| maps for all methods
    ct_metrics_barchart.png         ← PSNR / SSIM / RMSE bar chart
    ct_learned_params.png           ← μ, θ, ρ schedules across stages
```

Manual checkpoint override:
```bash
python ct/evaluate_all.py \
  --fista ct/weights/fistanet_ct_ep020_psnr31.2.pth \
  --ista  ct/weights/istanet_ct_ep015_psnr29.8.pth  \
  --fbpc  ct/weights/fbpconvnet_ct_ep010_psnr28.1.pth
```

Additional flags:
```bash
--n_display 4          # show 4 test slices in comparison figure (default: 3)
--fista_tv_iters 200   # more FISTA-TV iterations for better quality
--box_token YOUR_TOKEN # if slices not yet cached locally
```

---

## 8. Typical expected results (paper Table V equivalent)

| Method | PSNR (dB) | SSIM | RMSE |
|--------|-----------|------|------|
| FBP | ~23–25 | ~0.60 | ~0.06 |
| FISTA-TV | ~26–27 | ~0.72 | ~0.04 |
| ISTA-Net | ~27–28 | ~0.78 | ~0.03 |
| FBPConvNet | ~27–29 | ~0.80 | ~0.03 |
| **FISTA-Net** | **~29–31** | **~0.85** | **~0.02** |

Exact numbers depend on number of training epochs, slices used, and GPU.
The paper trains for longer with the full dataset — the above are typical
for 10 epochs with 30 slices/patient.

---

## 9. Experiment variations

### Vary the number of projection views

```bash
python ct/train_fistanet.py --n_views 30   # more undersampled
python ct/train_fistanet.py --n_views 90   # closer to full
python ct/evaluate_all.py   --n_views 30
```

### Change training data size

Edit `config.py`:
```python
slices_per_patient_train = 60,   # use more slices
slices_per_patient_train = 10,   # quick smoke-test run
```

### Ablation: momentum vs no-momentum

Compare FISTA-Net (with ρ) vs ISTA-Net (no ρ) — both trained above.
The PSNR difference shows the contribution of the learned momentum term.

### Ablation: number of stages

```python
# config.py
FISTA_NET = dict(n_stages=3, ...)   # fewer stages
FISTA_NET = dict(n_stages=10, ...)  # more stages
```

Retrain and compare — shows the trade-off between depth and performance.

### Quick smoke-test (no GPU needed)

```bash
python ct/train_fistanet.py --n_epochs 2 --batch_size 1 --patch_size 64 \
  --slices_per_patient 2   # not a CLI arg — edit config.py directly
```

---

## 10. Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `401 Unauthorized` from Box | Token expired | Regenerate at developer.box.com, update `CT["box_token"]` in config.py |
| `No slices found` | Wrong patient IDs or Box path | Verify `train_patients` in config.py match actual Box folder names |
| `CUDA out of memory` | Batch too large | Reduce `batch_size` or `patch_size` in config.py |
| `No checkpoint found` | Training not done | Run all three `ct/train_*.py` scripts first |
| Very low PSNR (~20 dB) | Too few epochs | Increase `n_epochs_ct` in config.py or use `--n_epochs 30` |
| Metrics much lower than paper | Slice count too low | Increase `slices_per_patient_train` in config.py |
