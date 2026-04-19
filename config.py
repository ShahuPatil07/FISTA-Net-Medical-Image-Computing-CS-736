"""
config.py  —  MASTER CONFIGURATION
===================================
Single source of truth for ALL hyperparameters and paths.
Edit here; every training/evaluation script imports from this file.

To override a value for a one-off run without editing this file, use:
    python ct/train_fistanet.py --n_epochs 20 --lr_net 5e-5
(each script exposes CLI overrides for the most-used params)
"""

from pathlib import Path
import torch

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ─────────────────────────────────────────────────────────────────────────────
# BOX TOKEN  (regenerate every 60 min at developer.box.com)
# Only needed on first run per split — individual slices are cached after that.
# Leave empty ("") if data is already on disk at CT_DATA_DIR.
# ─────────────────────────────────────────────────────────────────────────────
BOX_TOKEN = ""   # ← paste your token here, or pass --box_token on the CLI

# CT ─────────────────────────────────────────────────────────────────────────
# Used by MayoCTDataset (local-disk mode).
# Ignored when BOX_TOKEN is set (BoxCTDataset caches to ct_cache/slices/).
CT_DATA_DIR    = ROOT_DIR / "ct_cache" / "full_3mm" / "full_3mm"
CT_WEIGHTS_DIR = ROOT_DIR / "ct" / "weights"
CT_RESULTS_DIR = ROOT_DIR / "ct" / "results"

# EMT ────────────────────────────────────────────────────────────────────────
# After running emt/generate_data.py, HDF5 files land here:
#   EMT_DATA_DIR / emt_train.h5
#   EMT_DATA_DIR / emt_test.h5
#   EMT_DATA_DIR / sensitivity_matrix_A.npy  … etc.
EMT_DATA_DIR   = ROOT_DIR / "emt" / "data"
EMT_WEIGHTS_DIR = ROOT_DIR / "emt" / "weights"
EMT_RESULTS_DIR = ROOT_DIR / "emt" / "results"

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# CT — DATA  (Section IV-B of paper)
# ─────────────────────────────────────────────────────────────────────────────
CT = dict(
    # Mayo Clinic patients split (paper uses 10 patients total)
    train_patients = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"],
    val_patients   = ["L333"],
    test_patients  = ["L506"],

    # Slices per patient (set None to use all)
    slices_per_patient_train = 30,
    slices_per_patient_val   = 10,
    slices_per_patient_test  = 10,

    # Sparse-view CT settings (paper: 60 equi-spaced views)
    n_views    = 60,

    # Full DICOM slice is 512×512; we train on crops to fit VRAM
    image_size = 512,
    patch_size = 128,   # training crop; set None to use full slice

    # HU windowing → maps to [0, 1]
    win_min = -140,
    win_max =  260,

    # DataLoader
    batch_size   = 2,
    num_workers  = 2,
    pin_memory   = True,
)

# ─────────────────────────────────────────────────────────────────────────────
# EMT — DATA  (Section IV-A of paper)
# ─────────────────────────────────────────────────────────────────────────────
EMT = dict(
    # FEM mesh / physics
    n_coils    = 8,
    mesh_h0    = 0.04,   # element size — smaller = finer = slower
    dist_exc   = 1,      # adjacent electrode excitation

    # Image output
    img_size   = 64,
    domain_r   = 0.90,

    # Conductivity (S/m)
    sigma_bg   = 1.0,
    sigma_obj  = 3.0,
    obj_r_min  = 0.12,
    obj_r_max  = 0.28,

    # Dataset sizes (paper uses ~2000 train)
    n_train    = 1000,
    n_test     = 200,

    # Noise levels tested in paper: 20, 30, 40 dB
    noise_db   = 30,

    # DataLoader
    batch_size  = 16,
    num_workers = 2,
)

# ─────────────────────────────────────────────────────────────────────────────
# FISTA-Net  —  Architecture  (Section III of paper)
# ─────────────────────────────────────────────────────────────────────────────
FISTA_NET = dict(
    n_stages  = 7,    # number of unrolled iterations  (paper: 7)
    n_filters = 32,   # channels in ProximalMappingNetwork

    # Scalar parameter initialisations (paper Section III-C)
    # These are log-space initialisations fed through softplus
    init_w1 = -0.5,   # step-size μ
    init_c1 = -2.0,
    init_w2 = -0.2,   # threshold θ
    init_c2 = -1.0,
    init_w3 =  1.0,   # momentum ρ
    init_c3 =  0.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# ISTA-Net  —  Architecture  (baseline; same proximal network, no momentum)
# ─────────────────────────────────────────────────────────────────────────────
ISTA_NET = dict(
    n_stages  = 7,    # same as FISTA-Net for fair comparison
    n_filters = 32,
    init_mu    = 0.1,
    init_theta = 0.1,
)

# ─────────────────────────────────────────────────────────────────────────────
# FBPConvNet  —  Architecture  (U-Net baseline)
# ─────────────────────────────────────────────────────────────────────────────
FBPCONVNET = dict(
    base_ch = 32,   # feature channels at first encoder level
)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING  —  Common hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
TRAIN = dict(
    # CT
    n_epochs_ct   = 10,
    lr_net_ct     = 1e-4,   # ProxNet / U-Net learning rate
    lr_params_ct  = 1e-3,   # FISTA-Net scalar params (μ, θ, ρ) learning rate

    # EMT
    n_epochs_emt  = 100,
    lr_net_emt    = 1e-4,
    lr_params_emt = 1e-3,

    # Shared loss weights  (Eq. 14 of paper)
    lambda1 = 0.01,    # symmetry loss  L_sym
    lambda2 = 0.001,   # sparsity loss  L_spa

    # Gradient clipping
    grad_clip = 1.0,

    # Checkpoint save frequency (epochs)
    save_every = 5,
)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSICAL BASELINES
# ─────────────────────────────────────────────────────────────────────────────
CLASSICAL = dict(
    # FISTA-TV (CT)
    fista_tv_ct_iters  = 100,
    fista_tv_ct_lam    = 0.005,
    fista_tv_ct_step   = 0.001,

    # FISTA-TV (EMT)
    fista_tv_emt_iters = 100,
    fista_tv_emt_lam   = 0.001,

    # Laplacian regularisation (EMT only)
    lap_reg_lam = 0.001,
)

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  —  visualisation settings
# ─────────────────────────────────────────────────────────────────────────────
EVAL = dict(
    n_display   = 3,    # test slices shown in visual comparison figure
    figure_dpi  = 150,
)

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT NAMING CONVENTION
# ─────────────────────────────────────────────────────────────────────────────
# Saved as:  {model}_{modality}_ep{epoch:03d}_psnr{psnr:.2f}.pth
# Examples:
#   ct/weights/fistanet_ct_ep010_psnr28.45.pth
#   ct/weights/istanet_ct_ep010_psnr27.12.pth
#   ct/weights/fbpconvnet_ct_ep010_psnr26.89.pth
#   emt/weights/fistanet_emt_ep100_psnr22.34.pth

def weight_name(model: str, modality: str, epoch: int, psnr: float) -> str:
    return f"{model}_{modality}_ep{epoch:03d}_psnr{psnr:.2f}.pth"
