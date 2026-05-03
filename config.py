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

# CT ─────────────────────────────────────────────────────────────────────────
# Used by MayoCTDataset (local-disk mode).
# Ignored when CT["box_token"] is set (BoxCTDataset caches to ct_cache/slices/).
CT_DATA_DIR    = ROOT_DIR / "ct_cache" / "full_3mm" / "full_3mm"
CT_WEIGHTS_DIR = ROOT_DIR / "ct" / "weights"
CT_RESULTS_DIR = ROOT_DIR / "ct" / "results"

# EMT ────────────────────────────────────────────────────────────────────────
# Pre-generated dataset (.pt files) live here:
#   EMT_DATASET_DIR / train.pt, val.pt, test_set1.pt, test_set2.pt
#   EMT_DATASET_DIR / sensitivity_matrix_A.npy  (64 × 4096, pixel-space)
#   EMT_DATASET_DIR / laplacian_matrix_L.npy    (4096 × 4096)
EMT_DATASET_DIR = ROOT_DIR / "emt" / "FISTA_Net_EMT_Dataset"
EMT_DATA_DIR    = EMT_DATASET_DIR   # backward-compat alias
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
    # ── Box streaming (no full zip download) ────────────────────────────────
    # Paste your Box developer token here (regenerate every 60 min at
    # developer.box.com).  Only needed until slices are cached locally.
    # Leave "" to use local disk mode (data_root = CT_DATA_DIR above).
    # Override per-run: python ct/train_fistanet.py --box_token YOUR_TOKEN
    box_token = "KQnlIGH3cXOisdXBGMh59ZgacfiVSm9c",

    # Mayo Clinic patients split (paper uses 10 patients total)
    train_patients = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"],
    val_patients   = ["L333"],
    test_patients  = ["L506"],

    # Slices per patient used for training/val/test
    # (paper trains on limited slices for speed — full dataset has 500+ per patient)
    slices_per_patient_train = 30,   # 30 × 8 patients = 240 train slices
    slices_per_patient_val   = 10,   # 10 × 1 = 10 val slices
    slices_per_patient_test  = 10,   # 10 × 1 = 10 test slices

    # Sparse-view CT settings (paper: 60 equi-spaced views)
    n_views    = 60,

    # Full DICOM slice is 512×512; we train on random crops to fit VRAM
    image_size = 512,
    patch_size = 128,   # training crop size; test always uses full 512×512

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
    # FEM system (matches FISTA_Net_EMT_Dataset_FEM.ipynb)
    n_coils    = 8,       # electrodes on boundary
    n_meas     = 64,      # total measurements (8 excitations × 8 per excitation)
    mesh_h0    = 0.04,
    dist_exc   = 1,

    # Image output
    img_size   = 64,      # 64×64 pixel grid
    domain_r   = 0.90,

    # Conductivity (S/m)
    sigma_bg   = 1.0,
    sigma_obj  = 3.0,
    obj_r_min  = 0.12,
    obj_r_max  = 0.28,

    # Dataset sizes (from pre-generated .pt files)
    n_train    = 1920,
    n_val      = 480,
    n_test     = 1200,    # per test set (test_set1 and test_set2)

    # Subset of training data to use (None = full dataset).
    # 640 = 1/3 of 1920 → ~40 min/model → ~2 hr for all 3 models on CPU.
    # Val/test splits are always evaluated in full regardless of this setting.
    train_subset = 640,

    # Noise levels tested in paper: 20, 30, 40 dB
    noise_db   = 30,

    # DataLoader
    batch_size  = 16,
    num_workers = 2,
    pin_memory  = True,
)

# ─────────────────────────────────────────────────────────────────────────────
# FISTA-Net  —  Architecture  (Section III of paper)
# ─────────────────────────────────────────────────────────────────────────────
FISTA_NET = dict(
    n_stages  = 7,    # number of unrolled iterations  (paper: 7)
    n_filters = 32,   # channels in ProximalMappingNetwork

    # Scalar parameter initialisations (paper Section III-C)
    init_w1 = -0.2,   # step-size μ   (w_μ  in paper)
    init_c1 =  0.1,   #               (b_μ  in paper)
    init_w2 = -0.5,   # threshold θ   (w_θ  in paper)
    init_c2 = -2.0,   #               (b_θ  in paper)
    init_w3 =  0.5,   # momentum ρ    (w_ρ  in paper)
    init_c3 =  0.0,   #               (b_ρ  in paper)
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
    n_epochs_ct   = 20,
    lr_net_ct     = 1e-4,   # ProxNet / U-Net learning rate
    lr_params_ct  = 1e-3,   # FISTA-Net scalar params (μ, θ, ρ) learning rate

    # EMT
    n_epochs_emt  = 50,
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
