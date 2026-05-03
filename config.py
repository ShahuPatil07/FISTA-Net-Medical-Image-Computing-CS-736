from pathlib import Path
import torch

ROOT_DIR = Path(__file__).parent.resolve()

CT_DATA_DIR    = ROOT_DIR / "ct_cache" / "full_3mm" / "full_3mm"
CT_WEIGHTS_DIR = ROOT_DIR / "ct" / "weights"
CT_RESULTS_DIR = ROOT_DIR / "ct" / "results"

EMT_DATASET_DIR = ROOT_DIR / "emt" / "FISTA_Net_EMT_Dataset"
EMT_WEIGHTS_DIR = ROOT_DIR / "emt" / "weights"
EMT_RESULTS_DIR = ROOT_DIR / "emt" / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CT = dict(
    box_token = "KQnlIGH3cXOisdXBGMh59ZgacfiVSm9c",

    train_patients = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"],
    val_patients   = ["L333"],
    test_patients  = ["L506"],

    slices_per_patient_train = 30,
    slices_per_patient_val   = 10,
    slices_per_patient_test  = 10,

    n_views    = 60,

    image_size = 512,
    patch_size = 128,

    win_min = -140,
    win_max =  260,

    batch_size   = 2,
    num_workers  = 2,
    pin_memory   = True,
)

EMT = dict(
    n_coils    = 8,
    n_meas     = 64,
    mesh_h0    = 0.04,
    dist_exc   = 1,

    img_size   = 64,
    domain_r   = 0.90,

    sigma_bg   = 1.0,
    sigma_obj  = 3.0,
    obj_r_min  = 0.12,
    obj_r_max  = 0.28,

    n_train    = 1920,
    n_val      = 480,
    n_test     = 1200,

    train_subset = 640,

    noise_db   = 30,

    batch_size  = 16,
    num_workers = 2,
    pin_memory  = True,
)

FISTA_NET = dict(
    n_stages  = 7,
    n_filters = 32,

    init_w1 = -0.2,
    init_c1 =  0.1,
    init_w2 = -0.5,
    init_c2 = -2.0,
    init_w3 =  0.5,
    init_c3 =  0.0,
)

ISTA_NET = dict(
    n_stages  = 7,
    n_filters = 32,
    init_mu    = 0.1,
    init_theta = 0.1,
)

FBPCONVNET = dict(
    base_ch = 32,
)

TRAIN = dict(
    n_epochs_ct   = 20,
    lr_net_ct     = 1e-4,
    lr_params_ct  = 1e-3,

    n_epochs_emt  = 50,
    lr_net_emt    = 1e-4,
    lr_params_emt = 1e-3,

    lambda1 = 0.01,
    lambda2 = 0.001,

    grad_clip = 1.0,

    save_every = 5,
)

CLASSICAL = dict(
    fista_tv_ct_iters  = 100,
    fista_tv_ct_lam    = 0.005,
    fista_tv_ct_step   = 0.001,

    fista_tv_emt_iters = 100,
    fista_tv_emt_lam   = 0.001,

    lap_reg_lam = 0.001,
)

EVAL = dict(
    n_display   = 3,
    figure_dpi  = 150,
)


def weight_name(model: str, modality: str, epoch: int, psnr: float) -> str:
    return f"{model}_{modality}_ep{epoch:03d}_psnr{psnr:.2f}.pth"
