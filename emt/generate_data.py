"""
emt/generate_data.py
====================
Generate synthetic EMT dataset using FEM (pyeit).
Converts FISTA_Net_EMT_Dataset_FEM.ipynb into a runnable script.

Saves to EMT_DATA_DIR (configured in config.py):
    emt_train.h5              - training set
    emt_test.h5               - test set
    sensitivity_matrix_A.npy  - row-normalised Jacobian A  (N_MEAS x n_elements)
    baseline_voltages_v0.npy  - background voltages v0
    element_centroids.npy     - FEM element centroids (x, y)
    electrode_coords.npy      - electrode positions on boundary

Usage
-----
    python emt/generate_data.py                     # uses config.py defaults
    python emt/generate_data.py --n_train 2000      # override number of samples
    python emt/generate_data.py --noise_db 20       # override SNR

NOTE: Requires pyeit.  Install with:  pip install pyeit h5py
"""

import argparse, sys
from pathlib import Path

import numpy as np
import h5py
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMT, EMT_DATA_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Globals (populated in setup())
# ─────────────────────────────────────────────────────────────────────────────
eit_mesh     = None
eit_protocol = None
fwd_solver   = None
A            = None
v0           = None
CENTROIDS    = None
n_elements   = None
N_MEAS       = None
DOMAIN_MASK  = None
XX = YY      = None


def setup(args):
    """Build FEM mesh, compute sensitivity matrix A."""
    global eit_mesh, eit_protocol, fwd_solver, A, v0
    global CENTROIDS, n_elements, N_MEAS, DOMAIN_MASK, XX, YY

    import pyeit.mesh     as mesh_lib
    import pyeit.eit.protocol as protocol_lib
    import pyeit.eit.fem   as fem_lib

    print("Step 1: Creating FEM mesh …")
    eit_mesh   = mesh_lib.create(n_el=args.n_coils, h0=args.mesh_h0)
    n_nodes    = eit_mesh.node.shape[0]
    n_elements = eit_mesh.element.shape[0]
    print(f"  {n_nodes} nodes, {n_elements} triangular elements")

    el_pos_idx = eit_mesh.el_pos
    el_coords  = eit_mesh.node[el_pos_idx, :2]
    print("  Electrode positions (normalised circle):")
    for i, (x, y) in enumerate(el_coords):
        print(f"    Coil {i+1}: ({x:+.3f}, {y:+.3f})")

    print("\nStep 2: Setting up measurement protocol …")
    eit_protocol = protocol_lib.create(
        args.n_coils,
        dist_exc    = args.dist_exc,
        step_meas   = 1,
        parser_meas = "std",
    )
    N_MEAS = eit_protocol.n_exc * eit_protocol.n_meas
    print(f"  Excitations: {eit_protocol.n_exc} | Measurements each: {eit_protocol.n_meas}")
    print(f"  Total measurements: {N_MEAS}")

    print("\nStep 3: Computing FEM Jacobian (sensitivity matrix A) …")
    fwd_solver = fem_lib.EITForward(eit_mesh, eit_protocol)
    J, v0 = fwd_solver.compute_jac(
        perm      = args.sigma_bg * np.ones(n_elements),
        normalize = False,
    )
    print(f"  Jacobian shape: {J.shape}")

    row_norms = np.linalg.norm(J, axis=1, keepdims=True) + 1e-12
    A = J / row_norms

    CENTROIDS = eit_mesh.node[eit_mesh.element].mean(axis=1)[:, :2]

    # Pixel grid for rendering
    xi = np.linspace(-1, 1, args.img_size)
    yi = np.linspace(-1, 1, args.img_size)
    XX, YY = np.meshgrid(xi, yi)
    DOMAIN_MASK = (XX ** 2 + YY ** 2) <= args.domain_r ** 2

    # Save physics matrices
    EMT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMT_DATA_DIR / "sensitivity_matrix_A.npy", A)
    np.save(EMT_DATA_DIR / "baseline_voltages_v0.npy", v0)
    np.save(EMT_DATA_DIR / "element_centroids.npy",    CENTROIDS)
    np.save(EMT_DATA_DIR / "electrode_coords.npy",     el_coords)
    print(f"  Sensitivity matrix A saved → {EMT_DATA_DIR}/sensitivity_matrix_A.npy")

    return n_elements, N_MEAS


def make_phantom(args, n_elements_local, centroids_local):
    """Generate a random phantom with 1–3 circular conductive objects."""
    n_objects  = np.random.randint(1, 4)
    perm_elem  = args.sigma_bg * np.ones(n_elements_local)
    placed     = []
    attempts   = 0

    while len(placed) < n_objects and attempts < 300:
        attempts += 1
        r       = np.random.uniform(args.obj_r_min, args.obj_r_max)
        margin  = r + 0.04
        max_cen = args.domain_r - margin
        if max_cen <= 0:
            continue
        angle = np.random.uniform(0, 2 * np.pi)
        dist  = np.random.uniform(0, max_cen)
        cx, cy = dist * np.cos(angle), dist * np.sin(angle)

        overlap = any(
            np.sqrt((cx - px) ** 2 + (cy - py) ** 2) < (r + pr + 0.03)
            for (px, py, pr) in placed
        )
        if overlap:
            continue

        in_circle = (
            (centroids_local[:, 0] - cx) ** 2 +
            (centroids_local[:, 1] - cy) ** 2
        ) <= r ** 2
        perm_elem[in_circle] = args.sigma_obj
        placed.append((cx, cy, r))

    # Render onto pixel grid
    delta_perm = perm_elem - args.sigma_bg
    img_64 = griddata(
        centroids_local, delta_perm, (XX, YY),
        method="linear", fill_value=0.0,
    )
    img_64[~DOMAIN_MASK] = 0.0
    return perm_elem, img_64.astype(np.float32), len(placed)


def add_noise(signal, snr_db):
    power = np.mean(signal ** 2)
    snr   = 10 ** (snr_db / 10.0)
    std   = np.sqrt(power / (snr + 1e-12))
    return signal + np.random.randn(*signal.shape).astype(np.float32) * std


def generate_split(split_name, n_samples, args, n_elements_local, N_MEAS_local):
    print(f"\nGenerating {split_name} ({n_samples} samples, SNR={args.noise_db} dB) …")
    images    = np.zeros((n_samples, args.img_size, args.img_size), dtype=np.float32)
    meas      = np.zeros((n_samples, N_MEAS_local),                 dtype=np.float32)
    n_obj_log = np.zeros(n_samples,                                  dtype=np.int32)

    for i in range(n_samples):
        perm_elem, img_64, n_obj = make_phantom(args, n_elements_local, CENTROIDS)
        v1      = fwd_solver.solve_eit(perm=perm_elem)
        b_clean = v1 - v0
        b_noisy = add_noise(b_clean, args.noise_db)

        images[i]    = img_64
        meas[i]      = b_noisy.astype(np.float32)
        n_obj_log[i] = n_obj

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:4d}/{n_samples}] "
                  f"1-obj:{(n_obj_log[:i+1]==1).sum()}  "
                  f"2-obj:{(n_obj_log[:i+1]==2).sum()}  "
                  f"3-obj:{(n_obj_log[:i+1]==3).sum()}")

    h5_path = EMT_DATA_DIR / f"emt_{split_name}.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("images",       data=images, compression="gzip", compression_opts=4)
        f.create_dataset("measurements", data=meas,   compression="gzip", compression_opts=4)
        f.create_dataset("n_objects",    data=n_obj_log)
        f.attrs.update({
            "description": "Synthetic EMT dataset (FEM-based, pyeit)",
            "n_samples":   n_samples,
            "img_size":    args.img_size,
            "n_coils":     args.n_coils,
            "n_meas":      N_MEAS_local,
            "noise_db":    args.noise_db,
            "sigma_bg":    args.sigma_bg,
            "sigma_obj":   args.sigma_obj,
            "mesh_h0":     args.mesh_h0,
            "n_elements":  n_elements_local,
        })

    size_mb = h5_path.stat().st_size / 1e6
    print(f"  Saved → {h5_path}  ({size_mb:.1f} MB)")
    return images, meas, n_obj_log


def run(args):
    np.random.seed(42)
    n_elem, n_meas = setup(args)

    train_imgs, train_meas, train_nobj = generate_split("train", args.n_train, args, n_elem, n_meas)
    test_imgs,  test_meas,  test_nobj  = generate_split("test",  args.n_test,  args, n_elem, n_meas)

    print("\n" + "=" * 55)
    print("  EMT DATASET GENERATION COMPLETE")
    print("=" * 55)
    print(f"  Train : {args.n_train} samples")
    print(f"  Test  : {args.n_test}  samples")
    print(f"  Saved to: {EMT_DATA_DIR}")
    print(f"  Measurement dim : {n_meas}")
    print(f"  Image size      : {args.img_size}x{args.img_size}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate FEM-based EMT dataset")
    p.add_argument("--n_train",   type=int,   default=EMT["n_train"])
    p.add_argument("--n_test",    type=int,   default=EMT["n_test"])
    p.add_argument("--n_coils",   type=int,   default=EMT["n_coils"])
    p.add_argument("--mesh_h0",   type=float, default=EMT["mesh_h0"])
    p.add_argument("--dist_exc",  type=int,   default=EMT["dist_exc"])
    p.add_argument("--img_size",  type=int,   default=EMT["img_size"])
    p.add_argument("--domain_r",  type=float, default=EMT["domain_r"])
    p.add_argument("--sigma_bg",  type=float, default=EMT["sigma_bg"])
    p.add_argument("--sigma_obj", type=float, default=EMT["sigma_obj"])
    p.add_argument("--obj_r_min", type=float, default=EMT["obj_r_min"])
    p.add_argument("--obj_r_max", type=float, default=EMT["obj_r_max"])
    p.add_argument("--noise_db",  type=float, default=EMT["noise_db"])
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
