"""
ct/dataset.py
=============
Two dataset modes — use whichever fits your setup:

  BoxCTDataset      — streams individual .IMA files from the Box zip via HTTP
                      range requests.  No full download.  Only fetches the
                      slices you configure (slices_per_patient).  Caches them
                      to  ct_cache/slices/  so subsequent epochs read from disk.

  MayoCTDataset     — reads .IMA files that are already on disk (local or
                      cloud-mounted drive).  Use this if you have the data
                      extracted somewhere already.

build_ct_loaders()  — convenience wrapper; pass box_token to use Box streaming,
                      omit it (or pass None) to read from data_root on disk.

RadonOperator       — batched CPU numpy radon / iradon (skimage), used in the
                      training loops in place of an explicit A matrix.
"""

import io, struct, zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import radon, iradon
import pydicom
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CT, CT_DATA_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Box constants  (public AAPM shared link)
# ─────────────────────────────────────────────────────────────────────────────
_BOX_SHARED_LINK = "https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h"
_BOX_FILE_ID     = "856956352254"   # full_3mm.zip inside Training_Image_Data/3mm B30


# ─────────────────────────────────────────────────────────────────────────────
# Seekable HTTP stream  (lets zipfile read ranges from a remote zip)
# ─────────────────────────────────────────────────────────────────────────────

class _BoxSeekableStream(io.RawIOBase):
    """
    Wraps HTTP range-request GET calls so that Python's zipfile module can
    seek inside a remote zip without downloading the whole file.

    Box returns a 302 redirect for file content — this class follows it once
    and then issues Range: bytes=X-Y requests against the CDN URL.
    """

    def __init__(self, box_token: str):
        super().__init__()
        self._token = box_token
        self._pos   = 0
        self._cdn_url, self._cdn_headers = self._get_cdn_url()
        head = requests.head(self._cdn_url, headers=self._cdn_headers,
                             allow_redirects=True, timeout=30)
        head.raise_for_status()
        self.size = int(head.headers["content-length"])

    def _get_cdn_url(self):
        headers = {
            "Authorization": f"Bearer {self._token}",
            "BoxApi":        f"shared_link={_BOX_SHARED_LINK}",
        }
        resp = requests.get(
            f"https://api.box.com/2.0/files/{_BOX_FILE_ID}/content",
            headers=headers, allow_redirects=False, stream=True, timeout=30,
        )
        if resp.status_code in (301, 302):
            return resp.headers["Location"], {}
        resp.raise_for_status()
        return resp.url, headers

    # ── io.RawIOBase interface ────────────────────────────────────────────────

    def readable(self):  return True
    def seekable(self):  return True

    def seek(self, pos, whence=io.SEEK_SET):
        if   whence == io.SEEK_SET: self._pos = pos
        elif whence == io.SEEK_CUR: self._pos += pos
        elif whence == io.SEEK_END: self._pos = self.size + pos
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def tell(self): return self._pos

    def readinto(self, b):
        n    = len(b)
        end  = min(self._pos + n - 1, self.size - 1)
        if self._pos > end:
            return 0
        h    = {**self._cdn_headers, "Range": f"bytes={self._pos}-{end}"}
        resp = requests.get(self._cdn_url, headers=h, timeout=60)
        if resp.status_code == 401:
            self._cdn_url, self._cdn_headers = self._get_cdn_url()
            h    = {**self._cdn_headers, "Range": f"bytes={self._pos}-{end}"}
            resp = requests.get(self._cdn_url, headers=h, timeout=60)
        resp.raise_for_status()
        data = resp.content
        n_read = len(data)
        b[:n_read] = data
        self._pos += n_read
        return n_read


# ─────────────────────────────────────────────────────────────────────────────
# Shared DICOM helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_ima_file(path: Path) -> np.ndarray:
    ds        = pydicom.dcmread(str(path))
    img       = ds.pixel_array.astype(np.float32)
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu        = img * slope + intercept
    return np.clip(
        (hu - CT["win_min"]) / (CT["win_max"] - CT["win_min"]), 0.0, 1.0
    ).astype(np.float32)


def make_sparse_sinogram(img: np.ndarray, n_views: int):
    angles   = np.linspace(0.0, 180.0, n_views, endpoint=False)
    sinogram = radon(img, theta=angles, circle=True).astype(np.float32)
    fbp_init = np.clip(
        iradon(sinogram, theta=angles, filter_name="ramp", circle=True),
        0.0, 1.0,
    ).astype(np.float32)
    return sinogram, fbp_init


# ─────────────────────────────────────────────────────────────────────────────
# Box streaming dataset  ← DEFAULT when box_token is provided
# ─────────────────────────────────────────────────────────────────────────────

class BoxCTDataset(Dataset):
    """
    Fetches ONLY the needed .IMA slices from the Box zip via HTTP range
    requests — no full zip download.  On first use it caches the individual
    DICOM files to  ct_cache/slices/  so subsequent epochs run entirely
    from local disk.

    Parameters
    ----------
    box_token          : Box developer token (regenerate every 60 min at
                         developer.box.com if it hasn't been cached yet).
    patients           : list of patient IDs, e.g. ["L067", "L096"]
    slices_per_patient : how many slices to use per patient (None = all)
    n_views            : number of Radon projection angles
    patch_size         : random crop size during training (None = full slice)
    cache_dir          : where to store the cached individual .IMA files
    """

    def __init__(
        self,
        box_token:          str,
        patients:           list = CT["train_patients"],
        slices_per_patient: int  = CT["slices_per_patient_train"],
        n_views:            int  = CT["n_views"],
        patch_size:         int  = CT["patch_size"],
        cache_dir:          Path = None,
    ):
        self.n_views    = n_views
        self.patch_size = patch_size
        self.samples    = []

        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "ct_cache" / "slices"
        cache_dir = Path(cache_dir)

        # Check how many files we still need to fetch
        all_cached = True
        needed = {}
        for pid in patients:
            pid_cache = cache_dir / pid
            existing  = sorted(pid_cache.glob("*.IMA")) if pid_cache.exists() else []
            want      = slices_per_patient or 9999
            if len(existing) < want:
                all_cached = False
                needed[pid] = want - len(existing)

        if not all_cached:
            print(f"Fetching slices from Box zip via HTTP range requests …")
            print(f"  (only downloads {slices_per_patient} slices/patient — no full zip)")
            stream = io.BufferedReader(_BoxSeekableStream(box_token), buffer_size=64 * 1024)
            with zipfile.ZipFile(stream, "r") as zf:
                all_names = sorted(zf.namelist())
                for pid in patients:
                    pid_cache = cache_dir / pid
                    pid_cache.mkdir(parents=True, exist_ok=True)
                    existing = sorted(pid_cache.glob("*.IMA"))
                    if slices_per_patient and len(existing) >= slices_per_patient:
                        print(f"  {pid}: already cached ({len(existing)} slices)")
                        self.samples.extend(existing[:slices_per_patient])
                        continue

                    patient_imas = sorted([
                        n for n in all_names
                        if f"/{pid}/" in n and n.upper().endswith(".IMA")
                    ])
                    want  = slices_per_patient or len(patient_imas)
                    fetch = patient_imas[:want]
                    for zip_name in fetch:
                        fname      = Path(zip_name).name
                        local_path = pid_cache / fname
                        if not local_path.exists():
                            local_path.write_bytes(zf.read(zip_name))
                        self.samples.append(local_path)
                    print(f"  {pid}: fetched {len(fetch)} slices → {pid_cache}")
        else:
            print("All slices already cached locally — skipping Box fetch.")
            for pid in patients:
                pid_cache = cache_dir / pid
                slices    = sorted(pid_cache.glob("*.IMA"))
                chosen    = slices[:slices_per_patient] if slices_per_patient else slices
                self.samples.extend(chosen)
                print(f"  {pid}: {len(chosen)} slices from cache")

        print(f"\nDataset ready: {len(self.samples)} slices | {n_views}-view sparse CT")
        assert self.samples, "No slices found — check your Box token and patient IDs"

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        gt        = load_ima_file(self.samples[idx])
        sino, fbp = make_sparse_sinogram(gt, self.n_views)
        if self.patch_size:
            p = self.patch_size; H, W = gt.shape
            r = np.random.randint(0, H - p)
            c = np.random.randint(0, W - p)
            gt   = gt[r:r+p, c:c+p]
            fbp  = fbp[r:r+p, c:c+p]
            sino = sino[r:r+p, :]
        return (
            torch.tensor(fbp,  dtype=torch.float32).unsqueeze(0),
            torch.tensor(sino, dtype=torch.float32).unsqueeze(0),
            torch.tensor(gt,   dtype=torch.float32).unsqueeze(0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Local-disk dataset  (use when data is already extracted somewhere)
# ─────────────────────────────────────────────────────────────────────────────

class MayoCTDataset(Dataset):
    """
    Reads .IMA DICOM slices from a local (or cloud-mounted) directory.

    Parameters
    ----------
    data_root          : root directory (CT_DATA_DIR from config, or any path)
    patients           : list of patient IDs
    n_views            : number of projection angles
    slices_per_patient : int or None  (None = use all)
    patch_size         : int or None  (None = use full 512×512 slice)
    """

    def __init__(
        self,
        data_root:          Path = CT_DATA_DIR,
        patients:           list = CT["train_patients"],
        n_views:            int  = CT["n_views"],
        slices_per_patient: int  = CT["slices_per_patient_train"],
        patch_size:         int  = CT["patch_size"],
    ):
        self.n_views    = n_views
        self.patch_size = patch_size
        self.samples    = []

        for pid in patients:
            patient_dir = Path(data_root) / pid / "full_3mm"
            if not patient_dir.exists():
                print(f"  WARNING: {patient_dir} not found — skipping {pid}")
                continue
            all_slices = sorted(patient_dir.glob("*.IMA"))
            chosen     = all_slices[:slices_per_patient] if slices_per_patient else all_slices
            if not chosen:
                print(f"  WARNING: no .IMA files in {patient_dir}")
                continue
            self.samples.extend(chosen)
            print(f"  {pid}: using {len(chosen)}/{len(all_slices)} slices")

        print(f"\nDataset ready: {len(self.samples)} slices | {n_views}-view sparse CT")
        assert self.samples, f"No slices found under {data_root}"

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        gt        = load_ima_file(self.samples[idx])
        sino, fbp = make_sparse_sinogram(gt, self.n_views)
        if self.patch_size:
            p = self.patch_size; H, W = gt.shape
            r = np.random.randint(0, H - p)
            c = np.random.randint(0, W - p)
            gt   = gt[r:r+p, c:c+p]
            fbp  = fbp[r:r+p, c:c+p]
            sino = sino[r:r+p, :]
        return (
            torch.tensor(fbp,  dtype=torch.float32).unsqueeze(0),
            torch.tensor(sino, dtype=torch.float32).unsqueeze(0),
            torch.tensor(gt,   dtype=torch.float32).unsqueeze(0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader builder
# ─────────────────────────────────────────────────────────────────────────────

def build_ct_loaders(
    box_token:   str  = None,
    data_root:   Path = CT_DATA_DIR,
    n_views:     int  = CT["n_views"],
    patch_size:  int  = CT["patch_size"],
    batch_size:  int  = CT["batch_size"],
    num_workers: int  = CT["num_workers"],
):
    """
    Returns (train_loader, val_loader, test_loader).

    If box_token is provided, uses BoxCTDataset (streams from Box, caches
    individual slices locally — no full zip download).

    If box_token is None, falls back to MayoCTDataset which reads from
    data_root on disk.
    """
    def _make(patients, spp, ps):
        if box_token:
            return BoxCTDataset(
                box_token=box_token, patients=patients,
                slices_per_patient=spp, n_views=n_views, patch_size=ps,
            )
        return MayoCTDataset(
            data_root=data_root, patients=patients,
            n_views=n_views, slices_per_patient=spp, patch_size=ps,
        )

    print("Building CT datasets …")
    print("  TRAIN:")
    train_ds = _make(CT["train_patients"], CT["slices_per_patient_train"], patch_size)
    print("  VAL:")
    val_ds   = _make(CT["val_patients"],   CT["slices_per_patient_val"],   patch_size)
    print("  TEST (full slices):")
    test_ds  = _make(CT["test_patients"],  CT["slices_per_patient_test"],  None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=CT["pin_memory"])
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Radon Operator  (CPU numpy — batched, used inside training loops)
# ─────────────────────────────────────────────────────────────────────────────

class RadonOperator:
    """
    Thin wrapper around skimage radon / iradon for batched use.
    All tensors are detached to CPU — the Radon step has no gradient;
    only μ, θ, ρ and ProxNet weights are differentiated.
    """

    def __init__(self, image_size: int = 128, n_views: int = CT["n_views"]):
        self.angles     = np.linspace(0.0, 180.0, n_views, endpoint=False)
        self.image_size = image_size

    def batch_forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        sinos = [
            radon(x_batch[i, 0].detach().cpu().numpy(),
                  theta=self.angles, circle=True)
            for i in range(x_batch.shape[0])
        ]
        return torch.tensor(np.stack(sinos), dtype=torch.float32).unsqueeze(1)

    def batch_adjoint(self, sino_batch: torch.Tensor) -> torch.Tensor:
        imgs = [
            iradon(sino_batch[i, 0].detach().cpu().numpy(),
                   theta=self.angles, filter_name="ramp", circle=True)
            for i in range(sino_batch.shape[0])
        ]
        return torch.tensor(np.stack(imgs), dtype=torch.float32).unsqueeze(1)
