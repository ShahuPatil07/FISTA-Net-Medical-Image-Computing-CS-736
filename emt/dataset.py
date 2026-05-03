import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMT, EMT_DATASET_DIR


class EMTDataset(Dataset):
    _FILES = {
        "train": "train.pt",
        "val":   "val.pt",
        "test1": "test_set1.pt",
        "test2": "test_set2.pt",
    }

    def __init__(self, split: str = "train", data_dir: Path = EMT_DATASET_DIR):
        data_dir = Path(data_dir)
        if split not in self._FILES:
            raise ValueError(f"split must be one of {list(self._FILES)}, got '{split}'")
        pt_path = data_dir / self._FILES[split]
        assert pt_path.exists(), (
            f"Dataset file not found: {pt_path}\n"
            f"Expected pre-generated .pt files in {data_dir}"
        )
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.meas     = data["measurements"]
        self.x0       = data["x0"]
        self.phantoms = data["phantoms"]

        H = self.phantoms.shape[-1]
        print(f"EMTDataset [{split:5s}]: {len(self.meas):5d} samples | "
              f"meas_dim={self.meas.shape[1]} | image={H}×{H}")

    def __len__(self):
        return len(self.meas)

    def __getitem__(self, idx):
        return self.meas[idx], self.x0[idx], self.phantoms[idx]


def build_emt_loaders(
    data_dir:    Path = EMT_DATASET_DIR,
    batch_size:  int  = EMT["batch_size"],
    num_workers: int  = EMT["num_workers"],
    test_split:  str  = "test1",
    n_train:     int  = EMT.get("train_subset"),
):
    train_ds = EMTDataset("train", data_dir)
    if n_train is not None and n_train < len(train_ds):
        train_ds = Subset(train_ds, range(n_train))
        print(f"  Train subset: {n_train}/{EMT['n_train']} samples "
              f"({n_train/EMT['n_train']*100:.0f}%) | "
              f"{n_train // batch_size} batches/epoch")
    val_ds   = EMTDataset("val",   data_dir)
    test_ds  = EMTDataset(test_split, data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=EMT["pin_memory"])
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, val_loader, test_loader


def load_sensitivity_matrix(data_dir: Path = EMT_DATASET_DIR) -> np.ndarray:
    path = Path(data_dir) / "sensitivity_matrix_A.npy"
    assert path.exists(), f"sensitivity_matrix_A.npy not found in {data_dir}"
    return np.load(path)


def load_laplacian(data_dir: Path = EMT_DATASET_DIR) -> np.ndarray:
    path = Path(data_dir) / "laplacian_matrix_L.npy"
    assert path.exists(), f"laplacian_matrix_L.npy not found in {data_dir}"
    return np.load(path)
