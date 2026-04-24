"""
Phase 2 – Restoration Dataset
==============================
PyTorch Dataset that loads clean images from Phase 1 split CSVs,
applies synthetic degradation on-the-fly, and returns
(degraded_tensor, clean_tensor) pairs for the SMP restoration model.
"""

import random
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Reuse Phase 1 preprocessing utilities
from preprocessing import standardize_and_pad, degrade_image


class RestorationDataset(Dataset):
    """
    Loads clean images, standardises them, and applies degradation.

    For **training**, degradation is fully stochastic (different every epoch).
    For **validation/test**, a per-index seed makes degradation deterministic
    so metrics are comparable across runs.
    """

    def __init__(
        self,
        csv_path: str,
        target_height: int = 128,
        target_width: int = 512,
        deterministic_degradation: bool = False,
        base_seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        self.records = pd.read_csv(csv_path)
        if max_samples is not None:
            self.records = self.records.head(max_samples)
        self.target_height = target_height
        self.target_width = target_width
        self.deterministic = deterministic_degradation
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[idx]
        img_path = row["image_path"]

        # Load image (grayscale or colour — standardize_and_pad handles it)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Fallback: return a blank pair so DataLoader doesn't crash
            blank = np.full(
                (self.target_height, self.target_width), 255, dtype=np.uint8
            )
            t = torch.from_numpy(blank).float().unsqueeze(0) / 255.0
            return t, t

        # Standardise to uniform size
        clean = standardize_and_pad(img, self.target_height, self.target_width)

        # Apply degradation
        if self.deterministic:
            rng = random.Random(self.base_seed + idx)
        else:
            rng = None  # fully stochastic
        degraded = degrade_image(clean, rng=rng)

        # Convert to float32 tensors in [0, 1], shape (1, H, W)
        clean_t = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0) / 255.0
        degraded_t = torch.from_numpy(degraded.astype(np.float32)).unsqueeze(0) / 255.0

        return degraded_t, clean_t


def get_dataloaders(
    splits_dir: str = "splits",
    target_height: int = 128,
    target_width: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from the Phase 1 split CSVs.
    Returns a dict: {"train": ..., "val": ..., "test": ...}
    """
    splits_dir = Path(splits_dir)

    train_ds = RestorationDataset(
        csv_path=str(splits_dir / "train.csv"),
        target_height=target_height,
        target_width=target_width,
        deterministic_degradation=False,
        max_samples=max_train_samples,
    )
    val_ds = RestorationDataset(
        csv_path=str(splits_dir / "val.csv"),
        target_height=target_height,
        target_width=target_width,
        deterministic_degradation=True,
        max_samples=max_val_samples,
    )
    test_ds = RestorationDataset(
        csv_path=str(splits_dir / "test.csv"),
        target_height=target_height,
        target_width=target_width,
        deterministic_degradation=True,
        max_samples=max_test_samples,
    )

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return loaders
