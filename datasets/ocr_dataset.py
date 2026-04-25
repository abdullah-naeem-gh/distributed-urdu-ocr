"""
Phase 3 – OCR Dataset
======================
PyTorch Dataset that loads images with their text labels for the
Conv-Transformer OCR model. Filters to records with non-empty labels.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from preprocessing import standardize_and_pad
from models.vocab import Vocabulary, PAD_IDX


class OCRDataset(Dataset):
    """
    Loads images and encodes their text labels for sequence-to-sequence
    OCR training.

    Only records with non-empty labels are included.
    Returns (image_tensor, label_tensor, label_length, weight).
    """

    # Source-based weight mapping: handwritten (UHWR) gets higher weight
    DEFAULT_SOURCE_WEIGHTS = {"UHWR": 3.0, "MMU-OCR-21": 1.0}

    def __init__(
        self,
        csv_path: str,
        vocab: Vocabulary,
        target_height: int = 128,
        target_width: int = 2048,
        max_samples: Optional[int] = None,
        source_weights: Optional[Dict[str, float]] = None,
    ):
        df = pd.read_csv(csv_path)
        # Filter to records that have non-empty labels
        df["label"] = df["label"].fillna("").astype(str).str.strip()
        df = df[df["label"] != ""].reset_index(drop=True)
        if max_samples is not None:
            df = df.head(max_samples)
        self.records = df
        self.vocab = vocab
        self.target_height = target_height
        self.target_width = target_width
        self.source_weights = source_weights or self.DEFAULT_SOURCE_WEIGHTS

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
        row = self.records.iloc[idx]
        img_path = row["image_path"]
        label_text = row["label"]

        # Load and standardise image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.full(
                (self.target_height, self.target_width), 255, dtype=np.uint8
            )
        else:
            img = standardize_and_pad(img, self.target_height, self.target_width)

        # Image → tensor (1, H, W), float32 [0, 1]
        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0) / 255.0

        # Label → token indices (with SOS and EOS)
        label_indices = self.vocab.encode(label_text, add_sos=True, add_eos=True)
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        # Per-sample loss weight based on source dataset
        source = row.get("source", "MMU-OCR-21")
        weight = self.source_weights.get(source, 1.0)

        return img_tensor, label_tensor, len(label_indices), weight


def collate_ocr_batch(batch):
    """
    Custom collate function that pads label sequences to the
    max length in the batch.

    Returns:
        images: (B, 1, H, W)
        labels: (B, max_label_len) — padded with PAD_IDX
        lengths: (B,) — original label lengths
        weights: (B,) — per-sample loss weights
    """
    images, labels, lengths, weights = zip(*batch)
    images = torch.stack(images, dim=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=PAD_IDX)
    lengths = torch.tensor(lengths, dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.float32)
    return images, labels_padded, lengths, weights


def get_ocr_dataloaders(
    vocab: Vocabulary,
    splits_dir: str = "splits",
    target_height: int = 128,
    target_width: int = 2048,
    batch_size: int = 4,
    num_workers: int = 0,
    source_weights: Optional[Dict[str, float]] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders for OCR training.
    Returns a dict: {"train": ..., "val": ..., "test": ...}
    """
    from pathlib import Path
    splits_dir = Path(splits_dir)

    train_ds = OCRDataset(
        csv_path=str(splits_dir / "train.csv"),
        vocab=vocab,
        target_height=target_height,
        target_width=target_width,
        max_samples=max_train_samples,
        source_weights=source_weights,
    )
    val_ds = OCRDataset(
        csv_path=str(splits_dir / "val.csv"),
        vocab=vocab,
        target_height=target_height,
        target_width=target_width,
        max_samples=max_val_samples,
        source_weights=source_weights,
    )
    test_ds = OCRDataset(
        csv_path=str(splits_dir / "test.csv"),
        vocab=vocab,
        target_height=target_height,
        target_width=target_width,
        max_samples=max_test_samples,
        source_weights=source_weights,
    )

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate_ocr_batch,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_ocr_batch,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_ocr_batch,
        ),
    }
    return loaders
