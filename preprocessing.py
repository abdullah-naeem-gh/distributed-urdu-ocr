"""
Phase 1 – Data Preprocessing & Synthetic Degradation
=====================================================
Provides:
  - Dataset loading (MMU-OCR-21 & UHWR)
  - Image standardization (resize + pad to uniform height, grayscale)
  - Synthetic degradation pipeline (blur, noise, contrast, skew)
  - Train / Val / Test splitting with zero data-leakage guarantee
  - Visual helpers for side-by-side comparison grids
"""

import os
import random
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. Dataset Discovery
# ---------------------------------------------------------------------------

def discover_mmu_ocr21(dataset_root: str, level: str = "TextLines") -> List[Dict]:
    """
    Walk MMU-OCR-21 and return a list of dicts:
      {"image_path": str, "label": str, "source": "MMU-OCR-21", "font": "Nastaleeq"}
    """
    base = Path(dataset_root) / "MMU-OCR-21"
    
    # Map levels to their exact Nastaleeq paths
    level_map = {
        "textlines": {
            "img_dir": base / "TextLines" / "SentenceImages" / "Nastaleeq",
            "csv_path": base / "TextLines" / "UrduTextLineNastaleeqOutput.csv"
        },
        "words": {
            "img_dir": base / "Words" / "WordImages" / "Nastaleeq",
            "csv_path": base / "Words" / "UrduWordsNastaleeqOutput.csv"
        },
        "characters": {
            "img_dir": base / "Characters" / "CharacterImages" / "Nastaleeq",
            "csv_path": base / "Characters" / "UrduCharactersNastaleeqOutput.csv"
        }
    }
    
    level_lower = level.lower()
    if level_lower not in level_map:
        raise ValueError(f"Unknown level: {level}. Choose from TextLines, Words, Characters.")
        
    paths = level_map[level_lower]
    img_dir = paths["img_dir"]
    csv_path = paths["csv_path"]

    if not img_dir.exists():
        raise FileNotFoundError(f"Nastaleeq image directory not found: {img_dir}")
        
    gt_map: Dict[str, str] = {}
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
            except Exception:
                df = pd.DataFrame()
                
        if not df.empty:
            df.columns = [c.strip().lower() for c in df.columns]
            fname_col = next((c for c in df.columns if c in ("filename", "image", "file", "img", "image_name")), None)
            label_col = next((c for c in df.columns if c in ("text", "label", "ground_truth", "gt", "transcription")), None)
            if fname_col is None or label_col is None:
                if len(df.columns) >= 2:
                    fname_col, label_col = df.columns[0], df.columns[1]
            if fname_col is not None and label_col is not None:
                for _, row in df.iterrows():
                    key = str(row[fname_col]).strip()
                    gt_map[key] = str(row[label_col]).strip()

    records = []
    for img_file in img_dir.rglob("*"):
        if not img_file.is_file():
            continue
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        fname = img_file.name
        stem = img_file.stem
        label = gt_map.get(fname) or gt_map.get(stem) or ""
        records.append({
            "image_path": str(img_file),
            "label": label,
            "source": "MMU-OCR-21",
            "font": "Nastaleeq",
        })

    return records


def discover_uhwr(dataset_root: str) -> List[Dict]:
    """
    Walk UHWR dataset and return a list of dicts:
      {"image_path": str, "label": str, "source": "UHWR"}

    Expected layout:
      <dataset_root>/UHWR/images/  (or similar)
      <dataset_root>/UHWR/*.csv    (ground truth)
    """
    base = Path(dataset_root) / "UHWR"
    if not base.exists():
        raise FileNotFoundError(f"UHWR not found at {base}")

    # Load CSVs for ground truth
    gt_map: Dict[str, str] = {}
    csv_files = list(base.glob("*.csv"))
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig", on_bad_lines="skip")
            except Exception:
                continue
        df.columns = [c.strip().lower() for c in df.columns]
        fname_col = next((c for c in df.columns if c in ("filename", "image", "file", "img", "image_name")), None)
        label_col = next((c for c in df.columns if c in ("text", "label", "ground_truth", "gt", "transcription")), None)
        if fname_col is None or label_col is None:
            if len(df.columns) >= 2:
                fname_col, label_col = df.columns[0], df.columns[1]
            else:
                continue
        for _, row in df.iterrows():
            key = str(row[fname_col]).strip()
            gt_map[key] = str(row[label_col]).strip()

    # Walk all image files under base
    records = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for img_file in sorted(base.rglob("*")):
        if img_file.suffix.lower() not in image_exts:
            continue
        fname = img_file.name
        stem = img_file.stem
        label = gt_map.get(fname) or gt_map.get(stem) or ""
        records.append({
            "image_path": str(img_file),
            "label": label,
            "source": "UHWR",
        })

    return records


# ---------------------------------------------------------------------------
# 2. Image Standardization
# ---------------------------------------------------------------------------

def standardize_image(img: np.ndarray, target_height: int = 128) -> np.ndarray:
    """
    Resize *img* to *target_height* preserving aspect ratio, then pad
    the width to make the result a consistent width (the original
    scaled width, no extra padding needed for this stage — downstream
    batching will handle variable widths via collate).

    Converts to grayscale if not already.
    Returns a uint8 grayscale image of shape (target_height, new_width).
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_height, target_height), dtype=np.uint8)

    scale = target_height / h
    new_w = max(1, int(w * scale))
    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized


def standardize_and_pad(img: np.ndarray, target_height: int = 128,
                        target_width: int = 512,
                        pad_value: int = 255) -> np.ndarray:
    """
    Standardize height, then pad (or crop) width to *target_width*.
    Padding is added on the LEFT side (since Urdu is RTL, the text
    starts from the right edge).
    """
    resized = standardize_image(img, target_height)
    h, w = resized.shape[:2]

    if w >= target_width:
        # Center-crop width
        start = (w - target_width) // 2
        return resized[:, start:start + target_width]

    # Pad on the left
    pad_left = target_width - w
    padded = np.full((target_height, target_width), pad_value, dtype=np.uint8)
    padded[:, pad_left:] = resized
    return padded


# ---------------------------------------------------------------------------
# 3. Synthetic Degradation Pipeline
# ---------------------------------------------------------------------------

def apply_gaussian_blur(img: np.ndarray, ksize_range: Tuple[int, int] = (3, 7)) -> np.ndarray:
    """Apply random Gaussian blur."""
    k = random.choice(range(ksize_range[0], ksize_range[1] + 1, 2))  # must be odd
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_gaussian_noise(img: np.ndarray, mean: float = 0,
                         std_range: Tuple[float, float] = (10, 40)) -> np.ndarray:
    """Add Gaussian noise."""
    std = random.uniform(*std_range)
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_salt_pepper_noise(img: np.ndarray,
                            amount_range: Tuple[float, float] = (0.01, 0.05)) -> np.ndarray:
    """Add salt-and-pepper noise."""
    amount = random.uniform(*amount_range)
    noisy = img.copy()
    # Salt
    num_salt = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, max(1, d), num_salt) for d in img.shape]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    num_pepper = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, max(1, d), num_pepper) for d in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy


def apply_low_contrast(img: np.ndarray,
                       alpha_range: Tuple[float, float] = (0.4, 0.7),
                       beta_range: Tuple[int, int] = (30, 80)) -> np.ndarray:
    """Simulate faded / low-contrast text."""
    alpha = random.uniform(*alpha_range)
    beta = random.randint(*beta_range)
    faded = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return faded


def apply_affine_skew(img: np.ndarray,
                      max_angle: float = 5.0) -> np.ndarray:
    """Apply a small affine skew/rotation."""
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)
    return rotated


def degrade_image(img: np.ndarray, rng: Optional[random.Random] = None) -> np.ndarray:
    """
    Apply a random combination of degradations to a clean image.
    Each degradation has a probability of being applied, making
    the pipeline stochastic.
    """
    if rng is None:
        rng = random

    degraded = img.copy()

    # Gaussian blur (70% chance)
    if rng.random() < 0.7:
        degraded = apply_gaussian_blur(degraded)

    # Noise: pick one type (80% chance)
    if rng.random() < 0.8:
        if rng.random() < 0.5:
            degraded = apply_gaussian_noise(degraded)
        else:
            degraded = apply_salt_pepper_noise(degraded)

    # Low contrast / faded (50% chance)
    if rng.random() < 0.5:
        degraded = apply_low_contrast(degraded)

    # Affine skew (40% chance)
    if rng.random() < 0.4:
        degraded = apply_affine_skew(degraded)

    return degraded


# ---------------------------------------------------------------------------
# 4. Data Splitting (70 / 15 / 15)
# ---------------------------------------------------------------------------

def split_dataset(records: List[Dict],
                  train_ratio: float = 0.70,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split records into train / val / test with deterministic seeding.
    Returns (train, val, test) lists.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # First split: train vs (val+test)
    train, valtest = train_test_split(
        records, train_size=train_ratio, random_state=seed, shuffle=True
    )
    # Second split: val vs test (from the remaining portion)
    relative_val = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        valtest, train_size=relative_val, random_state=seed, shuffle=True
    )
    return train, val, test


def assert_no_data_leakage(train: List[Dict], val: List[Dict], test: List[Dict]) -> None:
    """Raise AssertionError if any image path appears in more than one split."""
    train_paths = {r["image_path"] for r in train}
    val_paths = {r["image_path"] for r in val}
    test_paths = {r["image_path"] for r in test}

    tv = train_paths & val_paths
    tt = train_paths & test_paths
    vt = val_paths & test_paths

    assert len(tv) == 0, f"Train-Val overlap: {len(tv)} images"
    assert len(tt) == 0, f"Train-Test overlap: {len(tt)} images"
    assert len(vt) == 0, f"Val-Test overlap: {len(vt)} images"

    total = len(train_paths) + len(val_paths) + len(test_paths)
    unique = len(train_paths | val_paths | test_paths)
    assert total == unique, f"Duplicate images detected: {total} vs {unique} unique"

    print(f"✓ No data leakage detected. "
          f"Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")


# ---------------------------------------------------------------------------
# 5. Visualisation Helpers
# ---------------------------------------------------------------------------

def make_comparison_grid(clean_imgs: List[np.ndarray],
                         degraded_imgs: List[np.ndarray],
                         n: int = 5,
                         target_h: int = 128,
                         target_w: int = 512) -> np.ndarray:
    """
    Build a side-by-side grid: left = clean, right = degraded.
    Returns a single image (numpy array) ready for display.
    """
    n = min(n, len(clean_imgs), len(degraded_imgs))
    rows = []
    for i in range(n):
        c = cv2.resize(clean_imgs[i], (target_w, target_h))
        d = cv2.resize(degraded_imgs[i], (target_w, target_h))
        # Add a thin white separator
        sep = np.full((target_h, 4), 200, dtype=np.uint8)
        row = np.hstack([c, sep, d])
        rows.append(row)

    # Horizontal separator between rows
    row_w = rows[0].shape[1]
    hsep = np.full((4, row_w), 200, dtype=np.uint8)
    grid_parts = []
    for i, row in enumerate(rows):
        grid_parts.append(row)
        if i < len(rows) - 1:
            grid_parts.append(hsep)

    return np.vstack(grid_parts)
