"""
Phase 2 – Image Restoration Model
===================================
SMP-based U-Net for denoising/restoring degraded Urdu document images.

Provides:
  - build_restoration_model()  — creates the SMP U-Net
  - RestorationTrainer          — training loop + validation + checkpointing
  - compute_psnr / compute_ssim — evaluation metrics
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# 1. Model Factory
# ---------------------------------------------------------------------------

def build_restoration_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 1,
    classes: int = 1,
) -> nn.Module:
    """
    Build an SMP U-Net for image restoration (image-to-image regression).

    Parameters match the Phase 2 spec:
      - encoder_name  : "resnet34" (robust pre-optimised backbone)
      - encoder_weights: "imagenet" (transfer learning for edge/stroke detection)
      - in_channels   : 1 (grayscale degraded input)
      - classes        : 1 (grayscale restored output)
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation="sigmoid",  # output pixel values in [0, 1]
    )
    return model


# ---------------------------------------------------------------------------
# 2. Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor,
                 max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio between pred and target tensors.
    Both should be float tensors in [0, max_val].
    Returns PSNR in dB.
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0  # effectively identical
    psnr = 10.0 * np.log10(max_val ** 2 / mse)
    return psnr


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Structural Similarity Index between pred and target.
    Uses scikit-image under the hood for correctness.
    Tensors are converted to numpy for computation.
    """
    from skimage.metrics import structural_similarity as sk_ssim

    # Move to CPU and convert to numpy
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()

    # Handle batched input: average SSIM across batch
    if p.ndim == 4:  # (B, C, H, W)
        ssim_vals = []
        for i in range(p.shape[0]):
            # squeeze channel dim for grayscale
            pi = p[i].squeeze()
            ti = t[i].squeeze()
            ssim_vals.append(
                sk_ssim(ti, pi, data_range=1.0)
            )
        return float(np.mean(ssim_vals))
    else:
        p = p.squeeze()
        t = t.squeeze()
        return float(sk_ssim(t, p, data_range=1.0))


# ---------------------------------------------------------------------------
# 3. Trainer
# ---------------------------------------------------------------------------

class RestorationTrainer:
    """
    Encapsulates training loop, validation, checkpointing, and metric logging
    for the SMP restoration model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        loss_fn: str = "mse",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        # History for plotting
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_psnr": [],
            "val_ssim": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch. Returns average training loss."""
        self.model.train()
        total_loss = 0.0
        count = 0

        for degraded, clean in train_loader:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(degraded)
            loss = self.criterion(pred, clean)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * degraded.size(0)
            count += degraded.size(0)

        return total_loss / max(count, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on a validation/test loader.
        Returns dict with 'loss', 'psnr', 'ssim'.
        """
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0
        n_batches = 0

        for degraded, clean in val_loader:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)

            pred = self.model(degraded)
            loss = self.criterion(pred, clean)

            total_loss += loss.item() * degraded.size(0)
            total_psnr += compute_psnr(pred, clean)
            total_ssim += compute_ssim(pred, clean)
            count += degraded.size(0)
            n_batches += 1

        n_batches = max(n_batches, 1)
        return {
            "loss": total_loss / max(count, 1),
            "psnr": total_psnr / n_batches,
            "ssim": total_ssim / n_batches,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_dir: str = "checkpoints",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with optional validation tracking and
        best-model checkpointing.
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            msg = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f}"

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_psnr"].append(val_metrics["psnr"])
                self.history["val_ssim"].append(val_metrics["ssim"])

                msg += (
                    f" | Val Loss: {val_metrics['loss']:.6f}"
                    f" | PSNR: {val_metrics['psnr']:.2f} dB"
                    f" | SSIM: {val_metrics['ssim']:.4f}"
                )

                # Save best model
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    path = os.path.join(save_dir, "best_restoration_model.pth")
                    torch.save(self.model.state_dict(), path)
                    msg += " ★"

            elapsed = time.time() - t0
            msg += f" | {elapsed:.1f}s"

            if verbose:
                print(msg)

        # Always save final model
        final_path = os.path.join(save_dir, "final_restoration_model.pth")
        torch.save(self.model.state_dict(), final_path)

        return self.history


def load_restoration_model(
    checkpoint_path: str,
    device: torch.device,
    encoder_name: str = "resnet34",
) -> nn.Module:
    """Load a trained restoration model from a checkpoint file."""
    model = build_restoration_model(encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
