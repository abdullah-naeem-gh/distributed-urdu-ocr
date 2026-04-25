"""
Phase 3 – OCR Trainer
======================
Training loop, validation, checkpointing, and CER/WER metrics
for the Conv-Transformer OCR model.
"""

import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vocab import Vocabulary, PAD_IDX, EOS_IDX, SOS_IDX
from models.ocr_model import ConvTransformerOCR


# ---------------------------------------------------------------------------
# Metrics: CER and WER
# ---------------------------------------------------------------------------

def _edit_distance(ref: str, hyp: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Character Error Rate: edit_distance(pred, ref) / len(ref),
    averaged over all pairs.
    """
    if not references:
        return 0.0
    total_dist = 0
    total_len = 0
    for pred, ref in zip(predictions, references):
        total_dist += _edit_distance(ref, pred)
        total_len += max(len(ref), 1)
    return total_dist / total_len


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Word Error Rate: edit_distance(pred_words, ref_words) / len(ref_words),
    averaged over all pairs.
    """
    if not references:
        return 0.0
    total_dist = 0
    total_len = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        total_dist += _edit_distance(ref_words, pred_words)
        total_len += max(len(ref_words), 1)
    return total_dist / total_len


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class OCRTrainer:
    """
    Training loop for the Conv-Transformer OCR model.

    Handles:
      - Cross-entropy loss with PAD token masking
      - Adam optimizer with spec hyperparams
      - Validation with CER/WER computation via greedy decode
      - Best-model checkpointing
      - Training history logging
    """

    def __init__(
        self,
        model: ConvTransformerOCR,
        vocab: Vocabulary,
        device: torch.device,
        lr: float = 3e-4,
        betas: tuple = (0.9, 0.98),
        eps: float = 1e-9,
    ):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_cer": [],
            "val_wer": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        count = 0

        for images, labels, lengths in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Teacher forcing: input = labels[:, :-1], target = labels[:, 1:]
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]

            self.optimizer.zero_grad()
            logits = self.model(images, tgt_input)  # (B, tgt_len, vocab)

            # Flatten for cross-entropy
            B, T, V = logits.shape
            loss = self.criterion(
                logits.reshape(B * T, V),
                tgt_output.reshape(B * T),
            )
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * B
            count += B

        return total_loss / max(count, 1)

    @torch.no_grad()
    def _greedy_decode(self, images: torch.Tensor,
                       max_len: int = 200) -> List[List[int]]:
        """Greedy decode a batch of images (for fast validation)."""
        self.model.eval()
        memory = self.model.encode(images)
        B = images.size(0)

        # Start with SOS token
        ys = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=self.device)

        for _ in range(max_len):
            logits = self.model.decode(memory, ys)  # (B, cur_len, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            ys = torch.cat([ys, next_token], dim=1)

            # Stop if all sequences have produced EOS
            if (next_token.squeeze(-1) == EOS_IDX).all():
                break

        return ys.tolist()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader,
                 max_decode_len: int = 200) -> Dict[str, float]:
        """
        Validate using greedy decoding.
        Returns dict with 'loss', 'cer', 'wer'.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0
        all_preds = []
        all_refs = []

        for images, labels, lengths in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Compute loss
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]
            logits = self.model(images, tgt_input)
            B, T, V = logits.shape
            loss = self.criterion(
                logits.reshape(B * T, V),
                tgt_output.reshape(B * T),
            )
            total_loss += loss.item() * B
            count += B

            # Greedy decode for CER/WER
            decoded_seqs = self._greedy_decode(images, max_decode_len)
            for i in range(B):
                pred_text = self.vocab.decode(decoded_seqs[i], strip_special=True)
                ref_text = self.vocab.decode(labels[i].tolist(), strip_special=True)
                all_preds.append(pred_text)
                all_refs.append(ref_text)

        cer = compute_cer(all_preds, all_refs)
        wer = compute_wer(all_preds, all_refs)

        return {
            "loss": total_loss / max(count, 1),
            "cer": cer,
            "wer": wer,
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
        Full training loop with validation and best-model checkpointing.
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            msg = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}"

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_cer"].append(val_metrics["cer"])
                self.history["val_wer"].append(val_metrics["wer"])

                msg += (
                    f" | Val Loss: {val_metrics['loss']:.4f}"
                    f" | CER: {val_metrics['cer']:.4f}"
                    f" | WER: {val_metrics['wer']:.4f}"
                )

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    path = os.path.join(save_dir, "best_ocr_model.pth")
                    torch.save(self.model.state_dict(), path)
                    msg += " *"

            elapsed = time.time() - t0
            msg += f" | {elapsed:.1f}s"

            if verbose:
                print(msg)

        # Save final model
        final_path = os.path.join(save_dir, "final_ocr_model.pth")
        torch.save(self.model.state_dict(), final_path)

        return self.history
