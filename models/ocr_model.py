"""
Phase 3 – Conv-Transformer Urdu OCR Model
===========================================
Sequence-to-sequence model: CNN Backbone → Positional Encoding →
Transformer (3 Enc + 3 Dec) → Character-level output.

Provides:
  - CNNBackbone       — stacked Conv layers with LeakyReLU + BatchNorm
  - PositionalEncoding — sinusoidal (Vaswani et al.)
  - ConvTransformerOCR — full end-to-end model with beam search
  - load_ocr_model     — checkpoint loading for deployment
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.checkpoint_utils import extract_state_dict, strip_module_prefix
from models.vocab import PAD_IDX, SOS_IDX, EOS_IDX


# ---------------------------------------------------------------------------
# 1. CNN Backbone (Feature Extraction)
# ---------------------------------------------------------------------------

class CNNBackbone(nn.Module):
    """
    Stacked convolutional feature extractor.

    Design per spec:
      - Conv layers with LeakyReLU activations and BatchNorm
      - MaxPool ONLY for dimensionality reduction
      - Spatial resolution preserved in Conv layers via padding=1
      - Output: (batch, seq_len, d_model)

    For input (B, 1, 128, 2048):
      Height: 128 → pooled down to 1 (7 vertical reductions)
      Width:  2048 → pooled to 128 (4 horizontal reductions)
      → seq_len=128, d_model=256
    """

    def __init__(self, in_channels: int = 1, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Channel progression: 1 → 32 → 64 → 128 → 128 → 256 → 256 → 256
        # Pool strategy: first 2 blocks pool (2,2), rest pool (2,1)
        self.layers = nn.Sequential(
            # Block 1: (B, 1, 128, 512) → pool → (B, 32, 64, 256)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Block 2: → pool → (B, 64, 32, 128)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Block 3: → pool (2,2) → (B, 128, 16, 64)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Block 4: → pool (2,2) → (B, 128, 8, 32)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Block 5: → pool (2,1) → (B, 256, 4, 128)
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Block 6: → pool (2,1) → (B, 256, 2, 128)
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Block 7: → pool (2,1) → (B, 256, 1, 128)
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) — grayscale image

        Returns:
            (B, seq_len, d_model) — feature sequence for Transformer
        """
        features = self.layers(x)          # (B, d_model, 1, seq_len)
        features = features.squeeze(2)     # (B, d_model, seq_len)
        features = features.permute(0, 2, 1)  # (B, seq_len, d_model)
        return features


# ---------------------------------------------------------------------------
# 2. Positional Encoding (Sinusoidal)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al. 2017).
    Injected into CNN feature maps before the Transformer encoder.
    """

    def __init__(self, d_model: int = 256, max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            (B, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 3. Conv-Transformer OCR Model
# ---------------------------------------------------------------------------

class ConvTransformerOCR(nn.Module):
    """
    Full sequence-to-sequence OCR model:
      CNN Backbone → Positional Encoding → Transformer (3 enc, 3 dec)
      → Linear projection → vocab_size logits

    Training: Teacher forcing with cross-entropy loss.
    Inference: Beam search decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # CNN feature extractor
        self.cnn = CNNBackbone(in_channels=1, d_model=d_model)

        # Positional encoding for encoder (CNN features)
        self.encoder_pe = PositionalEncoding(d_model, max_seq_len, dropout)

        # Target embedding + positional encoding for decoder
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.decoder_pe = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, sz: int,
                              device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal mask for decoder (bool type)."""
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def _make_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create padding mask: True where PAD, False elsewhere."""
        return tokens == PAD_IDX

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images through CNN + positional encoding.
        Args:
            images: (B, 1, H, W)
        Returns:
            memory: (B, seq_len, d_model)
        """
        features = self.cnn(images)            # (B, seq_len, d_model)
        memory = self.encoder_pe(features)     # (B, seq_len, d_model)
        return memory

    def decode(self, memory: torch.Tensor,
               tgt: torch.Tensor) -> torch.Tensor:
        """
        Decode from memory (encoder output) and target tokens.
        Args:
            memory: (B, src_len, d_model)
            tgt: (B, tgt_len) — token indices
        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_pe(tgt_emb)

        tgt_mask = self._generate_causal_mask(tgt.size(1), tgt.device)
        tgt_pad_mask = self._make_pad_mask(tgt)

        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        logits = self.output_proj(output)      # (B, tgt_len, vocab_size)
        return logits

    def forward(self, images: torch.Tensor,
                tgt_input: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass (training with teacher forcing).

        Args:
            images: (B, 1, H, W) — input images
            tgt_input: (B, tgt_len) — target tokens shifted right
                       (starts with SOS, does NOT include final EOS)

        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        memory = self.encode(images)
        logits = self.decode(memory, tgt_input)
        return logits

    @torch.no_grad()
    def beam_search_decode(
        self,
        image: torch.Tensor,
        beam_width: int = 5,
        max_len: int = 200,
        alpha: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """
        Character-level beam search with length penalty.

        Args:
            image: (1, 1, H, W) — single image
            beam_width: number of beams
            max_len: maximum output sequence length
            alpha: length penalty exponent (0.7 per spec)

        Returns:
            List of (decoded_indices, score) tuples, sorted by score desc.
        """
        self.eval()
        device = image.device

        memory = self.encode(image)  # (1, src_len, d_model)

        # Each beam: (token_indices, log_prob)
        beams = [([SOS_IDX], 0.0)]
        completed = []

        for step in range(max_len):
            candidates = []

            for tokens, score in beams:
                if tokens[-1] == EOS_IDX:
                    completed.append((tokens, score))
                    continue

                tgt = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = self.decode(memory, tgt)  # (1, len, vocab_size)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

                topk_probs, topk_idx = log_probs.topk(beam_width)

                for i in range(beam_width):
                    new_tokens = tokens + [topk_idx[i].item()]
                    new_score = score + topk_probs[i].item()
                    candidates.append((new_tokens, new_score))

            if not candidates:
                break

            # Score with length penalty: score / (length ** alpha)
            def penalized_score(item):
                tokens, score = item
                length = len(tokens) - 1  # exclude SOS
                if length <= 0:
                    length = 1
                return score / (length ** alpha)

            candidates.sort(key=penalized_score, reverse=True)
            beams = candidates[:beam_width]

            # Early stop if all beams are completed
            if all(b[0][-1] == EOS_IDX for b in beams):
                completed.extend(beams)
                break

        # Add any remaining incomplete beams
        completed.extend(beams)

        # Sort by penalized score
        def final_score(item):
            tokens, score = item
            length = max(len(tokens) - 1, 1)
            return score / (length ** alpha)

        completed.sort(key=final_score, reverse=True)
        return completed[:beam_width]


# ---------------------------------------------------------------------------
# 4. Model Loading for Deployment
# ---------------------------------------------------------------------------

def load_ocr_model(
    checkpoint_path: str,
    vocab_size: int,
    device: torch.device,
    d_model: int = 256,
) -> ConvTransformerOCR:
    """Load a trained OCR model from a checkpoint file."""
    model = ConvTransformerOCR(vocab_size=vocab_size, d_model=d_model)
    checkpoint_obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint_obj, checkpoint_path))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load OCR checkpoint '{checkpoint_path}'. "
            "This usually means model architecture or vocab size does not match inference config."
        ) from exc
    model = model.to(device)
    model.eval()
    return model
