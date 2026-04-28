# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Distributed Urdu OCR & Restoration System — a two-stage deep learning pipeline that first restores degraded Urdu document images (U-Net), then performs OCR on them (Conv-Transformer with beam search). Two datasets are used: MMU-OCR-21 (100k+ printed Nastaleeq samples) and UHWR (10k handwritten samples).

## Setup

```bash
python -m venv torch-env
source torch-env/bin/activate  # Windows: torch-env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Datasets are stored in a virtual disk (`dataset.vhdx`, excluded from git). Checkpoints are saved to `checkpoints/` (also excluded).

## Running the Pipeline

**Training & experimentation** — the Jupyter notebook is the primary entry point:
```bash
jupyter notebook project-notebook.ipynb
```

**Benchmarking against Tesseract:**
```bash
python benchmark.py \
  --test_csv splits/test.csv \
  --restoration_ckpt checkpoints/best_restoration_model.pth \
  --ocr_ckpt checkpoints/best_ocr_model.pth \
  --vocab checkpoints/vocab.json \
  [--device cuda] [--limit 100]
```

**Single-image inference:**
```python
from models.pipeline import UrduOCRPipeline
pipeline = UrduOCRPipeline(restoration_ckpt, ocr_ckpt, vocab_path, device)
restored_img, text = pipeline.predict("path/to/image.png")
```

## Architecture

### Image standardization
All images are resized to **128×2048** (H×W) with aspect-ratio preservation and zero-padding (`preprocessing.py`). This fixed size is assumed throughout the model stack.

### Phase 2 — Restoration (`models/restoration_model.py`)
- SMP U-Net with ResNet-34 encoder; input/output shape `(B, 1, 128, 2048)`
- `datasets/restoration_dataset.py` applies degradation on-the-fly: stochastic during training, seeded-deterministic during validation/test

### Phase 3 — OCR (`models/ocr_model.py`)
- CNNBackbone (7 conv blocks, 1→256 channels) compresses `(B, 1, 128, 2048)` → `(B, 128, 256)` sequence
- 6-layer Transformer (3 encoder + 3 decoder layers, 8 heads); vocabulary of 173 Urdu chars + special tokens
- Training uses teacher forcing; inference uses beam search (width=5, length penalty α=0.7)
- **Source-weighted loss**: UHWR handwritten samples get 3× weight vs. printed MMU-OCR-21 samples

### Vocabulary (`models/vocab.py`)
Character-level, JSON-serializable. Special token indices: `<PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3`. Always load from a saved `vocab.json` after the first training run; re-building from scratch changes token indices.

### Data splits
`splits/train.csv`, `splits/val.csv`, `splits/test.csv` — 70/15/15 stratified split created in Phase 1 of the notebook. Each row has `image_path`, `text`, and `source` columns.

## Key Design Decisions

- Urdu is RTL — text is rendered via `arabic_reshaper` + `python-bidi` before display; don't reverse strings manually.
- The CNN pooling is tuned so the sequence length fed to the Transformer is exactly 128 (not variable) — changing conv strides/pooling breaks this.
- Restoration is trained independently of OCR; checkpoints are loaded separately in the pipeline.
- Phase 5 (FastAPI distributed deployment) is not yet implemented.
