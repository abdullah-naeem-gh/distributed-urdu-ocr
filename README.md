# Distributed Deep Learning-Based Urdu OCR & Restoration System

This project is an end-to-end Optical Character Recognition (OCR) pipeline designed specifically for Urdu text. It features a dual-model architecture: a U-Net based Image Restoration model to clean degraded documents, followed by a Conv-Transformer sequence-to-sequence model to transcribe the text.

The system is trained on a combination of printed Nastaleeq text (`MMU-OCR-21`) and handwritten text (`UHWR`), ensuring robustness across different writing styles and qualities.

---

## 🚀 Project Status
- **Phase 1: Data Preprocessing & Pipeline Construction** — **[COMPLETED]**
- **Phase 2: Deep Learning Image Restoration (U-Net)** — **[COMPLETED]**
- **Phase 3: Deep Learning OCR (Conv-Transformer)** — **[COMPLETED]**
- **Phase 4: Pipeline Integration, Metrics & Visualization** — **[COMPLETED]**
- **Phase 5: Real-world Application / Deployment** — *[PENDING]*

---

## 🛠️ Architecture Overview

### 1. Data Preprocessing (Phase 1)
- **Standardization:** All images are padded/scaled to a uniform `128×2048` size to prevent cropping of wide handwritten images while maintaining aspect ratios.
- **Stochastic Degradation:** Synthetic noise (Gaussian blur, salt & pepper, low contrast, affine skew) is applied on-the-fly to train the restoration model.
- **Splitting:** 70% Train / 15% Val / 15% Test with zero data leakage.

### 2. Image Restoration Model (Phase 2)
- **Architecture:** Pre-trained `ResNet-34` U-Net (via `segmentation-models-pytorch`).
- **Goal:** Take noisy, degraded, or skewed document images and reconstruct clean, high-contrast text.
- **Metrics:** Mean Squared Error (MSE), PSNR, and SSIM.

### 3. OCR Sequence Model (Phase 3)
- **Architecture:** 
  - **Encoder:** Custom 7-block CNN Backbone that extracts spatial features, pooling the `128×2048` image down to a sequence of `128` tokens (`d_model=256`).
  - **Decoder:** 6-layer Transformer (3 Encoder / 3 Decoder) utilizing standard sinusoidal positional encoding and self-attention.
- **Training Strategy:** Implements a **source-weighted Cross-Entropy loss** (UHWR samples are weighted 3.0×, MMU-OCR-21 weighted 1.0×) to force the model to prioritize difficult handwritten text despite class imbalance.
- **Decoding:** Character-level Beam Search with length penalty.
- **Vocabulary:** 173 tokens covering Urdu characters, digits, punctuation, and control tokens (`<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`).

---

## 💻 Installation & Setup

To run this project locally, ensure you have Python installed, then set up the PyTorch environment:

```bash
# 1. Create and activate a virtual environment
python -m venv torch-env
torch-env\Scripts\activate

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining project dependencies
pip install -r requirements.txt
```

### Optional: Install Tesseract OCR (Urdu) for baseline comparison
`benchmark.py` compares this project against Tesseract (`pytesseract`, language code `urd`).  
To install Tesseract and ensure the Urdu language pack is present, run the included Windows script from the project root:

```powershell
.\install_tesseract_urdu.bat
```

If the script reports `urd` in `tesseract --list-langs`, the Tesseract baseline is ready for comparison.

### Optional: Jupyter Notebook Kernel
If you plan to run or modify the `project-notebook.ipynb`, you can register the environment as a Jupyter kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name=torch-env --display-name "Python (torch-env)"
```

---

## 📁 Project Structure

```text
├── splits/                    # Contains generated train/val/test CSVs
├── checkpoints/               # Saved model weights (.pth) and vocab.json
├── models/
│   ├── vocab.py               # Vocabulary builder and tokenization
│   ├── restoration_model.py   # U-Net architecture & builder
│   ├── ocr_model.py           # CNN + Transformer architecture
│   ├── ocr_trainer.py         # Custom training loops & weighted loss
│   └── pipeline.py            # End-to-End inference pipeline wrapper
├── datasets/
│   ├── restoration_dataset.py # On-the-fly degradation dataloader
│   └── ocr_dataset.py         # Sequence padding & source-weight dataloader
├── preprocessing.py           # Standardization and augmentation pipelines
├── benchmark.py               # Benchmarking script to evaluate CER/WER against Tesseract
├── install_tesseract_urdu.bat # Installs Tesseract + Urdu language pack for baseline comparison
├── project-notebook.ipynb     # Main orchestrator for training & visualization
└── requirements.txt           # Dependencies
```

---

## 📌 Next Steps
With Phases 1-4 complete, the upcoming objective is **Phase 5** (wrapping the dual-model pipeline into a FastAPI deployment and setting up distributed processing scaling).

## RunPod Serverless Inference

Use the serverless inference image in `deployment/inference/`:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f deployment/inference/Dockerfile \
  -t <registry>/<repo>:runpod-serverless \
  --push .
```

Set the RunPod worker environment:

- `DEVICE=cuda`
- `RESTORATION_CKPT=/app/checkpoints/best_restoration_model.pth`
- `OCR_CKPT=/app/checkpoints/best_ocr_model.pth`
- `VOCAB_PATH=/app/checkpoints/vocab.json`
