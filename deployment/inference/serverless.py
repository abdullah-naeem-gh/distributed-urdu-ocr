import base64
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import runpod
import torch

from models.pipeline import UrduOCRPipeline

RESTORATION_CKPT = os.environ.get(
    "RESTORATION_CKPT", "/app/checkpoints/best_restoration_model.pth"
)
OCR_CKPT = os.environ.get("OCR_CKPT", "/app/checkpoints/best_ocr_model.pth")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "/app/checkpoints/vocab.json")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading models on {DEVICE}...")
pipeline = UrduOCRPipeline(
    restoration_ckpt=RESTORATION_CKPT,
    ocr_ckpt=OCR_CKPT,
    vocab_path=VOCAB_PATH,
    device=DEVICE,
)
print("Models loaded successfully.")


def _decode_base64_image(base64_img: str) -> np.ndarray:
    img_data = base64.b64decode(base64_img)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img


def _parse_images(payload: Dict[str, Any]) -> List[str]:
    images = payload.get("images")
    if isinstance(images, list) and images:
        return images

    single_image = payload.get("image")
    if isinstance(single_image, str) and single_image:
        return [single_image]

    raise ValueError("Input must include 'images' (list) or 'image' (string).")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    payload = job.get("input", {})
    images = _parse_images(payload)

    texts: List[str] = []
    for base64_img in images:
        try:
            img = _decode_base64_image(base64_img)
            _, text = pipeline.predict(img)
            texts.append(text)
        except Exception as exc:
            texts.append(f"[Error: {str(exc)}]")

    return {
        "texts": texts,
        "device": DEVICE,
        "models_loaded": True,
    }


runpod.serverless.start({"handler": handler})
