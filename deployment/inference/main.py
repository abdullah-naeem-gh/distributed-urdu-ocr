import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch

from models.pipeline import UrduOCRPipeline

app = FastAPI(title="Urdu OCR Inference API")

# Load configuration from environment variables
RESTORATION_CKPT = os.environ.get("RESTORATION_CKPT", "/app/checkpoints/best_restoration_model.pth")
OCR_CKPT = os.environ.get("OCR_CKPT", "/app/checkpoints/best_ocr_model.pth")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "/app/checkpoints/vocab.json")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Initialize pipeline as a global variable
pipeline = None

@app.on_event("startup")
async def load_models():
    global pipeline
    print(f"Loading models on {DEVICE}...")
    try:
        pipeline = UrduOCRPipeline(
            restoration_ckpt=RESTORATION_CKPT,
            ocr_ckpt=OCR_CKPT,
            vocab_path=VOCAB_PATH,
            device=DEVICE
        )
        print("Models loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load inference pipeline: {e}") from e

class InferenceRequest(BaseModel):
    images: List[str]  # List of base64 encoded images

class InferenceResponse(BaseModel):
    texts: List[str]

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = []
    for base64_img in request.images:
        try:
            # Decode base64 to numpy array
            img_data = base64.b64decode(base64_img)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append("[Error: Invalid image data]")
                continue
            
            # Run inference
            _, text = pipeline.predict(img)
            results.append(text)
        except Exception as e:
            results.append(f"[Error: {str(e)}]")
            
    return InferenceResponse(texts=results)

@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "models_loaded": pipeline is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
