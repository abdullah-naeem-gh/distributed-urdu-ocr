"""
Non-Hadoop DL inference helpers.

Provides image encoding, RunPod submit/poll, and a per-image pipeline
(segment -> encode -> infer -> format result) that can be called directly
without any HDFS or Hadoop Streaming dependencies.
"""

import base64
import os
import time
from typing import List, Optional

import cv2
import numpy as np
import requests

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import preprocessing
from deployment.hadoop.line_segmenter import segment_lines


# ---------------------------------------------------------------------------
# Config (read from environment at import time; overridable in tests)
# ---------------------------------------------------------------------------

def _get_runpod_cfg():
    return {
        "api_key": os.environ.get("RUNPOD_API_KEY"),
        "endpoint_id": os.environ.get("RUNPOD_ENDPOINT_ID", "z3zabzqi52jyoh"),
        "timeout_s": int(os.environ.get("DL_RUNPOD_TIMEOUT_SECONDS", "120")),
        "poll_interval_s": float(os.environ.get("DL_RUNPOD_POLL_INTERVAL_SECONDS", "2")),
        "max_retries": int(os.environ.get("DL_RUNPOD_MAX_RETRIES", "3")),
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def encode_image(img: np.ndarray) -> str:
    """Standardize a grayscale line image and return a base64 PNG string."""
    standardized = preprocessing.standardize_and_pad(img, target_height=128, target_width=2048)
    _, buf = cv2.imencode('.png', standardized)
    return base64.b64encode(buf).decode('utf-8')


# ---------------------------------------------------------------------------
# RunPod client
# ---------------------------------------------------------------------------

def _runpod_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def call_runpod(encoded_images: List[str], cfg: Optional[dict] = None) -> List[str]:
    """
    Submit a batch of base64 PNG images to RunPod and poll until done.
    Returns a list of recognized text strings (one per image).
    Falls back to error strings on failure — never raises.
    """
    if cfg is None:
        cfg = _get_runpod_cfg()

    n = len(encoded_images)
    if not cfg["api_key"]:
        return [f"[Error: RUNPOD_API_KEY not set]"] * n

    ocr_mode = os.environ.get("OCR_MODE", "real").lower()
    if ocr_mode == "mock":
        time.sleep(0.5 + n * 0.1)
        return [f"[Mock OCR text for image {i + 1}]" for i in range(n)]

    headers = _runpod_headers(cfg["api_key"])
    run_url = f"https://api.runpod.ai/v2/{cfg['endpoint_id']}/run"

    last_err = "unknown error"
    for attempt in range(cfg["max_retries"]):
        try:
            resp = requests.post(
                run_url,
                json={"input": {"images": encoded_images}},
                headers=headers,
                timeout=30,
            )
            if resp.status_code != 200:
                last_err = f"API {resp.status_code}: {resp.text[:200]}"
                time.sleep(2 ** attempt)
                continue

            job_id = resp.json().get("id")
            status_url = f"https://api.runpod.ai/v2/{cfg['endpoint_id']}/status/{job_id}"

            deadline = time.time() + cfg["timeout_s"]
            while time.time() < deadline:
                sr = requests.get(status_url, headers=headers, timeout=10)
                if sr.status_code != 200:
                    time.sleep(cfg["poll_interval_s"])
                    continue

                data = sr.json()
                status = data.get("status")

                if status == "COMPLETED":
                    output = data.get("output", {})
                    texts = output.get("texts")
                    if isinstance(texts, list) and len(texts) == n:
                        return texts
                    return [f"[Error: unexpected output shape]"] * n

                if status == "FAILED":
                    last_err = f"RunPod job failed: {data.get('error', 'unknown')}"
                    break

                time.sleep(cfg["poll_interval_s"])
            else:
                last_err = "RunPod polling timeout"

        except Exception as exc:
            last_err = str(exc)
            time.sleep(2 ** attempt)

    return [f"[Error: {last_err}]"] * n


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def process_image_file(image_path: str) -> dict:
    """
    Full pipeline for a single image file on disk:
      1. Segment into lines
      2. Encode lines
      3. Call RunPod
      4. Return structured result dict

    Never raises — errors are captured in the result.
    """
    start = time.time()
    filename = os.path.basename(image_path)

    try:
        line_images = segment_lines(image_path)
        encoded = [encode_image(img) for img in line_images]
        recognized = call_runpod(encoded)
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "filename": filename,
            "lines": recognized,
            "processing_time_ms": elapsed_ms,
            "num_lines_detected": len(line_images),
        }
    except Exception as exc:
        return {
            "filename": filename,
            "lines": [],
            "error": str(exc),
            "processing_time_ms": int((time.time() - start) * 1000),
            "num_lines_detected": 0,
        }
