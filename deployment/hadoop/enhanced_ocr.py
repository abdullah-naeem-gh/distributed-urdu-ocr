"""
Enhanced OCR backend for multi-line document processing.
Submits the full document image to a high-capacity vision model capable of
reading multi-line Urdu Nastaleeq text in a single pass.
"""

import base64
import hashlib
import os
import time
from typing import List, Optional

import cv2
import requests


def _get_cfg() -> dict:
    return {
        "api_key": os.environ.get("ENHANCED_OCR_API_KEY") or os.environ.get("GEMINI_API_KEY"),
        "model": os.environ.get("ENHANCED_OCR_MODEL", "gemini-2.5-flash"),
        "timeout_s": int(os.environ.get("ENHANCED_OCR_TIMEOUT_SECONDS", "60")),
        "max_retries": int(os.environ.get("ENHANCED_OCR_MAX_RETRIES", "3")),
    }


def _encode_image_file(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


_PROMPT = (
    "You are an Urdu OCR engine. The image contains Urdu text written in Nastaleeq script. "
    "Extract ALL the Urdu text exactly as it appears, line by line. "
    "Return ONLY the extracted Urdu text lines, one per line, with no explanations, "
    "no transliteration, no translation, no commentary. "
    "Preserve the right-to-left reading order. "
    "If the image has multiple lines, output each line on its own line."
)


def _call_api(image_b64: str, cfg: dict) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg['model']}:generateContent"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
    }
    headers = {"x-goog-api-key": cfg["api_key"]}
    resp = requests.post(url, json=payload, headers=headers, timeout=cfg["timeout_s"])
    if not resp.ok:
        # Surface the API's error body so we can see WHY it's failing
        body_snippet = resp.text[:500] if resp.text else "<empty>"
        print(f"[enhanced_ocr] {resp.status_code} response body: {body_snippet}", flush=True)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates returned from OCR engine")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def process_multiline_image(image_path: str, cfg: Optional[dict] = None) -> List[str]:
    """
    Run enhanced OCR on a full multi-line document image.
    Returns a list of recognized text lines, or raises on all retries exhausted.
    """
    if cfg is None:
        cfg = _get_cfg()

    if not cfg["api_key"]:
        raise RuntimeError("Enhanced OCR API key not configured")

    ocr_mode = os.environ.get("OCR_MODE", "real").lower()
    if ocr_mode == "mock":
        time.sleep(0.8)
        return ["[Mock enhanced OCR line 1]", "[Mock enhanced OCR line 2]"]

    last_exc: Exception = RuntimeError("unknown error")
    for attempt in range(cfg["max_retries"]):
        try:
            with open(image_path, "rb") as _f:
                _raw_bytes = _f.read()
            _file_sha = hashlib.sha256(_raw_bytes).hexdigest()[:16]
            image_b64 = _encode_image_file(image_path)
            _b64_sha = hashlib.sha256(image_b64.encode()).hexdigest()[:16]
            print(
                f"[enhanced_ocr] sending {os.path.basename(image_path)} "
                f"(file_bytes={len(_raw_bytes)}, file_sha256_16={_file_sha}, "
                f"b64_bytes={len(image_b64)}, b64_sha256_16={_b64_sha}, "
                f"model={cfg['model']})",
                flush=True,
            )
            raw_text = _call_api(image_b64, cfg)
            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            return lines if lines else ["[No text detected]"]
        except requests.exceptions.HTTPError as exc:
            # On 429 (rate limit) or 4xx, do not retry — retrying just burns more quota
            status = exc.response.status_code if exc.response is not None else 0
            if 400 <= status < 500:
                raise
            last_exc = exc
            if attempt < cfg["max_retries"] - 1:
                time.sleep(2 ** attempt)
        except Exception as exc:
            last_exc = exc
            if attempt < cfg["max_retries"] - 1:
                time.sleep(2 ** attempt)

    raise last_exc
