import argparse
import random
import cv2
import pandas as pd
import numpy as np
import pytesseract
from tqdm import tqdm
import jiwer
from pathlib import Path

from preprocessing import degrade_image, standardize_image
from models.pipeline import UrduOCRPipeline

# Ensure pytesseract knows the language if needed (Urdu is 'urd' in tesseract)
# Users must have 'urd' language data installed for tesseract: tesseract-ocr-urd
TESSERACT_LANG = 'urd'

def calculate_metrics(references, hypotheses):
    """Calculate CER and WER using jiwer."""
    # Filter out empty references
    valid_refs = []
    valid_hyps = []
    
    for ref, hyp in zip(references, hypotheses):
        # Jiwer fails on empty strings, so we replace them with a special char or skip
        if not ref.strip():
            continue
        valid_refs.append(ref)
        valid_hyps.append(hyp if hyp.strip() else "<empty>")
        
    if not valid_refs:
        return 1.0, 1.0
        
    cer = jiwer.cer(valid_refs, valid_hyps)
    wer = jiwer.wer(valid_refs, valid_hyps)
    return cer, wer

def main(args):
    print("Loading test split...")
    df = pd.read_csv(args.test_csv)
    
    # Shuffle and limit for benchmarking if requested
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)
        
    print(f"Benchmarking on {len(df)} samples...")
    
    print("Initializing Deep Learning Pipeline...")
    pipeline = UrduOCRPipeline(
        restoration_ckpt=args.restoration_ckpt,
        ocr_ckpt=args.ocr_ckpt,
        vocab_path=args.vocab,
        device=args.device
    )
    
    ground_truths = []
    tesseract_preds = []
    dl_preds = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
        img_path = row["image_path"]
        label = str(row["label"])
        
        # 1. Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # 2. Standardize height (128)
        img = standardize_image(img, target_height=128)
            
        # 3. Apply synthetic degradation to create the un-restored degraded input
        degraded_img = degrade_image(img)
        
        # 4. Tesseract Baseline
        try:
            # Tesseract works best on PIL Image or numpy array
            tess_text = pytesseract.image_to_string(degraded_img, lang=TESSERACT_LANG).strip()
        except Exception as e:
            tess_text = ""
            
        # 5. Deep Learning Pipeline
        try:
            # We pass the degraded image directly
            _, dl_text = pipeline.predict(degraded_img)
        except Exception as e:
            dl_text = ""
            
        ground_truths.append(label)
        tesseract_preds.append(tess_text)
        dl_preds.append(dl_text)
        
    print("\nCalculating metrics...")
    
    # Calculate baseline metrics
    tess_cer, tess_wer = calculate_metrics(ground_truths, tesseract_preds)
    
    # Calculate pipeline metrics
    dl_cer, dl_wer = calculate_metrics(ground_truths, dl_preds)
    
    print("=" * 50)
    print("BENCHMARKING RESULTS")
    print("=" * 50)
    print(f"Total Evaluated Samples: {len(ground_truths)}")
    print("-" * 50)
    print("TESSERACT OCR (BASELINE)")
    print(f"  CER: {tess_cer:.4f} ({tess_cer*100:.2f}%)")
    print(f"  WER: {tess_wer:.4f} ({tess_wer*100:.2f}%)")
    print("-" * 50)
    print("DEEP LEARNING PIPELINE (OURS)")
    print(f"  CER: {dl_cer:.4f} ({dl_cer*100:.2f}%)")
    print(f"  WER: {dl_wer:.4f} ({dl_wer*100:.2f}%)")
    print("=" * 50)
    
    # Print a few examples
    print("\nExample Comparisons:")
    for i in range(min(5, len(ground_truths))):
        print(f"\nExample {i+1}:")
        print(f"  Ground Truth : {ground_truths[i]}")
        print(f"  Tesseract    : {tesseract_preds[i]}")
        print(f"  DL Pipeline  : {dl_preds[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Phase 4 DL Pipeline against Tesseract")
    parser.add_argument("--test_csv", type=str, default="splits/test.csv", help="Path to test split CSV")
    parser.add_argument("--restoration_ckpt", type=str, default="checkpoints/final_restoration_model.pth", help="Path to restoration model checkpoint")
    parser.add_argument("--ocr_ckpt", type=str, default="checkpoints/final_ocr_model.pth", help="Path to OCR model checkpoint")
    parser.add_argument("--vocab", type=str, default="checkpoints/vocab.json", help="Path to vocab file")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of samples for faster benchmarking")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)
