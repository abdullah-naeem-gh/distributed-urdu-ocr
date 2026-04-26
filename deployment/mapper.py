#!/usr/bin/env python3
import sys
import os
import json
import time
import subprocess
import socket

# Ensure CWD is in path
sys.path.append(os.getcwd())

# Add /app to sys.path for models/ and preprocessing.py when running in Docker
sys.path.append("/app")

# line_segmenter.py is shipped via -files, so it's in the CWD
import line_segmenter

OCR_MODE = os.environ.get("OCR_MODE", "mock").lower()
NODE_NAME = socket.gethostname()

pipeline = None
if OCR_MODE == "real":
    from models.pipeline import UrduOCRPipeline
    import torch
    
    # Initialize the pipeline once per mapper task
    # Paths are relative to /app inside the container
    restoration_ckpt = "/app/checkpoints/restoration.pth"
    ocr_ckpt = "/app/checkpoints/ocr.pth"
    vocab_path = "/app/checkpoints/vocab.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pipeline = UrduOCRPipeline(restoration_ckpt, ocr_ckpt, vocab_path, device=device)
    except Exception as e:
        sys.stderr.write(f"Warning: Failed to load real model: {e}\n")
        pipeline = None

def process_image(hdfs_path):
    start_time = time.time()
    filename = os.path.basename(hdfs_path)
    local_path = f"/tmp/{filename}"
    
    try:
        # Download image from HDFS
        subprocess.run(["/opt/hadoop-3.2.1/bin/hdfs", "dfs", "-get", "-f", hdfs_path, local_path], check=True, stderr=subprocess.PIPE)
        
        # Segment into lines
        line_images = line_segmenter.segment_lines(local_path)
        
        recognized_lines = []
        
        if OCR_MODE == "mock" or pipeline is None:
            # Simulate processing time
            time.sleep(1.0 + (len(line_images) * 0.2))
            for i in range(len(line_images)):
                recognized_lines.append(f"[Mock Text for line {i+1} of {filename} on {NODE_NAME}]")
        else:
            # Real inference
            for line_img in line_images:
                _, text = pipeline.predict(line_img)
                recognized_lines.append(text)
                
        # Clean up local file
        os.remove(local_path)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            "filename": filename,
            "lines": recognized_lines,
            "node": NODE_NAME,
            "processing_time_ms": processing_time_ms,
            "num_lines_detected": len(line_images)
        }
        
    except Exception as e:
        # Error handling as defined in the plan: emit error json, don't crash
        return {
            "filename": filename,
            "lines": [],
            "error": str(e),
            "node": NODE_NAME,
            "processing_time_ms": int((time.time() - start_time) * 1000),
            "num_lines_detected": 0
        }

if __name__ == "__main__":
    # Hadoop Streaming feeds data via stdin
    for line in sys.stdin:
        hdfs_path = line.strip()
        if not hdfs_path:
            continue
            
        result = process_image(hdfs_path)
        # Emit as a single JSON string
        print(json.dumps(result, ensure_ascii=False))
