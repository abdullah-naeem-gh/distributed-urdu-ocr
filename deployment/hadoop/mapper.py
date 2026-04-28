#!/usr/bin/env python3
import sys
import os
import json
import time
import subprocess
import socket

import base64
import requests
import cv2
import numpy as np

# Ensure CWD is in path
sys.path.append(os.getcwd())

# line_segmenter.py is shipped via -files, so it's in the CWD
import line_segmenter

OCR_MODE = os.environ.get("OCR_MODE", "real").lower()
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "z3zabzqi52jyoh")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
NODE_NAME = socket.gethostname()

def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def call_runpod_inference(encoded_images):
    """Call RunPod Serverless API with polling for results."""
    if not RUNPOD_API_KEY:
        return [f"[Error: RUNPOD_API_KEY not set]"] * len(encoded_images)
    
    # We'll process images one by one or as a batch depending on serverless.py capability
    # The current serverless.py handles single 'image' or 'images' list
    # Use the asynchronous /run endpoint
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Start job
        response = requests.post(url, json={"input": {"images": encoded_images}}, headers=headers, timeout=30)
        if response.status_code != 200:
            return [f"[Error: API {response.status_code}]"] * len(encoded_images)
        
        job_id = response.json().get("id")
        status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
        
        # Poll for completion
        for _ in range(60): # Max 120 seconds
            status_resp = requests.get(status_url, headers=headers, timeout=10)
            if status_resp.status_code != 200:
                continue
                
            data = status_resp.json()
            status = data.get("status")
            
            if status == "COMPLETED":
                # Assuming output format from serverless.py: {'texts': [...]}
                output = data.get("output", {})
                return output.get("texts", [f"[Error: No texts in output]"] * len(encoded_images))
            elif status == "FAILED":
                return [f"[Error: Job Failed: {data.get('error')}]"] * len(encoded_images)
            
            time.sleep(2)
            
        return [f"[Error: Timeout]"] * len(encoded_images)
    except Exception as e:
        return [f"[Error: {str(e)}]"] * len(encoded_images)

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
        
        if OCR_MODE == "mock":
            # Simulate processing time
            time.sleep(1.0 + (len(line_images) * 0.2))
            for i in range(len(line_images)):
                recognized_lines.append(f"[Mock Text for line {i+1} of {filename} on {NODE_NAME}]")
        else:
            # Real inference via RunPod API
            if line_images:
                encoded_images = [encode_image(img) for img in line_images]
                recognized_lines = call_runpod_inference(encoded_images)
                
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
        # Error handling
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
