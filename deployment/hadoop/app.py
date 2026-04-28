import os
import uuid
import zipfile
import subprocess
import threading
import time
import requests
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

import hdfs_client

app = FastAPI(title="Distributed Urdu OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for job tracking
# Structure: { job_id: {"state": "...", "app_id": "...", "nodes": [], "progress": 0} }
JOBS_STORE: Dict[str, Dict[str, Any]] = {}

YARN_RM_URL = "http://resourcemanager:8088/ws/v1/cluster/apps"
HADOOP_STREAMING_JAR = "/opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar"

def run_mapreduce_job(job_id: str, input_manifest_hdfs: str, output_hdfs: str):
    """Run the MapReduce job in a background thread."""
    JOBS_STORE[job_id]["state"] = "SUBMITTED"
    
    cmd = [
        "hadoop", "jar", HADOOP_STREAMING_JAR,
        "-D", "mapreduce.task.timeout=0",
        "-files", "/app/deployment/hadoop/mapper.py#mapper.py,/app/deployment/hadoop/reducer.py#reducer.py,/app/deployment/hadoop/line_segmenter.py#line_segmenter.py",
        "-cmdenv", "PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "-cmdenv", "PYTHONPATH=/app",
        "-cmdenv", f"OCR_MODE={os.environ.get('OCR_MODE', 'real')}",
        "-cmdenv", f"RUNPOD_ENDPOINT_ID={os.environ.get('RUNPOD_ENDPOINT_ID', 'z3zabzqi52jyoh')}",
        "-cmdenv", f"RUNPOD_API_KEY={os.environ.get('RUNPOD_API_KEY')}",
        "-mapper", "/opt/conda/bin/python3 mapper.py",
        "-reducer", "/opt/conda/bin/python3 reducer.py",
        "-input", input_manifest_hdfs,
        "-output", output_hdfs
    ]
    
    # Run the command and capture output to parse YARN application ID
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Parse stderr to find the YARN application ID
        # e.g., "Submitted application application_162000000_0001"
        for line in process.stderr:
            if "Submitted application application_" in line:
                app_id = line.strip().split(" ")[-1]
                JOBS_STORE[job_id]["app_id"] = app_id
                JOBS_STORE[job_id]["state"] = "RUNNING"
                break
                
        # Wait for job to finish
        process.wait()
        
        if process.returncode == 0:
            JOBS_STORE[job_id]["state"] = "SUCCEEDED"
            JOBS_STORE[job_id]["progress"] = 100
        else:
            JOBS_STORE[job_id]["state"] = "FAILED"
            
    except Exception as e:
        JOBS_STORE[job_id]["state"] = "FAILED"
        print(f"Job {job_id} failed: {str(e)}")

@app.post("/api/process-batch")
async def process_batch(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")
        
    job_id = str(uuid.uuid4())[:8]
    JOBS_STORE[job_id] = {
        "state": "UPLOADING_TO_HDFS",
        "app_id": None,
        "nodes": [],
        "progress": 0
    }
    
    # Create local temp dir
    local_tmp_dir = f"/tmp/{job_id}"
    os.makedirs(local_tmp_dir, exist_ok=True)
    
    local_zip_path = os.path.join(local_tmp_dir, "upload.zip")
    with open(local_zip_path, "wb") as f:
        f.write(await file.read())
        
    # Extract zip
    extract_dir = os.path.join(local_tmp_dir, "images")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    # Get image files recursively
    image_files = []
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Store path relative to extract_dir for HDFS organization
                rel_path = os.path.relpath(os.path.join(root, f), extract_dir)
                image_files.append(rel_path)

    if not image_files:
        raise HTTPException(status_code=400, detail="No images found in zip (checked recursively).")
        
    # Create HDFS directories
    hdfs_input_dir = f"/input/{job_id}"
    hdfs_client.mkdir(hdfs_input_dir)
    
    # Upload images to HDFS and create manifest
    manifest_lines = []
    for rel_img_path in image_files:
        local_img_path = os.path.join(extract_dir, rel_img_path)
        # Flatten the structure in HDFS or keep it? 
        # For simplicity in mapper, we'll flatten it by replacing slashes
        hdfs_filename = rel_img_path.replace(os.sep, "_")
        hdfs_img_path = f"{hdfs_input_dir}/{hdfs_filename}"
        hdfs_client.upload_file(local_img_path, hdfs_img_path)
        manifest_lines.append(hdfs_img_path)
        
    # Upload manifest
    manifest_local = os.path.join(local_tmp_dir, "manifest.txt")
    with open(manifest_local, "w") as f:
        f.write("\n".join(manifest_lines) + "\n")
        
    manifest_hdfs = f"{hdfs_input_dir}/manifest.txt"
    hdfs_client.upload_file(manifest_local, manifest_hdfs)
    
    # Setup HDFS output dir
    hdfs_output_dir = f"/output/{job_id}"
    hdfs_client.delete(hdfs_output_dir) # Ensure it's clear
    
    # Start MapReduce job
    background_tasks.add_task(run_mapreduce_job, job_id, manifest_hdfs, hdfs_output_dir)
    
    return {"job_id": job_id, "message": "Batch processing started."}

@app.get("/api/jobs/{job_id}/status")
def get_job_status(job_id: str):
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job_info = JOBS_STORE[job_id]
    
    # If we have an app_id and it's running, query YARN
    if job_info["app_id"] and job_info["state"] == "RUNNING":
        try:
            resp = requests.get(f"{YARN_RM_URL}/{job_info['app_id']}", timeout=2)
            if resp.status_code == 200:
                yarn_data = resp.json().get("app", {})
                job_info["progress"] = yarn_data.get("progress", 0)
                
                # Fetch node info (mocked for simplicity, in real life requires RM API deep dive)
                # But we can get states
                yarn_state = yarn_data.get("state")
                if yarn_state in ["FINISHED", "FAILED", "KILLED"]:
                    job_info["state"] = yarn_state
        except Exception as e:
            print(f"YARN API error: {e}")
            
    return job_info

@app.get("/api/jobs/{job_id}/results")
def get_job_results(job_id: str):
    if job_id not in JOBS_STORE:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if JOBS_STORE[job_id]["state"] not in ["SUCCEEDED", "FINISHED"]:
        raise HTTPException(status_code=400, detail="Job has not finished successfully yet.")
        
    hdfs_output_dir = f"/output/{job_id}"
    files = hdfs_client.list_dir(hdfs_output_dir)
    
    # Find the part file (usually part-00000)
    part_files = [f for f in files if f.startswith("part-")]
    if not part_files:
        return {"results": []}
        
    all_results = []
    for pf in part_files:
        try:
            content = hdfs_client.read_file(f"{hdfs_output_dir}/{pf}")
            for line in content.strip().split("\n"):
                if line:
                    # Parse JSON from reducer
                    all_results.append(json.loads(line))
        except Exception as e:
            print(f"Error reading {pf}: {e}")
            
    return {"results": all_results}

@app.get("/api/jobs")
def list_jobs():
    return JOBS_STORE

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    if job_id in JOBS_STORE:
        # Kill if running
        app_id = JOBS_STORE[job_id]["app_id"]
        if app_id and JOBS_STORE[job_id]["state"] == "RUNNING":
            subprocess.run(["yarn", "application", "-kill", app_id])
            
        # Cleanup HDFS
        hdfs_client.delete(f"/input/{job_id}")
        hdfs_client.delete(f"/output/{job_id}")
        
        del JOBS_STORE[job_id]
        return {"message": "Job deleted."}
    raise HTTPException(status_code=404, detail="Job not found")

@app.get("/api/cluster/info")
def get_cluster_info():
    try:
        resp = requests.get("http://resourcemanager:8088/ws/v1/cluster/metrics", timeout=2)
        metrics = resp.json().get("clusterMetrics", {})
        return {
            "activeNodes": metrics.get("activeNodes", 0),
            "totalMemoryMB": metrics.get("availableMB", 0) + metrics.get("allocatedMB", 0),
            "appsRunning": metrics.get("appsRunning", 0)
        }
    except Exception:
        return {"status": "unavailable"}
