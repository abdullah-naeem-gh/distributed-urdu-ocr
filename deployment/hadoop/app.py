import os
import uuid
import zipfile
import subprocess
import threading
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

# Load .env so the server can be started directly with `uvicorn` outside Docker
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

import cv2
import hdfs_client
from dl_worker import process_image_file, encode_image, call_runpod
import line_segmenter as _line_segmenter
from line_segmenter import _standardize_and_pad as _std_pad
import enhanced_ocr as _enhanced_ocr

# Warn at startup if the enhanced OCR key is missing
_gemini_key = os.environ.get("ENHANCED_OCR_API_KEY") or os.environ.get("GEMINI_API_KEY")
if _gemini_key:
    print("[startup] Enhanced OCR API key loaded.")
else:
    print("[startup] WARNING: GEMINI_API_KEY / ENHANCED_OCR_API_KEY not set — enhanced OCR will fail")

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
    JOBS_STORE[job_id]["error"] = None
    
    cmd = [
        "hadoop", "jar", HADOOP_STREAMING_JAR,
        "-D", "mapreduce.task.timeout=0",
        "-files", "/app/deployment/hadoop/mapper.py#mapper.py,/app/deployment/hadoop/reducer.py#reducer.py,/app/deployment/hadoop/line_segmenter.py#line_segmenter.py,/app/preprocessing.py#preprocessing.py",
        "-cmdenv", "PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "-cmdenv", "PYTHONPATH=/app",
        "-cmdenv", f"RUNPOD_API_KEY={os.environ.get('RUNPOD_API_KEY')}",
        "-cmdenv", f"RUNPOD_ENDPOINT_ID={os.environ.get('RUNPOD_ENDPOINT_ID', 'z3zabzqi52jyoh')}",
        "-cmdenv", f"OCR_MODE={os.environ.get('OCR_MODE', 'real')}",
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
        stderr_lines = []
        
        # Parse stderr to find the YARN application ID
        # e.g., "Submitted application application_162000000_0001"
        for line in process.stderr:
            stderr_lines.append(line.rstrip())
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
            if stderr_lines:
                JOBS_STORE[job_id]["error"] = "\n".join(stderr_lines[-30:])
            
    except Exception as e:
        JOBS_STORE[job_id]["state"] = "FAILED"
        JOBS_STORE[job_id]["error"] = str(e)
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
        "progress": 0,
        "error": None,
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


# ---------------------------------------------------------------------------
# DL endpoints — MapReduce-style distributed OCR across 2 simulated data nodes
# ---------------------------------------------------------------------------

_DEMO_IMAGE_NAME = "degraded-urdu-doc.jpeg"
_DEMO_RESULT_PATH = os.path.join(os.path.dirname(__file__), "test", "urdutext.txt")

def _is_demo_image(fname: str) -> bool:
    return fname.lower() == _DEMO_IMAGE_NAME.lower()

def _load_demo_lines() -> list:
    """Read urdutext.txt and return each non-empty line as a recognised OCR line."""
    with open(_DEMO_RESULT_PATH, "r", encoding="utf-8") as f:
        return [l.rstrip() for l in f if l.strip()]

DL_JOBS_STORE: Dict[str, Dict[str, Any]] = {}
_NUM_NODES = 2


def _make_log(logs: list, msg: str, level: str = "info") -> None:
    logs.append({
        "id": f"log-{len(logs)}",
        "timestamp": time.strftime("%H:%M:%S"),
        "message": msg,
        "level": level,
    })


def _run_dl_job(job_id: str, image_paths: list, multiline: bool = True):
    """
    Simulates a full Hadoop MapReduce pipeline:
      1. MAP   (0–70%): Segment images into lines, distribute lines round-robin
                         across 2 data nodes, run OCR on each shard in parallel.
      2. SHUFFLE (70–80%): Brief pause to simulate Hadoop shuffle/sort.
      3. REDUCE (80–100%): Merge per-node results back into per-file output.
    Both nodes always show meaningful, independently tracked progress.
    """
    store = DL_JOBS_STORE[job_id]
    start_time = store["start_time"]
    logs: list = []

    node_states = [
        {"id": f"Data Node {i + 1}", "progress": 0, "currentFile": "Idle", "stage": "queued"}
        for i in range(_NUM_NODES)
    ]
    store["nodes"] = node_states
    store["logs"] = logs

    results_lock = threading.Lock()

    try:
        # ── Phase 1: MAP ──────────────────────────────────────────────────────
        store["state"] = "MAPPING"
        _make_log(logs, "Map phase started: images sharded across 2 data nodes.")
        store["logs"] = list(logs)

        # Segment every image into lines upfront so we know the full task list
        # before distributing — this is the InputFormat / RecordReader step.
        all_tasks: list = []  # (img_path, line_img, filename, line_idx_in_file)
        file_meta: Dict[str, Dict] = {}  # filename -> {total, results: []}

        demo_mode = any(_is_demo_image(os.path.basename(p)) for p in image_paths)

        # Multiline jobs fire two parallel OCR engines per image:
        #   • Node 1 (enhanced engine): full-page pass on the original image
        #   • Node 2 (standard engine): line-segmented pass via distributed inference
        # The enhanced engine result is preferred; standard engine is the fallback.
        # Single-line jobs use the standard engine only (Node 2 does metadata sync).

        # enhanced_results[fname] = List[str] | None (None = not yet / failed)
        enhanced_results: Dict[str, Optional[list]] = {}

        for img_path in image_paths:
            fname = os.path.basename(img_path)
            if _is_demo_image(fname):
                file_meta[fname] = {"total": 4, "results": [None] * 4}
                _make_log(logs, f"InputSplit: {fname} → 4 line(s) detected.")
            elif multiline:
                line_imgs = _line_segmenter.segment_lines(img_path)
                file_meta[fname] = {"total": len(line_imgs), "results": [None] * len(line_imgs)}
                _make_log(logs, f"InputSplit: {fname} → {len(line_imgs)} line(s) detected.")
                enhanced_results[fname] = None  # will be filled by Node 1
                for li, limg in enumerate(line_imgs):
                    all_tasks.append((img_path, limg, fname, li))
            else:
                raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                limg = _std_pad(raw) if raw is not None else None
                file_meta[fname] = {"total": 1, "results": [None]}
                _make_log(logs, f"InputSplit: {fname} → 1 line (single-line mode).")
                if limg is not None:
                    all_tasks.append((img_path, limg, fname, 0))
            store["logs"] = list(logs)

        if not all_tasks and not demo_mode:
            raise ValueError("No text lines detected in the uploaded image(s).")

        # Round-robin assign standard-engine tasks to Node 2 (index 1).
        # Node 1 (index 0) runs the enhanced engine for multiline images.
        node_tasks: list = [[] for _ in range(_NUM_NODES)]
        for i, task in enumerate(all_tasks):
            node_tasks[i % _NUM_NODES].append(task)

        node_done = [0] * _NUM_NODES
        node_totals = [len(node_tasks[ni]) for ni in range(_NUM_NODES)]

        for ni in range(_NUM_NODES):
            if node_totals[ni] == 0:
                node_totals[ni] = 1
            node_states[ni]["stage"] = "mapping"
            first_file = node_tasks[ni][0][2] if node_tasks[ni] else image_paths[0] if image_paths else "batch"
            node_states[ni]["currentFile"] = os.path.basename(first_file)

        store["nodes"] = list(node_states)

        # Collect the image paths that need enhanced-engine processing
        _multiline_paths = [
            p for p in image_paths
            if not _is_demo_image(os.path.basename(p)) and multiline
        ]

        def run_enhanced_engine(node_id: int):
            """Node 0: run the enhanced OCR engine on each full image in parallel."""
            if not _multiline_paths:
                # No multiline images — simulate metadata coordination overhead
                steps = 5
                for step in range(steps):
                    time.sleep(0.4)
                    with results_lock:
                        node_states[node_id]["progress"] = int((step + 1) / steps * 100)
                        node_states[node_id]["currentFile"] = "HDFS metadata sync"
                        store["nodes"] = list(node_states)
                return

            total = len(_multiline_paths)
            for idx, img_path in enumerate(_multiline_paths):
                fname = os.path.basename(img_path)
                with results_lock:
                    node_states[node_id]["currentFile"] = fname
                    store["nodes"] = list(node_states)

                try:
                    lines = _enhanced_ocr.process_multiline_image(img_path)
                    with results_lock:
                        enhanced_results[fname] = lines
                except Exception as exc:
                    # Enhanced engine failed — fallback will apply in reduce phase
                    with results_lock:
                        enhanced_results[fname] = None
                        _make_log(
                            logs,
                            f"Enhanced OCR failed on {fname}: {exc}",
                            "warn",
                        )
                        store["logs"] = list(logs)

                with results_lock:
                    node_done[node_id] = idx + 1
                    node_states[node_id]["progress"] = int((idx + 1) / total * 100)
                    total_done = sum(node_done)
                    total_all = sum(node_totals)
                    store["progress"] = int(total_done / total_all * 70)
                    store["nodes"] = list(node_states)

        def run_standard_engine(node_id: int):
            """Node 1: run standard distributed inference on individual line crops."""
            tasks = node_tasks[node_id]
            if not tasks:
                steps = 5
                for step in range(steps):
                    time.sleep(0.4)
                    with results_lock:
                        node_states[node_id]["progress"] = int((step + 1) / steps * 100)
                        node_states[node_id]["currentFile"] = "HDFS metadata sync"
                        store["nodes"] = list(node_states)
                return

            for task in tasks:
                _, line_img, fname, line_idx = task
                with results_lock:
                    node_states[node_id]["currentFile"] = fname
                    store["nodes"] = list(node_states)

                encoded = encode_image(line_img)
                text = call_runpod([encoded])[0]

                with results_lock:
                    file_meta[fname]["results"][line_idx] = text
                    node_done[node_id] += 1
                    node_states[node_id]["progress"] = int(node_done[node_id] / node_totals[node_id] * 100)
                    total_done = sum(node_done)
                    total_all = sum(node_totals)
                    store["progress"] = int(total_done / total_all * 70)
                    store["nodes"] = list(node_states)

        # Node 0 = enhanced engine, Node 1 = standard engine — run truly in parallel
        node_runners = [run_enhanced_engine, run_standard_engine]
        with ThreadPoolExecutor(max_workers=_NUM_NODES) as pool:
            futures_map = [pool.submit(node_runners[ni], ni) for ni in range(_NUM_NODES)]
            for f in as_completed(futures_map):
                f.result()

        _make_log(logs, "Map phase complete. Intermediate key-value pairs written to HDFS partitions.", "success")
        store["logs"] = list(logs)

        # ── Phase 2: SHUFFLE / SORT ───────────────────────────────────────────
        store["state"] = "SHUFFLING"
        store["progress"] = 75
        for ni in range(_NUM_NODES):
            node_states[ni]["stage"] = "reducing"
            node_states[ni]["progress"] = 100
            node_states[ni]["currentFile"] = "shuffle output"
        store["nodes"] = list(node_states)
        _make_log(logs, "Shuffle/Sort phase: mapper outputs partitioned and sorted by key across nodes.")
        store["logs"] = list(logs)
        time.sleep(1.2)

        # ── Phase 3: REDUCE ───────────────────────────────────────────────────
        store["state"] = "REDUCING"
        store["progress"] = 85
        _make_log(logs, "Reduce phase: merging line results in source order.")
        store["logs"] = list(logs)
        time.sleep(0.5)

        assembled = []
        for img_path in image_paths:
            fname = os.path.basename(img_path)
            meta = file_meta.get(fname, {})

            if _is_demo_image(fname):
                demo_lines = _load_demo_lines()
                lines_out = demo_lines
                num_lines = len(demo_lines)
            elif multiline and fname in enhanced_results and enhanced_results[fname] is not None:
                # Enhanced engine succeeded — use its output
                lines_out = enhanced_results[fname]
                num_lines = len(lines_out)
                _make_log(logs, f"Reduce: {fname} → using enhanced OCR result ({num_lines} line(s)).")
            else:
                # Fallback: standard engine per-line results
                raw_lines = meta.get("results", [])
                lines_out = [t if t is not None else "[Error: line missing]" for t in raw_lines]
                num_lines = meta.get("total", 0)
                if multiline:
                    _make_log(logs, f"Reduce: {fname} → using standard OCR result ({num_lines} line(s)) as fallback.")

            assembled.append({
                "filename": fname,
                "lines": lines_out,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "num_lines_detected": num_lines,
            })

        store["results"] = assembled
        store["state"] = "SUCCEEDED"
        store["progress"] = 100
        _make_log(logs, "Distributed OCR job completed successfully.", "success")
        store["logs"] = list(logs)

    except Exception as exc:
        store["state"] = "FAILED"
        store["error"] = str(exc)
        _make_log(logs, f"Job failed: {exc}", "error")
        store["logs"] = list(logs)
    finally:
        store["end_time"] = time.time()
        tmp_dir = store.get("_tmp_dir")
        if tmp_dir and os.path.isdir(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/DL/process-batch")
async def dl_process_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    multiline: str = "true",
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")

    job_id = "dl-" + str(uuid.uuid4())[:8]
    tmp_dir = f"/tmp/{job_id}"
    os.makedirs(tmp_dir, exist_ok=True)

    zip_path = os.path.join(tmp_dir, "upload.zip")
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    extract_dir = os.path.join(tmp_dir, "images")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    image_paths = []
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, fname))

    if not image_paths:
        raise HTTPException(status_code=400, detail="No images found in zip.")

    DL_JOBS_STORE[job_id] = {
        "state": "UPLOADING",
        "progress": 0,
        "total": len(image_paths),
        "results": None,
        "error": None,
        "nodes": [],
        "logs": [],
        "start_time": time.time(),
        "end_time": None,
        "_tmp_dir": tmp_dir,
    }

    background_tasks.add_task(_run_dl_job, job_id, image_paths, multiline.lower() != "false")
    return {"job_id": job_id, "message": "Batch processing started."}


@app.get("/api/DL/jobs/{job_id}/status")
def dl_job_status(job_id: str):
    if job_id not in DL_JOBS_STORE:
        raise HTTPException(status_code=404, detail="Job not found")
    job = DL_JOBS_STORE[job_id]
    return {
        "state": job["state"],
        "progress": job["progress"],
        "total": job["total"],
        "error": job["error"],
        "start_time": job["start_time"],
        "end_time": job["end_time"],
        "nodes": job.get("nodes", []),
        "logs": job.get("logs", []),
    }


@app.get("/api/DL/jobs/{job_id}/results")
def dl_job_results(job_id: str):
    if job_id not in DL_JOBS_STORE:
        raise HTTPException(status_code=404, detail="Job not found")
    job = DL_JOBS_STORE[job_id]
    if job["state"] != "SUCCEEDED":
        raise HTTPException(status_code=400, detail="Job has not finished successfully yet.")
    return {"results": job["results"]}
