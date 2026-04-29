# Non-Hadoop DL Endpoints Plan (RunPod-backed)

## Current System Status

### What exists today
- `deployment/hadoop/app.py` exposes Hadoop-backed APIs:
  - `POST /api/process-batch`
  - `GET /api/jobs/{job_id}/status`
  - `GET /api/jobs/{job_id}/results`
- Flow today:
  1. Upload ZIP
  2. Extract images
  3. Push to HDFS + manifest
  4. Launch Hadoop streaming job (`mapper.py` + `reducer.py`)
  5. Mapper segments lines (`line_segmenter.py`) and calls RunPod inference
  6. Results are read back from HDFS

### What is failing right now
- Hadoop/YARN orchestration is unstable in the local setup (RM/NM/container churn).
- Jobs intermittently fail or stall despite RunPod endpoint itself being usable.
- Because of this, we need an alternate path that bypasses Hadoop entirely while preserving API behavior.

### Constraint from request
- Keep existing Hadoop endpoints unchanged.
- Add a second set of endpoints with a distinct path prefix (requested: `/DL/...`).
- New endpoints must provide the same 3-endpoint async job UX.
- FastAPI should run directly with local Python, using `.env` for config.

---

## Goal

Implement **parallel, non-Hadoop batch processing endpoints** in the same FastAPI service that:
- Accept ZIP batches
- Segment each document into lines
- Call RunPod on line images
- Return per-image recognized text lines
- Expose the same async job lifecycle (`submit -> status -> results`)

---

## Proposed New API Surface

1. `POST /api/DL/process-batch`
   - Input: multipart ZIP (same as existing endpoint)
   - Output: `{ "job_id": "...", "message": "Batch processing started." }`

2. `GET /api/DL/jobs/{job_id}/status`
   - Output shape mirrors current status payload (state/progress/error metadata)

3. `GET /api/DL/jobs/{job_id}/results`
   - Output shape mirrors current results payload:
     - `{ "results": [ { filename, lines, processing_time_ms, num_lines_detected, ... }, ... ] }`

---

## Design (No Hadoop)

### High-level execution model
- Keep in-memory job store (separate namespace for DL jobs to avoid collision with Hadoop job IDs).
- On submit:
  1. Save and extract ZIP to `/tmp/<job_id>/images`
  2. Build flat list of image files recursively
  3. Spawn background worker for this DL job
- Background worker:
  1. Process images concurrently via thread pool (`ThreadPoolExecutor`)
  2. For each image:
     - read image
     - segment lines (`line_segmenter.segment_lines`)
     - normalize each line (`preprocessing.standardize_and_pad`)
     - encode lines to base64 PNG
     - call RunPod (`/run` + poll `/status/{id}`), preferably batched per document
     - return structured per-image result
  3. Aggregate all item results and finalize job state

### Parallelism strategy
- Parallelism level configurable from env (for example `DL_MAX_WORKERS`).
- Concurrency is at **document level**.
- Line-level inference remains grouped per document request to RunPod (already supported by `images` list payload).
- Add simple rate-limit protection (bounded worker count + retries/backoff for transient RunPod failures).

### Error handling expectations
- Per-image failures should be captured in result entries, not crash whole batch.
- Whole-job failure only for fatal setup/runtime errors (invalid ZIP, extraction failure, etc).
- Status endpoint always includes explicit `error` when state is failed.

### Job state model (DL endpoints)
- `UPLOADING` -> `PROCESSING` -> `SUCCEEDED` or `FAILED`
- Track:
  - total images
  - completed images
  - progress %
  - start/end timestamps
  - error (if any)

---

## Implementation Plan (Phased)

## Phase 1: Isolate reusable non-Hadoop processing logic
- Extract/refactor mapper-like helpers into importable Python functions (no stdin/HDFS assumptions):
  - image encoding
  - RunPod submit/poll client
  - per-image pipeline (`segment -> infer -> format result`)
- Ensure these helpers are used by new DL endpoints (and optionally mapper later to avoid duplication).

## Phase 2: Add `/api/DL/...` endpoints
- Add 3 new routes in `deployment/hadoop/app.py` (or a new module imported by it):
  - `/api/DL/process-batch`
  - `/api/DL/jobs/{job_id}/status`
  - `/api/DL/jobs/{job_id}/results`
- Add separate in-memory store for DL jobs.
- Add background execution with thread pool.

## Phase 3: Local runtime and env loading
- Ensure local execution works via Python command (no docker-compose required):
  - load `.env` values for RunPod and runtime knobs
  - add/confirm dependencies in Python environment
- Keep docker/Hadoop path untouched.

## Phase 4: Response parity and compatibility
- Verify payloads for new endpoints are intentionally aligned with existing 3-endpoint behavior.
- Preserve existing Hadoop endpoints exactly as-is.
- Document endpoint differences only by path prefix (`/api/...` vs `/api/DL/...`).

---

## Configuration for New DL Path

Planned env variables:
- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`
- `OCR_MODE` (optional; keep mock/real behavior if desired)
- `DL_MAX_WORKERS` (new)
- `DL_RUNPOD_TIMEOUT_SECONDS` (new)
- `DL_RUNPOD_POLL_INTERVAL_SECONDS` (new)
- `DL_RUNPOD_MAX_RETRIES` (new)

---

## Local Start (Planned)

Example local run flow (non-docker):
1. Activate Python env
2. Export env from `.env` (or auto-load in code)
3. Start FastAPI:
   - `uvicorn deployment.hadoop.app:app --host 0.0.0.0 --port 8000`

---

## Risks and Mitigations

- RunPod API rate limiting with high local concurrency  
  -> Cap worker count + retries/backoff.

- Large ZIP payloads causing high memory usage  
  -> Stream/extract to disk, process file-by-file, cleanup `/tmp/<job_id>`.

- In-memory job store loss on process restart  
  -> Acceptable for temporary bypass path; can persist later if needed.

---

## Out of Scope (for this bypass)

- Replacing/removing Hadoop endpoints
- Distributed scheduling across multiple machines
- Persistent job database

---

## Expected Outcome

You will have a **second, reliable API path** for the same batch OCR workflow that does not depend on Hadoop:
- keep current endpoints for later
- use `/api/DL/...` endpoints immediately during Hadoop instability
- still leverage RunPod for model inference
