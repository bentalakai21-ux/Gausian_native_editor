H A N D O F F
==============

This doc equips a fresh engineer (or a restarted AI instance) with the minimal, exact context needed to continue the work without re‑discovering decisions. It assumes no prior conversation history.

Project Snapshot
----------------

- Repo: native desktop video editor (Rust/egui) with optional cloud generation/encoding via Modal and ComfyUI
- App path: apps/desktop (Rust); supporting crates under crates/
- Cloud scaffold: modal_app (Python) with two functions (H100 generator, L4 encoder)

Current Status
--------------

Desktop (Rust):
- Assets panel + timeline + GPU preview are functional
- Local ComfyUI auto‑import works (videos + images). The watcher monitors <repo_path>/output and moves finished files into <project>/media/comfy/YYYY‑MM‑DD
- Cloud section (always visible) can:
  - Test connection (GET {base}/health)
  - Queue job to ComfyUI /prompt (POST {base}/prompt)
  - Choose Target: “ComfyUI /prompt” (expects prompt JSON) or “Workflow (auto‑convert)” (best‑effort converter wraps/derives prompt)
  - Show server error bodies on non‑2xx
- Live job monitors:
  - Local WS (/ws) for ComfyUI, and Cloud WS (/events) for backend/remote. Threads are non‑blocking on stop and auto‑download artifacts for import
- Default UI: “Open inside editor” unchecked; payload/log ScrollAreas have explicit id_source to avoid egui ID collisions

Modal Scaffold (Python):
- generate_frames (H100) — currently generates dummy frames, uploads to S3, writes manifest.json
- encode_video (L4) — downloads frames, runs FFmpeg (NVENC if available; fallback to libx264), uploads MP4
- Simple health endpoint; best‑effort backend callback hook (POST BACKEND_URL/events if set)

Open Challenges
---------------

1) Cloud backend (control plane) not implemented in this repo:
   - Needs /jobs (submit), /jobs/{id} (status/artifacts), WS /events, orchestration H100→L4
   - Needs token/billing later (out of scope for now)

2) ComfyUI prompt runner on H100 (in modal_app) is a stub:
   - Replace generate_dummy_frames with a real ComfyUI invocation that writes frames

3) NVENC build for FFmpeg in L4 image:
   - The scaffold uses apt ffmpeg; replace with an NVENC‑enabled build for production

4) Workflow converter (desktop):
   - Best‑effort JSON conversion; does not reconstruct graph links fully. Prefer “Copy API” prompts or do server‑side conversion

Next Steps
----------

Backend (outside this repo):
- Implement /jobs, /jobs/{id}, /events; issue pre‑signed URLs; schedule Modal functions
- On H100 completion: validate manifest.json; enqueue L4 encode; publish events
- On L4 completion: publish job_completed with artifact URL; app auto‑imports

Desktop app:
- Add an Encoder dropdown (h264_nvenc / hevc_nvenc / av1_nvenc / libx264) to the Cloud section and pass it to the backend payload
- Optional: “frames only” mode that imports the image sequence and offers “Encode locally/Cloud” action
- Auto‑rebind cloud monitor to current project on project switch (today: toggle off/on)

Modal app:
- Implement real ComfyUI prompt runner in generate_frames (start ComfyUI or call /prompt, write frames)
- Replace l4.Dockerfile ffmpeg with an NVENC build; verify -encoders lists *_nvenc

Important Paths & Artifacts
---------------------------

Desktop app imports to:
- Local ComfyUI: <project-base>/media/comfy/YYYY‑MM‑DD/<file>
- Cloud/Modal: <project-base>/media/cloud/YYYY‑MM‑DD/<file>
- Project base path is stored in SQLite (crates/project). If a prior single‑file import set base to a file, the app auto‑heals to its parent directory before importing

Modal scaffold writes to S3:
- jobs/{job_id}/manifest.json
- jobs/{job_id}/frames/frame_%06d.png
- jobs/{job_id}/video/out.mp4

Recent Test Results & Logs
--------------------------

Local auto‑import:
- “Watching Comfy outputs: …/ComfyUI/output”
- “Moved into <project_id>: <path>” / “Copied into …”

Cloud queue (/prompt):
- On success: “Queued job: <prompt_id>”
- On error: “Queue failed: HTTP 400\n<server body>”

Cloud monitor (/events):
- “Cloud monitor: connected” → on job_completed the app downloads artifacts and enqueues imports

Schemas / Contracts
-------------------

ComfyUI /prompt (expected by app when Target=Prompt):
```json
{
  "prompt": { "<node-id>": { "class_type": "…", "inputs": { … } }, … },
  "client_id": "<uuid>"
}
```

Manifest (written by generator, read by encoder):
```json
{
  "job_id": "<uuid>",
  "width": 1920,
  "height": 1080,
  "fps": 30,
  "count": 150,
  "format": "png",
  "frames_prefix": "frames/frame_",
  "pad": 6,
  "audio": null,
  "colorspace": "rgb"
}
```

Exact Environment / Versions
----------------------------

Rust (workspace):
- egui 0.29, eframe 0.29 (wgpu), wgpu 0.20
- symphonia 0.5, crossbeam‑channel 0.5, walkdir 2
- ureq 2, tungstenite 0.21, url 2, urlencoding 2
- native‑decoder (local crate; gstreamer feature)

External:
- FFmpeg/ffprobe installed on PATH

Modal (Python):
- Python 3.10+
- modal, boto3, requests, Pillow (modal_app/requirements.txt)
- Base images: nvidia/cuda:12.1.1‑runtime‑ubuntu22.04

Notes to avoid duplicate envs:
- Do not upgrade egui/eframe/wgpu without checking the desktop UI code
- Keep Modal Python deps unpinned for now (scaffold), but pin when promoting to prod
- Ensure only one FFmpeg is on PATH to avoid probe flakiness

Repo Map (updated)
------------------

```
apps/desktop/               # Rust GUI app (egui)
crates/                     # timeline, project, media-io, exporters, …
modal_app/                  # Modal scaffold (Python)
  app.py                    # Modal App & functions (H100/L4)
  images/h100.Dockerfile    # Generator image (ComfyUI to be added)
  images/l4.Dockerfile      # Encoder image (NVENC FFmpeg needed)
  common/                   # s3 utils, notify, manifest, prompt_runner
README.md                   # You are here (holistic, updated)
HANDOFF.md                  # This document (handoff context)
```

Contact Points / Decisions
--------------------------

- Local auto‑import: requires ffprobe; toggle under ComfyUI header (local)
- Cloud queue: POST /prompt; Target selector controls payload wrapping/conversion
- Hardware encode on H100 is not available (no NVENC). Use L4 for encode or fallback to libx264

