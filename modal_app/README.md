Modal App (H100 generator + L4 encoder)

Overview

- generate_frames: runs on H100 to produce a frame sequence + manifest.json and uploads to S3.
- encode_video: runs on an NVENC-capable GPU (L4/T4/A10G) to encode frames into MP4/HEVC/AV1 and uploads to S3.
- /health web endpoint for simple connectivity checks.
- /healthz returns recent job artifacts (absolute URLs) + ws endpoint for end-to-end testing.

This is a scaffold to integrate with your backend: the backend submits a job, triggers generate_frames, then triggers encode_video on completion. The desktop app already listens for job events and downloads/imports the final artifact.

Prereqs

- Python 3.10+
- Modal CLI installed and logged in (https://modal.com/docs/guide)
- AWS credentials available to the functions (Modal Secret named `aws-credentials`) or use your cloud IAM role.

Secrets / Env

- Modal Secret: `aws-credentials` (optional if using IAM role) with AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION.
- Env vars (set via Modal environment or your backend when invoking):
  - S3_BUCKET (target bucket)
  - BACKEND_URL (optional) and BACKEND_TOKEN for event callbacks.

Directory

- modal_app/app.py                  # Modal App definition and functions
- modal_app/common/s3_utils.py      # S3 helpers (upload/download/list)
- modal_app/common/notify.py        # Backend event callback helper
- modal_app/common/manifest.py      # Manifest read/write helpers
- modal_app/common/prompt_runner.py # Placeholder for ComfyUI runner
- modal_app/images/h100.Dockerfile  # Base image for H100 generator
- modal_app/images/l4.Dockerfile    # Base image for L4 encoder (FFmpeg + NVENC)

Deploy

1) Create or set the AWS secret (if not using IAM):
   modal secret create aws-credentials

2) Deploy (from repository root):
   modal deploy modal_app/app.py

Invoke (examples)

- Health:
  modal run modal_app.app::health
  curl https://<your-endpoint>/healthz

- Generate frames:
  modal run modal_app.app::generate_frames --job-id test123 --width 1280 --height 720 --fps 30 --count 60 --bucket your-bucket

- Encode video:
  modal run modal_app.app::encode_video --manifest-url s3://your-bucket/jobs/test123/manifest.json --bucket your-bucket --codec h264_nvenc

Notes

- h100.Dockerfile is a placeholder; add your ComfyUI install/model bootstrap here.
- l4.Dockerfile should include an FFmpeg build with NVENC enabled. The simple apt ffmpeg often lacks NVENC; prefer a static build or compile with --enable-nonfree --enable-nvenc.
- The functions use a temp working dir under /tmp for high I/O. Ensure sufficient ephemeral disk capacity on the worker.
