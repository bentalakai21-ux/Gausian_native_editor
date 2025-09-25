# NVENC-capable encoder image (L4 / T4 / A10G)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# NOTE: stock Ubuntu ffmpeg may not include NVENC. For a real deployment,
# install a build with --enable-nonfree --enable-nvenc. This keeps the
# scaffold simple and focuses on structure.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
# Use repo-root context path; copy the Modal app requirements
COPY modal_app/requirements.txt /root/requirements.txt
RUN pip3 install --no-cache-dir -r /root/requirements.txt
