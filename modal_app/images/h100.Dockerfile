# Placeholder H100 generator image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ensure python/pip shims exist for tooling that probes `python --version`
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /root
# Use repo-root context path; copy the Modal app requirements
COPY modal_app/requirements.txt /root/requirements.txt
RUN pip3 install --no-cache-dir -r /root/requirements.txt

# TODO: install ComfyUI and models here if you want to run real prompts
# RUN git clone https://github.com/comfyanonymous/ComfyUI /root/ComfyUI
# ... add model bootstrap
