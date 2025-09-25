from __future__ import annotations

"""
Placeholder for a ComfyUI prompt runner.

In a production setup you would:
 - Start ComfyUI (or call its API) with the provided prompt JSON
 - Stream progress
 - Save frames to a local work dir

This scaffold instead generates synthetic frames if no real runner is wired in.
"""

import pathlib
from typing import Optional
from PIL import Image, ImageDraw, ImageFont


def generate_dummy_frames(out_dir: str, width: int, height: int, count: int, fps: int) -> None:
    path = pathlib.Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        pass
    for i in range(count):
        img = Image.new("RGB", (width, height), (30, 30, 30))
        d = ImageDraw.Draw(img)
        txt = f"Frame {i+1}/{count} @ {fps}fps"
        d.text((20, 20), txt, fill=(240, 240, 240), font=font)
        img.save(path / f"frame_{i:06d}.png")

