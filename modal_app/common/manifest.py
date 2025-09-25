from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Manifest:
    job_id: str
    width: int
    height: int
    fps: int
    count: int
    format: str
    frames_prefix: str
    pad: int = 6
    audio: Optional[dict] = None
    colorspace: str = "rgb"

    def to_dict(self) -> dict:
        return asdict(self)

