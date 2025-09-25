from __future__ import annotations

import os
from typing import Any, Dict
import requests


def post_event(event: Dict[str, Any]) -> None:
    url = os.getenv("BACKEND_URL")
    token = os.getenv("BACKEND_TOKEN")
    if not url:
        return
    try:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        requests.post(url.rstrip("/") + "/events", json=event, timeout=10, headers=headers)
    except Exception:
        # Best-effort; do not crash the worker if callback fails
        pass

