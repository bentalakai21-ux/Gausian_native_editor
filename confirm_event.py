#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
import time
import uuid
from typing import Optional, Tuple

import requests
import websockets


def derive_ws_from_base(base: str) -> str:
    b = base.rstrip("/")
    if b.startswith("https://"):
        return b.replace("https://", "wss://", 1) + "/events"
    if b.startswith("http://"):
        return b.replace("http://", "ws://", 1) + "/events"
    # Assume https if scheme was omitted
    return "wss://" + b + "/events"


async def ws_listener(uri: str, token: Optional[str], expected_id: str, timeout_s: int = 120) -> bool:
    headers: list[tuple[str, str]] = []
    if token:
        headers.append(("Authorization", f"Bearer {token}"))
    start = time.time()
    print(f"[WS] Connecting to {uri} ...")
    # Support websockets>=12 (additional_headers) and older (extra_headers)
    conn = None
    try:
        conn = websockets.connect(
            uri,
            additional_headers=(headers or None),
            ping_interval=20,
            ping_timeout=20,
        )
    except TypeError:
        conn = websockets.connect(
            uri,
            extra_headers=(headers or None),
            ping_interval=20,
            ping_timeout=20,
        )
    async with conn as ws:
        print("[WS] Connected. Waiting for events ...")
        while True:
            # Timeout guard
            if time.time() - start > timeout_s:
                print("[WS] Timeout waiting for events.")
                return False
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                # Keep waiting — server may only push intermittently
                continue
            except Exception as e:
                print(f"[WS] Error/closed: {e}")
                return False

            try:
                data = json.loads(msg)
            except Exception:
                print(f"[WS] Non-JSON message: {msg}")
                continue

            t = data.get("type")
            if t == "hello":
                print("[WS] hello")
                continue

            if t == "status":
                # status can include a jobs[] array; print any entry matching our id
                jobs = data.get("jobs") or []
                for it in jobs:
                    if str(it.get("job_id")) == expected_id:
                        pr = it.get("progress_percent")
                        cur = it.get("current_step")
                        tot = it.get("total_steps")
                        print(f"[WS] status: job_id={expected_id} progress={pr} cur={cur} tot={tot}")
                continue

            if t == "progress":
                jid = str(data.get("job_id"))
                if jid == expected_id:
                    pr = data.get("progress_percent")
                    cur = data.get("current_step")
                    tot = data.get("total_steps")
                    print(f"[WS] progress: job_id={jid} progress={pr} cur={cur} tot={tot}")
                continue

            if t == "job_completed":
                jid = str(data.get("job_id"))
                if jid == expected_id:
                    print(f"[WS] job_completed: job_id={jid}")
                    return True

            # Other types (executed/executing/job_failed/etc.)
            # Only echo if they reference our job
            jid = data.get("job_id")
            if jid and str(jid) == expected_id:
                print(f"[WS] {t}: {json.dumps(data)}")


def queue_prompt(base: str, prompt_json: dict, token: Optional[str]) -> Optional[str]:
    url = base.rstrip("/") + "/prompt"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    print(f"[HTTP] POST {url}")
    r = requests.post(url, json=prompt_json, headers=headers, timeout=60)
    print(f"[HTTP] -> {r.status_code}")
    try:
        body = r.json()
    except Exception:
        print("[HTTP] Non-JSON response:", r.text[:500])
        return None
    if r.status_code != 200:
        print("[HTTP] Error:", body)
        return None
    # Accept prompt_id / id / number (server-dependent)
    pid = body.get("prompt_id") or body.get("id") or body.get("number")
    if not pid:
        print("[HTTP] No prompt_id in response:", body)
        return None
    print(f"[HTTP] queued: prompt_id={pid}")
    return str(pid)


def http_progress_poll(base: str, token: Optional[str], job_ids: list[str], timeout_s: int = 300) -> bool:
    """Synchronous HTTP poller for job status/progress.
    Uses requests to avoid adding aiohttp dependency.
    """
    start = time.time()
    last = {"status": None, "cur": None, "tot": None, "pr": None}
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    while True:
        if time.time() - start > timeout_s:
            print("[HTTP] Timeout polling job status")
            return False
        got_any = False
        for jid in job_ids:
            url = base.rstrip("/") + f"/jobs/{jid}"
            try:
                r = requests.get(url, headers=headers, timeout=30)
                if r.status_code // 100 != 2:
                    continue
                body = r.json()
            except Exception:
                continue
            status = body.get("status")
            pr = body.get("progress_percent")
            cur = body.get("current_step")
            tot = body.get("total_steps")
            if status is None and body.get("artifacts"):
                status = "completed"
            got_any = True
            # Suppress noisy unknown states for the fallback ID
            if status and status != "unknown":
                if status != last["status"] or pr != last["pr"] or cur != last["cur"] or tot != last["tot"]:
                    print(f"[HTTP] status: job_id={jid} status={status} progress={pr} cur={cur} tot={tot}")
                    last.update({"status": status, "pr": pr, "cur": cur, "tot": tot})
            if status == "completed":
                return True
            if status == "error":
                return False
        if not got_any:
            # No readable status yet; keep waiting
            pass
        time.sleep(2)

def _normalize_workflow_payload(raw: dict) -> Tuple[dict, Optional[str]]:
    """Translate a saved ComfyUI workflow JSON into the payload our headless
    server expects, and trim it to the Comfy /prompt schema.

    Accepts either:
    - full export with keys like {"prefer_video": bool, "prompt": {...}}
    - or just the {node_id: {..}} mapping itself.

    Returns (payload, error_message_if_any).
    """
    try:
        prefer_video = bool(raw.get("prefer_video")) if isinstance(raw, dict) else False
        prompt_obj = raw.get("prompt") if isinstance(raw, dict) else None
        if prompt_obj is None:
            # Assume the entire object is the nodes mapping
            prompt_obj = raw

        if not isinstance(prompt_obj, dict) or not prompt_obj:
            return {}, "prompt graph is missing or invalid"

        # Always provide a client_id for Comfy
        cid = raw.get("client_id") if isinstance(raw, dict) else None
        if not isinstance(cid, str) or not cid:
            cid = str(uuid.uuid4())

        # Headless server consumes prefer_video at top-level, but Comfy should
        # only receive {client_id, prompt}. The headless server will forward
        # only those keys downstream.
        payload = {"client_id": cid, "prompt": prompt_obj}
        if prefer_video:
            payload["prefer_video"] = True

        # Optionally pass through comfy_url if present in the source
        cu = raw.get("comfy_url") if isinstance(raw, dict) else None
        if isinstance(cu, str) and cu:
            payload["comfy_url"] = cu

        return payload, None
    except Exception as e:
        return {}, f"normalize error: {e}"


async def main():
    ap = argparse.ArgumentParser(description="Confirm WS events flow for queued job.")
    ap.add_argument("--base", required=True, help="Base URL of your headless ASGI app (e.g., https://...modal.run)")
    ap.add_argument("--ws", default=None, help="WS URL (wss://.../events). If omitted, derived from --base.")
    ap.add_argument("--token", default=None, help="Bearer token for HTTP/WS (if required).")
    ap.add_argument(
        "--prompt",
        default="wan22_t2v_flexible.json",
        help="Path to ComfyUI workflow JSON (defaults to wan22_t2v_flexible.json).",
    )
    ap.add_argument("--timeout", type=int, default=300, help="Overall timeout (seconds).")
    ap.add_argument(
        "--comfy-url",
        default=None,
        help="Override Comfy base URL; placed into request payload for the headless server.",
    )
    ap.add_argument("--project-id", default=None, help="Project ID to associate with the generated media.")
    ap.add_argument("--user-id", default=None, help="User ID to associate with the generated media.")
    args = ap.parse_args()

    base = args.base
    ws_url = args.ws or derive_ws_from_base(base)
    token = args.token

    # Load workflow and translate to headless/Comfy payload
    try:
        with open(args.prompt, "r", encoding="utf-8") as f:
            raw_workflow = json.load(f)
    except Exception as e:
        print(f"[ERR] Failed to read workflow file: {e}")
        sys.exit(2)

    prompt_payload, err = _normalize_workflow_payload(raw_workflow)
    if err:
        print(f"[ERR] {err}")
        sys.exit(2)

    # Allow overriding the Comfy URL to be used by the headless server
    if args.comfy_url:
        prompt_payload["comfy_url"] = args.comfy_url
    # Optional association for immediate backend import
    if args.project_id:
        prompt_payload["project_id"] = args.project_id
    if args.user_id:
        prompt_payload["user_id"] = args.user_id

    # Connect WS first, then queue, then wait for events for that prompt_id
    # We’ll open WS and run listener in the background; once queued, we pass prompt_id into the listener
    # Simpler approach: queue first, get prompt_id, then connect WS and listen filtered by that id
    # (risk of missing earliest status snapshot is fine for a confirm script)
    pid = queue_prompt(base, prompt_payload, token)
    if not pid:
        sys.exit(3)

    # Prefer HTTP polling as authoritative; keep WS for logs only.
    loop = asyncio.get_running_loop()
    ws_task = asyncio.create_task(ws_listener(ws_url, token, expected_id=pid, timeout_s=args.timeout))
    client_id = str(prompt_payload.get("client_id")) if isinstance(prompt_payload.get("client_id"), str) else None
    poll_ids = [pid] + ([client_id] if client_id else [])
    http_done = await loop.run_in_executor(None, lambda: http_progress_poll(base, token, poll_ids, args.timeout))
    try:
        ws_task.cancel()
    except Exception:
        pass
    sys.exit(0 if http_done else 4)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
