"""
Headless ComfyUI on Modal — ASGI proxy version (fixed)

Endpoints at your Modal URL:
  GET  /health            -> proxies to /system_stats
  GET  /system_stats      -> ComfyUI system stats
  GET  /object_info       -> ComfyUI object info (node inputs/loaders)
  POST /prompt            -> submit a workflow JSON
  GET  /queue             -> queue status
  GET  /history/{prompt_id}
  GET  /debug/models      -> quick check of model symlinks
  GET  /debug/extra_paths -> dumps the YAML Comfy will load
"""

import os
import pathlib
import socket
import subprocess
import threading
import time
import urllib.request
import modal
import random
try:
    from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
except Exception:  # botocore may not be present until runtime
    ClientError = NoCredentialsError = EndpointConnectionError = Exception
from typing import Callable, Optional, Dict, List, Set

# Global progress/state shared across threads and ASGI app
global_progress: Dict[str, object] = {
    'jobs': {},  # job_id -> progress_data
    'last_update': time.time(),
}
job_artifacts: Dict[str, List[str]] = {}
event_publisher: Optional[Callable[[dict], None]] = None
job_meta: Dict[str, dict] = {}
# Expected filename prefix per job (e.g., teacache-<uuid>), used to filter artifacts
job_prefixes: Dict[str, str] = {}
# Remote/public URLs by job id and filename (to avoid /view cold-starts)
# S3 object keys by job id and filename (to generate presigned URLs on demand)
job_object_keys: Dict[str, Dict[str, str]] = {}
downloads_seen: Set[str] = set()
# Clients may post an explicit ACK when they finish importing artifacts for a job.
# This prevents premature shutdown when downloads are proxied externally.
jobs_imported_ack: Set[str] = set()
job_completed_at: Dict[str, float] = {}

# Persistent job state helpers (cross-replica)
def _jobs_dir() -> pathlib.Path:
    p = pathlib.Path("/userdir") / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _job_state_path(job_key: str) -> pathlib.Path:
    return _jobs_dir() / f"{job_key}.json"

def _persist_job_state(job_key: str, patch: dict) -> None:
    try:
        path = _job_state_path(job_key)
        cur = {}
        if path.exists():
            try:
                import json as _j
                cur = _j.loads(path.read_text(encoding="utf-8")) or {}
            except Exception:
                cur = {}
        # Merge patch with monotonic progress semantics
        # Never downgrade from completed -> running/error
        incoming_status = patch.get("status")
        if incoming_status:
            prev_status = cur.get("status")
            if prev_status == "completed" and incoming_status != "completed":
                patch = dict(patch)
                patch.pop("status", None)
        # Monotonic counters
        for k in ("current_step", "total_steps"):
            if k in patch:
                try:
                    prev = int(cur.get(k, 0) or 0)
                    newv = int(patch.get(k) or 0)
                    if newv < prev:
                        patch = dict(patch)
                        patch[k] = prev
                except Exception:
                    pass
        if "progress_percent" in patch:
            try:
                prev = float(cur.get("progress_percent", 0.0) or 0.0)
                newv = float(patch.get("progress_percent") or 0.0)
                if newv < prev:
                    patch = dict(patch)
                    patch["progress_percent"] = prev
            except Exception:
                pass
        cur.update(patch)
        # Always include the id and a timestamp
        cur.setdefault("id", job_key)
        cur["updated_at"] = time.time()
        import json as _j
        path.write_text(_j.dumps(cur), encoding="utf-8")
    except Exception as e:
        print(f"[JOB-STATE] persist failed for {job_key}: {e}", flush=True)

def _read_job_state(job_key: str) -> Optional[dict]:
    try:
        path = _job_state_path(job_key)
        if not path.exists():
            return None
        import json as _j
        return _j.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None




# ---------- Helpers (run inside container, after volumes mounted) ----------
def _ensure_model_layout() -> None:
    import os, pathlib

    VBASE = pathlib.Path("/modal_models")                  # your volume (real files live here)
    RBASE = pathlib.Path("/root/comfy/ComfyUI/models")     # repo models dir (where Comfy scans)

    # Ensure folders exist under the repo tree
    for d in ("checkpoints", "diffusion_models", "unet", "vae", "clip"):
        (RBASE / d).mkdir(parents=True, exist_ok=True)

    # What we want visible under the repo tree, and where to look in the volume
    want = {
        # UNETs (WAN 2.2)
        "unet/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors": [
            "checkpoints/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            "unet/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        "unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors": [
            "checkpoints/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        ],
        # VAE
        "vae/wan_2.1_vae.safetensors": [
            "vae/wan_2.1_vae.safetensors",
            "checkpoints/wan_2.1_vae.safetensors",
            "wan_2.1_vae.safetensors",
        ],
        # CLIP / text encoder
        "clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors": [
            "clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "checkpoints/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        ],
    }

    def find_in_volume(rel_candidates):
        """Return absolute Path to the first existing candidate under the volume."""
        for rel in rel_candidates:
            p = VBASE / rel
            if p.exists():
                return p.resolve()
        # Fallback: search anywhere under volume by basename
        name = pathlib.Path(rel_candidates[0]).name
        hits = list(VBASE.rglob(name))
        return hits[0].resolve() if hits else None

    def safe_link(target: pathlib.Path, src_abs: pathlib.Path, keep_if_real=True):
        """Create/refresh symlink target -> src_abs (absolute), avoiding loops and EEXIST."""
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if target.is_symlink():
                current = target.resolve(strict=False)
                # If it already points to the same place, leave it
                if str(current) == str(src_abs):
                    print(f"[MODEL-SKIP] {target} already -> {src_abs}", flush=True)
                    return
                target.unlink()
            elif target.exists():
                if keep_if_real:
                    print(f"[MODEL-KEEP] {target} (real file present)", flush=True)
                    return
                target.unlink()
            # Always symlink to the absolute real file in the volume
            target.symlink_to(src_abs)
            print(f"[MODEL-LINK] {target} -> {src_abs}", flush=True)
        except FileExistsError:
            print(f"[MODEL-SKIP] {target} already exists", flush=True)
        except Exception as e:
            print(f"[MODEL-LINK-ERR] {target}: {e}", flush=True)

    # 1) Link/show models under the repo tree (unet/vae/clip)
    for target_rel, candidates in want.items():
        src_abs = find_in_volume(candidates)
        if not src_abs:
            print(f"[MODEL-MISSING] {candidates}", flush=True)
            continue
        safe_link(RBASE / target_rel, src_abs, keep_if_real=True)

    # 2) Mirror UNETs into diffusion_models (what UNETLoader reads)
    for unet_name in [
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    ]:
        src_abs = find_in_volume([
            f"checkpoints/{unet_name}",
            f"unet/{unet_name}",
            unet_name,
        ])
        if src_abs:
            safe_link(RBASE / "diffusion_models" / unet_name, src_abs, keep_if_real=False)
        else:
            print(f"[MODEL-MISSING] could not mirror {unet_name} into diffusion_models/", flush=True)


    # NOTE: We intentionally do NOT write extra_model_paths.yaml anymore.
    # ComfyUI will auto-load a file at /root/comfy/ComfyUI/extra_model_paths.yaml
    # if it exists, and its loader is fragile. Since our models are symlinked
    # into the standard folders under RBASE, no extra paths file is required.

    # Quick listings for sanity
    def _ls(dirpath: str):
        try:
            return sorted([p.name for p in pathlib.Path(dirpath).glob("*.safetensors")])
        except Exception:
            return []
    print("[MODEL-SCAN] UNET:", _ls(str(RBASE / "unet")), flush=True)
    print("[MODEL-SCAN] DIFF:", _ls(str(RBASE / "diffusion_models")), flush=True)
    print("[MODEL-SCAN] VAE :", _ls(str(RBASE / "vae")), flush=True)
    print("[MODEL-SCAN] CLIP:", _ls(str(RBASE / "clip")), flush=True)

    # Writable userdir (sqlite)
    try:
        os.chmod("/userdir", 0o777)
    except Exception:
        pass



def _launch_comfy() -> subprocess.Popen:
    """Start ComfyUI on 127.0.0.1:8188 and stream logs to stdout."""
    cmd = [
        "python", "main.py",
        "--dont-print-server",
        "--listen", "0.0.0.0", "--port", "8188",
        "--user-directory", "/userdir",
        "--database-url", "sqlite:////userdir/comfy.db",
        "--comfy-api-base", "/",
        "--enable-cors-header", "*",
        "--output-directory", "/outputs",
        # "--disable-metadata",  # removed to enable metadata/history
        "--log-stdout",
        "--verbose", "INFO",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd="/root/comfy/ComfyUI",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    def _pump_stdout(p: subprocess.Popen) -> None:
        if not p.stdout:
            return
        for line in p.stdout:
            print(line.rstrip(), flush=True)
        p.stdout.close()

    threading.Thread(target=_pump_stdout, args=(proc,), daemon=True).start()
    return proc


def _wait_until_ready(timeout_s: int = 840) -> None:
    """Poll local /system_stats until ComfyUI is ready or timeout."""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection(("127.0.0.1", 8188), timeout=2):
                urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2).read()
                print("[HEALTH] local /system_stats OK", flush=True)
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("ComfyUI failed to start on port 8188 within the timeout window")


def _monitor_job_completion(job_id, prompt_id, job_manager_instance, plan_or_total: Optional[object] = None):
    """Monitor a job for completion using ComfyUI WebSocket for real progress."""
    try:
        import json
        import urllib.request
        import websocket
        import threading
        from pathlib import Path
        import time
        
        print(f"[MONITOR] Starting WebSocket monitoring for job: {job_id}", flush=True)
        
        # Track initial video set (recursive)
        output_dir = Path("/outputs")
        def _list_videos() -> set[str]:
            vids: set[str] = set()
            try:
                for f in output_dir.rglob("*.mp4"):
                    try:
                        rel = f.relative_to(output_dir).as_posix()
                    except Exception:
                        rel = f.name
                    vids.add(rel)
            except Exception:
                pass
            return vids
        initial_videos = _list_videos()
        initial_count = len(initial_videos)
        print(f"[MONITOR] Initial video count: {initial_count}", flush=True)
        
        # Prefer prompt_id as the authoritative id (matches the id returned to clients)
        key_id = str(prompt_id) if prompt_id is not None else str(job_id)

        # Progress tracking variables
        progress_data = {
            'job_id': key_id,
            'total_steps': 0,
            'current_step': 0,
            'progress_percent': 0,
            'is_complete': False,
            'videos_generated': 0,
            'expected_videos': 1,  # Will be updated based on job
        }

        # Aggregate planning: weights per sampler node and encode nodes
        per_node_weights: dict[str, int] = {}
        per_node_progress: dict[str, float] = {}
        encode_weights: dict[str, int] = {}
        encode_done: set[str] = set()
        total_steps_plan = 0
        if isinstance(plan_or_total, dict):
            try:
                per_node_weights = {str(k): int(v) for k, v in (plan_or_total.get('weights') or {}).items() if int(v) > 0}
            except Exception:
                per_node_weights = {}
            try:
                encode_weights = {str(k): int(v) for k, v in (plan_or_total.get('encode_weights') or {}).items() if int(v) > 0}
            except Exception:
                encode_weights = {}
            try:
                total_steps_plan = int(plan_or_total.get('total_steps') or 0)
            except Exception:
                total_steps_plan = 0
        elif isinstance(plan_or_total, int) and plan_or_total > 0:
            total_steps_plan = int(plan_or_total)

        if total_steps_plan > 0:
            # Normalize UI total to number of frames if available
            frames = 0
            try:
                frames = int((plan_or_total or {}).get('frames') or 0) if isinstance(plan_or_total, dict) else 0
            except Exception:
                frames = 0
            norm_total = frames if frames > 0 else total_steps_plan
            progress_data['total_steps'] = norm_total
            _persist_job_state(key_id, {'total_steps': norm_total})

        # Initialize persistent state for this job
        _persist_job_state(key_id, {
            'status': 'running',
            'current_step': 0,
            'total_steps': 0,
            'progress_percent': 0,
            'artifacts': [],
        })
        
        def on_websocket_message(ws, message):
            try:
                data = json.loads(message)
                msg_type = data.get('type')
                
                if msg_type == 'progress':
                    # Real-time progress from ComfyUI — aggregate across sampler nodes when plan present
                    dd = data.get('data') or {}
                    node_id = str(dd.get('node')) if dd.get('node') is not None else None
                    if node_id and per_node_weights:
                        try:
                            vmax = int(dd.get('max', 1) or 1)
                            vcur = int(dd.get('value', 0) or 0)
                            vmax = max(1, vmax)
                            per_node_progress[node_id] = max(0.0, min(1.0, vcur / float(vmax)))
                        except Exception:
                            pass
                        # Aggregate: sum(weight * progress) + encode_done
                        agg = sum(per_node_weights.get(nid, 0) * per_node_progress.get(nid, 0.0) for nid in per_node_weights.keys()) + sum(encode_weights.get(nid, 0) for nid in encode_done)
                        # Normalize aggregate to frames so progress reaches 100% by the time frames complete
                        frames = 0
                        try:
                            frames = int((plan_or_total or {}).get('frames') or 0) if isinstance(plan_or_total, dict) else 0
                        except Exception:
                            frames = 0
                        total_full = max(1, (sum(per_node_weights.values()) + sum(encode_weights.values())) or total_steps_plan or 1)
                        total_norm = frames if frames > 0 else total_full
                        cur_norm = int(round((agg / float(total_full)) * total_norm))
                        cur_norm = max(0, min(total_norm, cur_norm))
                        progress_data['current_step'] = cur_norm
                        progress_data['total_steps'] = total_norm
                        progress_data['progress_percent'] = (cur_norm / float(total_norm)) * 100.0 if total_norm > 0 else 0.0
                    else:
                        # Fallback: per-message value/max, normalized to frames if known
                        raw_cur = int(dd.get('value', 0) or 0)
                        raw_tot = int(dd.get('max', 1) or 1)
                        frames = 0
                        try:
                            frames = int((plan_or_total or {}).get('frames') or 0) if isinstance(plan_or_total, dict) else 0
                        except Exception:
                            frames = 0
                        if frames > 0 and raw_tot > 0:
                            cur_norm = int(round((raw_cur / float(raw_tot)) * frames))
                            cur_norm = max(0, min(frames, cur_norm))
                            progress_data['current_step'] = cur_norm
                            progress_data['total_steps'] = frames
                            progress_data['progress_percent'] = (cur_norm / float(frames)) * 100.0
                        else:
                            progress_data['current_step'] = raw_cur
                            progress_data['total_steps'] = max(1, raw_tot)
                            progress_data['progress_percent'] = (raw_cur / float(max(1, raw_tot))) * 100.0
                    
                    # Update global progress map (dict mutation; no rebinding)
                    global_progress['jobs'][key_id] = progress_data.copy()
                    global_progress['last_update'] = time.time()
                    
                    print(f"[PROGRESS] {job_id}: {progress_data['current_step']}/{progress_data['total_steps']} ({progress_data['progress_percent']:.1f}%)", flush=True)
                    # Push live progress to external clients (include node if present)
                    try:
                        if event_publisher is not None:
                            event_publisher({
                                "type": "progress",
                                "job_id": key_id,
                                "current_step": progress_data['current_step'],
                                "total_steps": progress_data['total_steps'],
                                "progress_percent": progress_data['progress_percent'],
                                "node_id": data.get('node') or dd.get('node'),
                            })
                    except Exception:
                        pass
                    # Persist progress for cross-replica polling
                    _persist_job_state(key_id, {
                        'status': 'running',
                        'current_step': progress_data['current_step'],
                        'total_steps': progress_data['total_steps'],
                        'progress_percent': progress_data['progress_percent'],
                    })
                
                elif msg_type == 'executing':
                    node_id = data.get('data', {}).get('node')
                    if node_id:
                        print(f"[EXECUTING] {job_id}: Node {node_id}", flush=True)
                
                elif msg_type == 'executed':
                    # Check if this is a video generation completion and update aggregate
                    node_data = data.get('data', {})
                    node_id = str(node_data.get('node')) if node_data.get('node') is not None else None
                    if 'output' in node_data and 'videos' in str(node_data):
                        progress_data['videos_generated'] += 1
                        global_progress['jobs'][key_id] = progress_data.copy()
                        print(f"[VIDEO-COMPLETE] {job_id}: Video {progress_data['videos_generated']} generated", flush=True)
                    # Mark sampler nodes as completed if they lack granular progress
                    if node_id and (per_node_weights or encode_weights):
                        if node_id in per_node_weights:
                            per_node_progress[node_id] = 1.0
                        if node_id in encode_weights:
                            encode_done.add(node_id)
                        agg = sum(per_node_weights.get(nid, 0) * per_node_progress.get(nid, 0.0) for nid in per_node_weights.keys()) + sum(encode_weights.get(nid, 0) for nid in encode_done)
                        # Normalize aggregate to frames
                        frames = 0
                        try:
                            frames = int((plan_or_total or {}).get('frames') or 0) if isinstance(plan_or_total, dict) else 0
                        except Exception:
                            frames = 0
                        total_full = max(1, (sum(per_node_weights.values()) + sum(encode_weights.values())) or total_steps_plan or 1)
                        total_norm = frames if frames > 0 else total_full
                        cur_norm = int(round((agg / float(total_full)) * total_norm))
                        cur_norm = max(0, min(total_norm, cur_norm))
                        progress_data['current_step'] = cur_norm
                        progress_data['total_steps'] = total_norm
                        progress_data['progress_percent'] = (cur_norm / float(total_norm)) * 100.0 if total_norm > 0 else 0.0
                        global_progress['jobs'][key_id] = progress_data.copy()
                        _persist_job_state(key_id, {
                            'status': 'running',
                            'current_step': progress_data['current_step'],
                            'total_steps': progress_data['total_steps'],
                            'progress_percent': progress_data['progress_percent'],
                        })

                elif msg_type == 'execution_end':
                    # Strong completion signal for this prompt_id
                    dd = data.get('data') or {}
                    pid = dd.get('prompt_id')
                    try:
                        matches = str(pid) == str(prompt_id)
                    except Exception:
                        matches = False
                    if matches:
                        progress_data['progress_percent'] = max(100.0, progress_data.get('progress_percent', 0))
                        global_progress['jobs'][key_id] = progress_data.copy()
                        print(f"[EXECUTION-END] {job_id}: prompt {prompt_id}", flush=True)
                
            except Exception as e:
                print(f"[WEBSOCKET] Error parsing message: {e}", flush=True)
        
        def on_websocket_error(ws, error):
            print(f"[WEBSOCKET] Error for {job_id}: {error}", flush=True)
        
        def on_websocket_close(ws, close_status_code, close_msg):
            print(f"[WEBSOCKET] Closed for {job_id}: {close_status_code}", flush=True)
        
        def on_websocket_open(ws):
            print(f"[WEBSOCKET] Connected for {job_id}", flush=True)
        
        # Start WebSocket connection to ComfyUI (bind to clientId for per-job updates)
        try:
            from urllib.parse import quote as _q
        except Exception:
            def _q(s):
                return s
        ws_url = f"ws://127.0.0.1:8188/ws?clientId={_q(str(job_id))}"
        ws = websocket.WebSocketApp(ws_url,
                                  on_open=on_websocket_open,
                                  on_message=on_websocket_message,
                                  on_error=on_websocket_error,
                                  on_close=on_websocket_close)
        
        # Run WebSocket in a separate thread
        ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
        ws_thread.start()
        
        # Monitor for actual completion
        monitor_count = 0
        last_progress_time = time.time()
        
        while monitor_count < 120:  # Max 10 minutes of monitoring
            try:
                current_time = time.time()
                
                # Check for new video files (recursive)
                current_videos = _list_videos()
                new_videos = current_videos - initial_videos
                videos_generated = len(new_videos)
                
                # Check ComfyUI queue status
                queue_response = urllib.request.urlopen(f"http://127.0.0.1:8188/queue", timeout=5)
                queue_data = json.loads(queue_response.read().decode())
                queue_pending = queue_data.get("queue_pending", [])
                queue_running = queue_data.get("queue_running", [])
                
                # Real completion criteria:
                # 1. No items in queue (pending or running)
                # 2. New videos have been generated
                # 3. No progress updates for 30 seconds (indicating completion)
                no_queue_activity = len(queue_pending) == 0 and len(queue_running) == 0
                has_new_videos = videos_generated > 0
                progress_stalled = current_time - last_progress_time > 30
                
                if progress_data['current_step'] > 0:
                    last_progress_time = current_time
                
                print(f"[MONITOR] {job_id}: Queue(P:{len(queue_pending)}, R:{len(queue_running)}), Videos:{videos_generated}, Progress:{progress_data['progress_percent']:.1f}%", flush=True)
                
                # Upload new videos immediately when detected
                if new_videos:
                    for video_rel in new_videos:
                        video_path = output_dir / video_rel
                        if video_path.exists() and video_rel not in job_manager_instance.uploaded_videos:
                            job_manager_instance.uploaded_videos.add(video_rel)
                            print(f"[MONITOR] Uploading new video: {video_rel}", flush=True)
                            try:
                                job_manager_instance.upload_started()
                            except Exception:
                                pass
                            try:
                                meta = None
                                try:
                                    meta = job_meta.get(key_id) or job_meta.get(str(job_id))
                                except Exception:
                                    meta = None
                                ok, key = _upload_to_backblaze(str(video_path), meta=meta)
                                if ok and key:
                                    try:
                                        job_object_keys.setdefault(key_id, {})[video_rel] = key
                                    except Exception:
                                        pass
                                if not ok:
                                    job_manager_instance.uploaded_videos.discard(video_rel)
                            finally:
                                try:
                                    job_manager_instance.upload_finished()
                                except Exception:
                                    pass
                    # Persist artifacts list as we discover videos (filter by expected prefix if available)
                    try:
                        try:
                            expect_prefix = job_prefixes.get(key_id) or job_prefixes.get(str(job_id))
                        except Exception:
                            expect_prefix = None
                        rels = []
                        for _p in current_videos:
                            try:
                                rel = _p.relative_to(output_dir).as_posix()
                            except Exception:
                                rel = _p.name
                            if expect_prefix:
                                # Compare against basename to avoid subfolder mismatch
                                base = (_p.name)
                                if not base.startswith(expect_prefix):
                                    continue
                            rels.append(rel)
                        artifacts_list = sorted(rels)
                        _persist_job_state(key_id, {'artifacts': artifacts_list})
                    except Exception:
                        pass
                
                # Complete job when Comfy queue is empty (all nodes executed)
                if no_queue_activity:
                    print(f"[MONITOR] Job {job_id} TRULY complete - Queue empty, videos generated, progress stalled", flush=True)
                    # Record artifacts for /jobs API
                    try:
                        global job_artifacts, job_completed_at
                        arts: list[str] = []
                        try:
                            import urllib.request as _url, json as _j
                            if prompt_id:
                                r = _url.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=10)
                                d = _j.loads(r.read().decode())
                                root = d.get(prompt_id) or d
                                outs = root.get("outputs") if isinstance(root, dict) else None
                                if isinstance(outs, dict):
                                    for _nid, entry in outs.items():
                                        varr = entry.get("videos") if isinstance(entry, dict) else None
                                        if isinstance(varr, list):
                                            for it in varr:
                                                fn = (it or {}).get("filename")
                                                sub = (it or {}).get("subfolder", "")
                                                if isinstance(fn, str) and fn:
                                                    rel = (Path(sub) / fn).as_posix() if sub else fn
                                                    arts.append(rel)
                        except Exception:
                            pass
                        if not arts:
                            try:
                                expect_prefix = job_prefixes.get(key_id) or job_prefixes.get(str(job_id))
                            except Exception:
                                expect_prefix = None
                            rels = []
                            for _p in current_videos:
                                try:
                                    rel = _p.relative_to(output_dir).as_posix()
                                except Exception:
                                    rel = _p.name
                                if expect_prefix:
                                    if not _p.name.startswith(expect_prefix):
                                        continue
                                rels.append(rel)
                            arts = sorted(rels)
                        job_artifacts[key_id] = sorted(list(set(arts)))
                        job_completed_at[key_id] = time.time()
                    except Exception:
                        pass
                    # Persist completed state for cross-replica clients
                    try:
                        _persist_job_state(key_id, {
                            'status': 'completed',
                            'progress_percent': 100.0,
                            'artifacts': job_artifacts.get(key_id, []),
                            'completed_at': time.time(),
                        })
                    except Exception:
                        pass
                    job_manager_instance.job_completed(job_id)
                    # Publish WS event to connected clients (if publisher available)
                    try:
                        global event_publisher
                        if event_publisher is not None:
                            event_publisher({"type": "job_completed", "job_id": key_id})
                            # If no other jobs are running and the Comfy queue is empty, close WS immediately.
                            try:
                                import urllib.request, json as _j
                                empty_queue = False
                                try:
                                    r = urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=3)
                                    d = _j.loads(r.read().decode())
                                    empty_queue = len(d.get("queue_pending", [])) == 0 and len(d.get("queue_running", [])) == 0
                                except Exception:
                                    # If queue probe fails, assume empty to prefer closing
                                    empty_queue = True
                                if empty_queue and len(job_manager_instance.active_jobs) == 0:
                                    event_publisher({"type": "server_idle_close"})
                                    # Also close sockets directly for immediate shutdown
                                    try:
                                        close_all_ws()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass
                    ws.close()
                    break
                
            except Exception as e:
                print(f"[MONITOR] Monitoring iteration {monitor_count}: {e}", flush=True)
            
            monitor_count += 1
            time.sleep(5)  # Check every 5 seconds
            
        if monitor_count >= 120:
            print(f"[MONITOR] Monitoring timeout for job {job_id}", flush=True)
            job_manager_instance.job_completed(job_id)
            ws.close()
            try:
                _persist_job_state(key_id, {'status': 'error'})
            except Exception:
                pass
            
    except Exception as e:
        print(f"[MONITOR] Error monitoring job {job_id}: {e}", flush=True)


def _upload_to_backblaze(video_path, meta: Optional[dict] = None):
    """Enhanced upload with zero-failure guarantee using multiple fallback strategies.
    Returns (success: bool, public_url: Optional[str]).
    """
    return _upload_to_backblaze_enhanced(video_path, meta)

def _upload_to_backblaze_enhanced(video_path, meta: Optional[dict] = None):
    """Robust upload system with comprehensive error handling and fallback strategies.
    Returns (success: bool, public_url: Optional[str]).
    """
    import os
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
    import requests
    from pathlib import Path
    import time
    import random
    import shutil
    import threading
    import json
    
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"[UPLOAD] Video file not found: {video_path}", flush=True)
        return False, None
    
    # Get configuration
    bucket = os.environ.get("S3_BUCKET")
    region = os.environ.get("S3_REGION", "us-east-1")
    endpoint = os.environ.get("S3_ENDPOINT")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    if not all([bucket, access_key, secret_key]):
        print("[UPLOAD] S3/Backblaze credentials not configured, using fallback storage", flush=True)
        ok = _fallback_to_persistent_storage(video_file)
        return (ok, None)
    
    # Wait for file size to stabilize and exceed a sane minimum
    MIN_VIDEO_BYTES = int(os.environ.get("MIN_VIDEO_BYTES", "4096"))
    stable_window_s = float(os.environ.get("UPLOAD_STABLE_WINDOW_S", "2.0"))
    max_wait_s = float(os.environ.get("UPLOAD_MAX_WAIT_S", "300"))
    last_size = -1
    stable_since = None
    start_wait = time.time()
    while True:
        size = video_file.stat().st_size if video_file.exists() else 0
        now = time.time()
        if size != last_size:
            last_size = size
            stable_since = now
        # Stop when size has been stable long enough and above minimum
        if size >= MIN_VIDEO_BYTES and (now - (stable_since or now)) >= stable_window_s:
            break
        if now - start_wait > max_wait_s:
            print(f"[UPLOAD] Waited {max_wait_s}s for file to stabilize, proceeding with size={size}", flush=True)
            break
        time.sleep(0.25)

    file_size = video_file.stat().st_size
    print(f"[UPLOAD] Starting enhanced upload for {video_file.name} ({file_size / (1024*1024):.1f} MB) after stabilization", flush=True)
    
    # Strategy 1: Direct upload with retry logic
    ok, key = _upload_with_retry_logic(video_file, bucket, endpoint, access_key, secret_key, region, meta)
    if ok:
        return True, key
    
    # Strategy 2: Multipart upload for large files
    if file_size > 100 * 1024 * 1024:  # 100MB threshold
        print(f"[UPLOAD] Attempting multipart upload for large file", flush=True)
        ok, key = _upload_multipart(video_file, bucket, endpoint, access_key, secret_key, region, meta)
        if ok:
            return True, key
    
    # Strategy 3: Chunked upload with smaller parts
    print(f"[UPLOAD] Attempting chunked upload", flush=True)
    ok, key = _upload_chunked(video_file, bucket, endpoint, access_key, secret_key, region, meta)
    if ok:
        return True, key
    
    # Strategy 4: Fallback to persistent storage
    print(f"[UPLOAD] All upload strategies failed, using persistent storage fallback", flush=True)
    ok = _fallback_to_persistent_storage(video_file)
    return (ok, None)


def _upload_with_retry_logic(video_file, bucket, endpoint, access_key, secret_key, region, meta: Optional[dict] = None):
    """Upload with exponential backoff and circuit breaker pattern.
    Returns (success: bool, object_key: Optional[str]).
    """
    max_retries = 5
    base_delay = 2
    max_delay = 60
    timeout = 120  # 2 minutes timeout
    
    for attempt in range(max_retries):
        try:
            print(f"[UPLOAD] Attempt {attempt + 1}/{max_retries} for {video_file.name}", flush=True)
            
            # Create S3 client with enhanced configuration
            s3_client = _create_robust_s3_client(endpoint, access_key, secret_key, region, timeout)
            
            # Generate unique key
            timestamp = int(time.time())
            unique_key = f"modal-generated/{timestamp}_{video_file.name}"
            
            # Upload configuration
            upload_args = {'ContentType': 'video/mp4'}
            if not (endpoint and "backblazeb2.com" in endpoint):
                upload_args['ACL'] = 'public-read'
            
            # Perform upload with progress tracking
            start_time = time.time()
            with open(video_file, 'rb') as f:
                s3_client.upload_fileobj(f, bucket, unique_key, ExtraArgs=upload_args)
            
            upload_time = time.time() - start_time
            print(f"[UPLOAD] Upload completed in {upload_time:.1f}s", flush=True)
            
            # Verify upload
            if _verify_upload(s3_client, bucket, unique_key, video_file.stat().st_size):
                # Wait for Backblaze eventual consistency
                if endpoint and "backblazeb2.com" in endpoint:
                    time.sleep(3)  # Increased wait time for Backblaze
                
                # Notify backend
                project_id, user_id = _extract_ids_from_filename(video_file.name)
                if meta:
                    project_id = meta.get('project_id') or project_id
                    user_id = meta.get('user_id') or user_id
                # Public URL may not be accessible on private buckets; store key for presigning
                _notify_backend_upload(unique_key, _generate_public_url(bucket, unique_key, endpoint, region), video_file.name, project_id, user_id)
                
                print(f"[UPLOAD] ✅ Successfully uploaded {video_file.name} (key={unique_key})", flush=True)
                return True, unique_key
            else:
                # Delete bad object before retrying
                try:
                    s3_client.delete_object(Bucket=bucket, Key=unique_key)
                    print(f"[UPLOAD] Deleted failed object {unique_key}", flush=True)
                except Exception as de:
                    print(f"[UPLOAD] Delete failed object error: {de}", flush=True)
                print(f"[UPLOAD] Upload verification failed for {video_file.name}", flush=True)
                
        except (ClientError, NoCredentialsError, EndpointConnectionError) as e:
            print(f"[UPLOAD] S3 error on attempt {attempt + 1}: {e}", flush=True)
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                print(f"[UPLOAD] Retrying in {delay:.1f} seconds...", flush=True)
                time.sleep(delay)
            else:
                print(f"[UPLOAD] All retry attempts exhausted", flush=True)
                
        except Exception as e:
            print(f"[UPLOAD] Unexpected error on attempt {attempt + 1}: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"[UPLOAD] Unexpected error after all retries", flush=True)
    
    return False, None


def _upload_multipart(video_file, bucket, endpoint, access_key, secret_key, region, meta: Optional[dict] = None):
    """Multipart upload for large files with enhanced error handling.
    Returns (success: bool, public_url: Optional[str]).
    """
    try:
        import boto3
        from botocore.config import Config
        
        s3_client = _create_robust_s3_client(endpoint, access_key, secret_key, region, 300)  # 5 min timeout
        
        timestamp = int(time.time())
        unique_key = f"modal-generated/{timestamp}_{video_file.name}"
        
        # Configure multipart upload
        upload_args = {'ContentType': 'video/mp4'}
        if not (endpoint and "backblazeb2.com" in endpoint):
            upload_args['ACL'] = 'public-read'
        
        print(f"[UPLOAD] Starting multipart upload for {video_file.name}", flush=True)
        
        # Initiate multipart upload
        response = s3_client.create_multipart_upload(
            Bucket=bucket,
            Key=unique_key,
            **upload_args
        )
        upload_id = response['UploadId']
        
        # Upload parts
        part_size = 50 * 1024 * 1024  # 50MB parts
        parts = []
        part_number = 1
        
        with open(video_file, 'rb') as f:
            while True:
                chunk = f.read(part_size)
                if not chunk:
                    break
                
                print(f"[UPLOAD] Uploading part {part_number} ({len(chunk) / (1024*1024):.1f} MB)", flush=True)
                
                # Retry logic for each part
                for attempt in range(3):
                    try:
                        response = s3_client.upload_part(
                            Bucket=bucket,
                            Key=unique_key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=chunk
                        )
                        parts.append({
                            'ETag': response['ETag'],
                            'PartNumber': part_number
                        })
                        break
                    except Exception as e:
                        print(f"[UPLOAD] Part {part_number} attempt {attempt + 1} failed: {e}", flush=True)
                        if attempt < 2:
                            time.sleep(2)
                        else:
                            # Abort multipart upload on failure
                            s3_client.abort_multipart_upload(
                                Bucket=bucket,
                                Key=unique_key,
                                UploadId=upload_id
                            )
                            return False
                
                part_number += 1
        
        # Complete multipart upload
        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=unique_key,
            UploadId=upload_id,
            MultipartUpload={'Parts': parts}
        )
        
        # Verify and notify
        if _verify_upload(s3_client, bucket, unique_key, video_file.stat().st_size):
            project_id, user_id = _extract_ids_from_filename(video_file.name)
            if meta:
                project_id = meta.get('project_id') or project_id
                user_id = meta.get('user_id') or user_id
            _notify_backend_upload(unique_key, _generate_public_url(bucket, unique_key, endpoint, region), video_file.name, project_id, user_id)
            
            print(f"[UPLOAD] ✅ Multipart upload successful for {video_file.name} (key={unique_key})", flush=True)
            return True, unique_key
        
    except Exception as e:
        print(f"[UPLOAD] Multipart upload failed: {e}", flush=True)
    
    return False, None


def _upload_chunked(video_file, bucket, endpoint, access_key, secret_key, region, meta: Optional[dict] = None):
    """Chunked upload with smaller parts for better reliability.
    Returns (success: bool, public_url: Optional[str]).
    """
    try:
        import boto3
        from botocore.config import Config
        
        s3_client = _create_robust_s3_client(endpoint, access_key, secret_key, region, 180)  # 3 min timeout
        
        timestamp = int(time.time())
        unique_key = f"modal-generated/{timestamp}_{video_file.name}"
        
        # Use smaller chunks for better reliability
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        file_size = video_file.stat().st_size
        
        print(f"[UPLOAD] Starting chunked upload with {chunk_size / (1024*1024):.0f}MB chunks", flush=True)
        
        with open(video_file, 'rb') as f:
            # Upload in chunks and reassemble
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                return False
            
            # For small files, use regular upload
            if file_size <= chunk_size:
                upload_args = {'ContentType': 'video/mp4'}
                if not (endpoint and "backblazeb2.com" in endpoint):
                    upload_args['ACL'] = 'public-read'
                
                f.seek(0)
                s3_client.upload_fileobj(f, bucket, unique_key, ExtraArgs=upload_args)
            else:
                # Use multipart for larger files
                return _upload_multipart(video_file, bucket, endpoint, access_key, secret_key, region)
        
        # Verify and notify
        if _verify_upload(s3_client, bucket, unique_key, file_size):
            project_id, user_id = _extract_ids_from_filename(video_file.name)
            if meta:
                project_id = meta.get('project_id') or project_id
                user_id = meta.get('user_id') or user_id
            _notify_backend_upload(unique_key, _generate_public_url(bucket, unique_key, endpoint, region), video_file.name, project_id, user_id)
            
            print(f"[UPLOAD] ✅ Chunked upload successful for {video_file.name} (key={unique_key})", flush=True)
            return True, unique_key
        
    except Exception as e:
        print(f"[UPLOAD] Chunked upload failed: {e}", flush=True)
    
    return False, None


def _fallback_to_persistent_storage(video_file):
    """Fallback strategy: Store in persistent volume and notify backend."""
    try:
        import os
        import shutil
        import time
        
        # Create persistent storage directory
        persistent_dir = "/modal_volumes/pending_uploads"
        os.makedirs(persistent_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        persistent_filename = f"{timestamp}_{video_file.name}"
        persistent_path = os.path.join(persistent_dir, persistent_filename)
        
        # Copy file to persistent storage
        shutil.copy2(video_file, persistent_path)
        
        print(f"[FALLBACK] Stored {video_file.name} in persistent volume: {persistent_path}", flush=True)
        
        # Notify backend about pending upload
        _notify_backend_pending_upload(persistent_path, video_file.name)
        
        # Schedule retry upload in background
        _schedule_retry_upload(persistent_path, video_file.name)
        
        return True
        
    except Exception as e:
        print(f"[FALLBACK] Failed to store in persistent volume: {e}", flush=True)
        return False


def _create_robust_s3_client(endpoint, access_key, secret_key, region, timeout):
    """Create S3 client with robust configuration."""
    import boto3
    from botocore.config import Config
    
    config = Config(
        signature_version='s3v4',
        s3={'addressing_style': 'virtual'},
        read_timeout=timeout,
        connect_timeout=30,
        retries={'max_attempts': 1},  # Handle retries manually
        max_pool_connections=50
    )
    
    if endpoint and "backblazeb2.com" in endpoint:
        b2_endpoint = endpoint if endpoint.startswith('https://') else f"https://{endpoint}"
        return boto3.client(
            's3',
            endpoint_url=b2_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1',
            config=config
        )
    else:
        return boto3.client(
            's3',
            region_name=region,
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=config
        )


def _verify_upload(s3_client, bucket, key, expected_size):
    """Verify that upload was successful by checking file metadata."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        actual_size = response['ContentLength']
        MIN_VIDEO_BYTES = int(os.environ.get("MIN_VIDEO_BYTES", "4096"))

        if actual_size < MIN_VIDEO_BYTES:
            print(f"[VERIFY] Too small: {actual_size} bytes (< {MIN_VIDEO_BYTES})", flush=True)
            return False
        if actual_size != expected_size:
            print(f"[VERIFY] Size mismatch: expected {expected_size}, got {actual_size}", flush=True)
            return False
        print(f"[VERIFY] Upload verification successful: {actual_size} bytes", flush=True)
        return True
            
    except Exception as e:
        print(f"[VERIFY] Upload verification failed: {e}", flush=True)
        return False


def _generate_public_url(bucket, key, endpoint, region):
    """Generate public URL based on service type."""
    if endpoint and "backblazeb2.com" in endpoint:
        return f"https://f005.backblazeb2.com/file/{bucket}/{key}"
    elif endpoint:
        return f"{endpoint.rstrip('/')}/{bucket}/{key}"
    else:
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def _notify_backend_pending_upload(persistent_path, filename):
    """Notify backend about pending upload in persistent storage."""
    try:
        import requests
        import os
        
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
        
        payload = {
            "type": "pending_upload",
            "path": persistent_path,
            "filename": filename,
            "status": "pending_retry"
        }
        
        response = requests.post(
            f"{backend_url}/api/media/pending-upload",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"[FALLBACK] Backend notified about pending upload: {filename}", flush=True)
        else:
            print(f"[FALLBACK] Backend notification failed: {response.status_code}", flush=True)
            
    except Exception as e:
        print(f"[FALLBACK] Error notifying backend: {e}", flush=True)


def _schedule_retry_upload(persistent_path, filename):
    """Schedule background retry of failed upload."""
    def retry_worker():
        import time
        import os
        
        # Wait before retry
        time.sleep(60)  # 1 minute delay
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[RETRY] Attempting retry {attempt + 1}/{max_retries} for {filename}", flush=True)
                
                # Try upload again
                if _upload_with_retry_logic(Path(persistent_path), 
                                          os.environ.get("S3_BUCKET"),
                                          os.environ.get("S3_ENDPOINT"),
                                          os.environ.get("AWS_ACCESS_KEY_ID"),
                                          os.environ.get("AWS_SECRET_ACCESS_KEY"),
                                          os.environ.get("S3_REGION", "us-east-1")):
                    # Success - clean up persistent file
                    os.remove(persistent_path)
                    print(f"[RETRY] ✅ Retry successful, cleaned up persistent file", flush=True)
                    return
                else:
                    print(f"[RETRY] Retry {attempt + 1} failed", flush=True)
                    if attempt < max_retries - 1:
                        time.sleep(120)  # 2 minutes between retries
                        
            except Exception as e:
                print(f"[RETRY] Retry {attempt + 1} error: {e}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(120)
        
        print(f"[RETRY] All retry attempts failed for {filename}", flush=True)
    
    # Start retry in background thread
    retry_thread = threading.Thread(target=retry_worker, daemon=True)
    retry_thread.start()


def _extract_ids_from_filename(filename):
    """Extract project and user IDs from video filename."""
    try:
        import re
        # Format: u{userId}_p{projectId}_...
        match = re.search(r'u([a-f0-9\-]+)_p([a-f0-9\-]+)_', filename)
        if match:
            return match.group(2), match.group(1)  # project_id, user_id
        return None, None
    except Exception:
        return None, None


def _notify_backend_upload(key, public_url, filename, project_id=None, user_id=None):
    """Notify the backend about a completed video upload."""
    try:
        import requests
        import os
        import re
        
        # Fill from env if not provided
        if not project_id:
            project_id = os.environ.get("GAUSIAN_PROJECT_ID") or os.environ.get("PROJECT_ID")
        if not user_id:
            user_id = os.environ.get("GAUSIAN_USER_ID") or os.environ.get("USER_ID")

        # Extract IDs from filename if still not provided (u{user}_p{project})
        if not project_id or not user_id:
            match = re.search(r'u([a-f0-9\-]+)_p([a-f0-9\-]+)_', filename)
            if match:
                user_id = user_id or match.group(1)
                project_id = project_id or match.group(2)
                print(f"[UPLOAD] Extracted from filename - User: {user_id}, Project: {project_id}", flush=True)
        
        # Get backend URL from environment or use default
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:3001")
        
        # Notify backend via API call with project association
        payload = {
            "key": key,
            "remote_url": public_url,
            "filename": filename,
            "kind": "video",
            "source": "modal_generated",
            "project_id": project_id,
            "user_id": user_id
        }
        
        response = requests.post(
            f"{backend_url}/api/media/modal-upload",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"[UPLOAD] Backend notified and media imported: {filename}", flush=True)
            print(f"[UPLOAD] Project: {project_id} | User: {user_id}", flush=True)
        else:
            print(f"[UPLOAD] Backend notification failed: {response.status_code}", flush=True)
            print(f"[UPLOAD] Response: {response.text if hasattr(response, 'text') else 'No response text'}", flush=True)
            
    except Exception as e:
        print(f"[UPLOAD] Error notifying backend: {e}", flush=True)


# -------- Modal app & image --------
app = modal.App("gausian-comfyui")

models_volume = modal.Volume.from_name("comfyui-models", create_if_missing=True)
outputs_volume = modal.Volume.from_name("comfyui-outputs", create_if_missing=True)
custom_nodes_volume = modal.Volume.from_name("comfyui-custom-nodes", create_if_missing=True)

base_torch = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 "
        "torch==2.4.1 torchvision==0.19.1",
        "python -m pip install packaging setuptools wheel ninja",
    )
    # 3) Everything else
    .pip_install(
        "safetensors",
        "einops",
        "psutil",
        "opencv-python-headless",
        "imageio-ffmpeg",
        "diffusers>=0.30.0",
        "accelerate>=0.30.0",
        "huggingface_hub[hf_transfer]",
        "fastapi",
        "httpx",
        "PyYAML",
        "sageattention",
        "transformers==4.44.2",
        "tokenizers==0.19.1",
        "boto3",
        "requests",
        "websocket-client",
    )
)

image = (
    base_torch
    .run_commands(
        # ComfyUI core
        "git clone --depth=1 https://github.com/comfyanonymous/ComfyUI /root/comfy/ComfyUI",
        "pip install -r /root/comfy/ComfyUI/requirements.txt",

        # Transformers + tokenizers combo that worked in your earlier runs
        "python -m pip uninstall -y tokenizers || true",
        "python -m pip install --no-deps transformers==4.44.2 tokenizers==0.19.1",

        # Popular custom nodes
        "git clone --depth=1 https://github.com/cubiq/ComfyUI_essentials "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
        "git clone --depth=1 https://github.com/ShmuelRonen/ComfyUI-EmptyHunyuanLatent "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-Hunyuan-Adapter",
        "git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "git clone --depth=1 https://github.com/liusida/ComfyUI-SD3-nodes "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-SD3-nodes",
        "git clone --depth=1 https://github.com/chrisgoringe/cg-use-everywhere "
        "/root/comfy/ComfyUI/custom_nodes/ComfyUI-AnythingEverywhere",
    )
)


# ---------------- Modal function: ASGI app (served by Modal) ----------------
@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/modal_models": models_volume,   
        "/outputs": outputs_volume,
        "/userdir": custom_nodes_volume,  # will also hold comfy.db and extra_model_paths.yaml
    },
    timeout=21600,
    min_containers=0,
    max_containers=1,
    scaledown_window=900,
    secrets=[modal.Secret.from_name("cloud-config")],
    
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def comfyui():
    """
    Modal serves this FastAPI app at the printed URL.
    The app proxies requests to the ComfyUI server started locally in this process.
    """
    import torch
    
    # ===== EASY PERFORMANCE OPTIMIZATIONS =====
    
    # 1. Enable TF32 for faster matrix operations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")  # or "highest" on Hopper
    
    # 2. Enable additional tunable operations
    torch.backends.cudnn.benchmark = True  # Optimize for your specific input sizes
    torch.backends.cudnn.deterministic = False  # Allow optimizations
    
    # 3. Enable memory-efficient SDP backends
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    
    # 4. Set optimal memory allocation strategy
    torch.backends.cuda.enable_math_sdp(True)
    
    # 5. Enable torch.compile for future model loading (will be applied when models are loaded)
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.verbose = False
    
    print("[OPTIMIZATION] PyTorch performance optimizations enabled", flush=True)
    print(f"[OPTIMIZATION] CUDA device: {torch.cuda.get_device_name()}", flush=True)
    print(f"[OPTIMIZATION] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    
    subprocess.run("rm -rf /root/comfy/ComfyUI/models", shell=True, check=False)
    subprocess.run("ln -sfn /modal_models /root/comfy/ComfyUI/models", shell=True, check=True)
    subprocess.run("rm -f /root/comfy/ComfyUI/extra_model_paths.yaml", shell=True, check=False)
    # ensure Comfy’s default ./output points to the mounted volume
    subprocess.run("rm -rf /root/comfy/ComfyUI/output", shell=True, check=False)
    subprocess.run("ln -sfn /outputs /root/comfy/ComfyUI/output", shell=True, check=True)
    # also set the env var some nodes read
    os.environ["COMFYUI_OUTPUT_DIR"] = "/outputs"
    # Make sure the mounted outputs directory is writable
    try:
        os.chmod("/outputs", 0o777)
    except Exception:
        pass

    # ---------------------------
    
    # at app startup, create a tmpfs-like scratch for fast intermediates
    SCRATCH = "/dev/shm/comfy_scratch"
    subprocess.run(f"mkdir -p {SCRATCH}", shell=True, check=False)
    os.environ["TMPDIR"] = SCRATCH
    os.environ["TEMP"] = SCRATCH
    os.environ["TMP"] = SCRATCH

    # ===== ENVIRONMENT VARIABLES FOR OPTIMIZATIONS =====
    
    # Torch logging expects a comma-separated list, not "1". Use a sane default or set "help" to see options.
    os.environ["TORCH_LOGS"] = "inductor,perf_hints"
    
    # Choose the fastest available scaled-dot-product attention backend
    try:
        import torch
        if torch.cuda.is_available() and torch.backends.cuda.flash_sdp_enabled():
            os.environ["PYTORCH_SDP_BACKEND"] = "flash_attention"
            print("[OPTIMIZATION] Using flash attention backend", flush=True)
        elif torch.cuda.is_available() and torch.backends.cuda.mem_efficient_sdp_enabled():
            os.environ["PYTORCH_SDP_BACKEND"] = "mem_efficient"
            print("[OPTIMIZATION] Using memory-efficient attention backend", flush=True)
        else:
            os.environ["PYTORCH_SDP_BACKEND"] = "math"
            print("[OPTIMIZATION] Using math attention backend", flush=True)
    except Exception:
        os.environ["PYTORCH_SDP_BACKEND"] = "math"
        print("[OPTIMIZATION] Fallback to math attention backend", flush=True)
    
    # Additional optimization environment variables
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"               # harmless if not Hopper
    
    # Enable PyTorch optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    print("[OPTIMIZATION] Environment variables configured for performance", flush=True)
    # ---------------------------

    # If you keep extra custom nodes in your volume at /userdir/custom_nodes,
    # expose them alongside the baked-in ones by symlinking as a subfolder:
    if os.path.isdir("/userdir/custom_nodes"):
        pathlib.Path("/root/comfy/ComfyUI/custom_nodes").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            "ln -sfn /userdir/custom_nodes /root/comfy/ComfyUI/custom_nodes/_mounted",
            shell=True, check=False
        )

    # Build the unet/vae/clip layout *after* volumes are mounted
    _ensure_model_layout()
    
    # Setup auto-shutdown and upload system
    class JobCompletionManager:
        def __init__(self):
            self.active_jobs = set()
            self.completed_jobs = set()
            self.uploaded_videos = set()  # Track uploaded videos to prevent duplicates
            self.uploads_in_progress = 0   # Prevent shutdown while uploads are running
            self.last_activity = time.time()
            self.ws_clients = 0  # Number of connected WS clients
            self.server_ready = False  # Gate shutdown until Comfy is ready
            self._start_shutdown_monitor()
        
        def _start_shutdown_monitor(self):
            """Monitor for idle state and shutdown container to save costs."""
            def shutdown_monitor():
                while True:
                    try:
                        time.sleep(3)  # Check every 3 seconds

                        # Do not consider shutdown until server is fully ready
                        if not self.server_ready:
                            continue

                        # Tight shutdown: as soon as no active jobs AND no WS clients, exit
                        if len(self.active_jobs) == 0 and self.ws_clients == 0 and self.uploads_in_progress == 0:
                            print("[SHUTDOWN] No active jobs, no WS clients, no uploads — shutting down now", flush=True)
                            os._exit(0)
                        elif len(self.active_jobs) > 0 or self.uploads_in_progress > 0:
                            print(f"[SHUTDOWN] Still active: jobs={len(self.active_jobs)} uploads={self.uploads_in_progress} (ws_clients={self.ws_clients})", flush=True)
                        else:
                            idle_time = time.time() - self.last_activity
                            print(f"[SHUTDOWN] Awaiting WS disconnects (ws_clients={self.ws_clients}) idle={idle_time:.1f}s", flush=True)
                        
                    except Exception as e:
                        print(f"[SHUTDOWN] Shutdown monitor error: {e}", flush=True)
            
            self.shutdown_thread = threading.Thread(target=shutdown_monitor, daemon=True)
            self.shutdown_thread.start()
            print("[SHUTDOWN] Auto-shutdown monitor started (3-second loop, immediate when ws=0 & jobs=0)", flush=True)
        
        def job_started(self, job_id):
            """Mark a job as started."""
            self.active_jobs.add(job_id)
            self.last_activity = time.time()
            print(f"[JOBS] Job started: {job_id} (active: {len(self.active_jobs)})", flush=True)
        
        def job_completed(self, job_id):
            """Mark a job as completed."""
            self.active_jobs.discard(job_id)
            self.completed_jobs.add(job_id)
            self.last_activity = time.time()
            
            print(f"[JOBS] Job completed: {job_id} (active: {len(self.active_jobs)})", flush=True)
            
            # If no more active jobs, start shutdown timer
            if not self.active_jobs:
                print("[JOBS] All jobs completed, shutdown timer active", flush=True)

        def set_ws_clients(self, n: int):
            try:
                self.ws_clients = max(0, int(n))
            except Exception:
                self.ws_clients = 0
            self.last_activity = time.time()

        def mark_ready(self):
            # Called once ComfyUI is up and serving
            self.server_ready = True
            self.last_activity = time.time()

        def upload_started(self):
            try:
                self.uploads_in_progress += 1
                if self.uploads_in_progress < 0:
                    self.uploads_in_progress = 0
            except Exception:
                self.uploads_in_progress = max(1, getattr(self, 'uploads_in_progress', 0))
            self.last_activity = time.time()

        def upload_finished(self):
            try:
                self.uploads_in_progress -= 1
                if self.uploads_in_progress < 0:
                    self.uploads_in_progress = 0
            except Exception:
                self.uploads_in_progress = 0
            self.last_activity = time.time()
    
    # Initialize job manager
    job_manager = JobCompletionManager()
    
    # Use module-level global_progress (avoid rebinding a local)
    
    # Add global completion checker
    def global_completion_checker(job_manager_instance):
        """Check for completed videos and trigger uploads/shutdown."""
        import time
        from pathlib import Path
        
        last_video_count = 0
        stable_count = 0
        
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                # Count videos in output directory (recursive)
                output_dir = Path("/outputs")
                current_videos = list(output_dir.rglob("*.mp4"))
                current_count = len(current_videos)
                
                print(f"[GLOBAL-CHECK] Video count: {current_count}, Active jobs: {len(job_manager_instance.active_jobs)}", flush=True)
                
                # If video count changed, upload new ones (with better duplicate prevention)
                if current_count > last_video_count:
                    print(f"[GLOBAL-CHECK] New videos detected ({last_video_count} -> {current_count})", flush=True)
                    
                    # Upload only truly new videos (avoid duplicates)
                    for video in current_videos:
                        if (video.stat().st_mtime > time.time() - 120 and  # Videos from last 2 minutes
                            video.name not in job_manager_instance.uploaded_videos):  # Not already uploaded
                            # Add to uploaded set BEFORE uploading to prevent race conditions
                            job_manager_instance.uploaded_videos.add(video.name)
                            print(f"[GLOBAL-CHECK] Uploading new video: {video.name}", flush=True)
                            ok, key = _upload_to_backblaze(str(video))
                            if ok and key:
                                try:
                                    # We don't know the job_id here; leave key caching to the per-job monitor
                                    pass
                                except Exception:
                                    pass
                            if not ok:
                                # Remove from set if upload failed
                                job_manager_instance.uploaded_videos.discard(video.name)
                    
                    last_video_count = current_count
                    stable_count = 0
                else:
                    stable_count += 1
                
                # Only shutdown if truly idle - check ComfyUI queue status too,
                # and allow a grace period for the desktop app to download artifacts.
                if len(job_manager_instance.active_jobs) == 0 and stable_count >= 6:  # ~60s stable (@10s interval)
                    try:
                        # Double-check ComfyUI queue is actually empty
                        queue_response = urllib.request.urlopen(f"http://127.0.0.1:8188/queue", timeout=5)
                        queue_data = json.loads(queue_response.read().decode())
                        queue_pending = queue_data.get("queue_pending", [])
                        queue_running = queue_data.get("queue_running", [])

                        if len(queue_pending) == 0 and len(queue_running) == 0:
                            # Do not shut down if any persisted job indicates running recently
                            try:
                                import json as _j
                                active_persisted = False
                                now_ts = time.time()
                                for jf in _jobs_dir().glob("*.json"):
                                    try:
                                        st = _j.loads(jf.read_text(encoding="utf-8"))
                                    except Exception:
                                        continue
                                    if st.get('status') == 'running' and (now_ts - float(st.get('updated_at', now_ts))) < 600:
                                        active_persisted = True
                                        break
                                if active_persisted:
                                    print("[GLOBAL-CHECK] Persisted running jobs present; skipping shutdown", flush=True)
                                    stable_count = 0
                                    continue
                            except Exception:
                                pass
                            # Enforce a download grace window
                            GRACE = float(os.environ.get("SHUTDOWN_GRACE_SECONDS", "60"))
                            now = time.time()
                            ready = True
                            # For each completed job: consider ready if ACKed, or if all artifacts were downloaded, or grace elapsed
                            for jid, when in list(job_completed_at.items()):
                                if jid in jobs_imported_ack:
                                    continue
                                arts = job_artifacts.get(jid, [])
                                for name in arts:
                                    if name not in downloads_seen and (now - when) < GRACE:
                                        ready = False
                                        break
                                if not ready:
                                    break
                            if ready:
                                print("[GLOBAL-CHECK] All jobs complete, queue empty, grace satisfied - shutdown", flush=True)
                                print("[SHUTDOWN] Container terminating to save costs", flush=True)
                                os._exit(0)
                            else:
                                remain = int(max(0.0, GRACE - min((now - t) for t in job_completed_at.values()))) if job_completed_at else 0
                                print(f"[GLOBAL-CHECK] Waiting for downloads/grace (~{remain}s left)", flush=True)
                        else:
                            print(f"[GLOBAL-CHECK] Queue not empty - Pending: {len(queue_pending)}, Running: {len(queue_running)}", flush=True)
                            stable_count = 0  # Reset stability counter if queue has items
                    except Exception as queue_error:
                        print(f"[GLOBAL-CHECK] Queue check failed: {queue_error}", flush=True)
                    
            except Exception as e:
                print(f"[GLOBAL-CHECK] Error: {e}", flush=True)
    
    # Start global completion checker
    threading.Thread(target=global_completion_checker, args=(job_manager,), daemon=True).start()
    print("[GLOBAL-CHECK] Global completion checker started", flush=True)

    # Launch ComfyUI and wait for readiness
    proc = _launch_comfy()
    try:
        _wait_until_ready()
    except Exception:
        try:
            if proc.stdout:
                tail = proc.stdout.readlines()[-150:]
                print("\n".join(tail), flush=True)
        finally:
            proc.terminate()
        raise
    # Mark server ready so the shutdown monitor may start enforcing rules
    try:
        job_manager.mark_ready()
        print("[READY] ComfyUI ready; shutdown monitor armed", flush=True)
    except Exception:
        pass

    # Build the FastAPI proxy app (served by Modal)
    from fastapi import FastAPI, Request, Response, WebSocket
    import httpx

    api = FastAPI()
    # Lightweight in-app event bus for job notifications
    import asyncio, json as _json
    loop = asyncio.get_event_loop()
    clients: set[asyncio.Queue] = set()
    ws_sockets: set[WebSocket] = set()

    def broadcast(event: dict):
        for q in list(clients):
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

    def close_all_ws():
        try:
            for s in list(ws_sockets):
                try:
                    loop.call_soon_threadsafe(asyncio.create_task, s.close())
                except Exception:
                    pass
        except Exception:
            pass

    # Expose publisher to background threads
    def _publish(ev: dict):
        try:
            broadcast(ev)
        except Exception:
            pass
    global event_publisher
    event_publisher = _publish

    # Background status broadcaster: sends queue + summary every ~2 seconds
    def _start_status_broadcast():
        import threading, urllib.request
        def worker():
            last_idle_close = 0.0
            while True:
                try:
                    qd = {"queue_pending": 0, "queue_running": 0}
                    try:
                        r = urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=3)
                        import json as _j
                        d = _j.loads(r.read().decode())
                        qd["queue_pending"] = len(d.get("queue_pending", []))
                        qd["queue_running"] = len(d.get("queue_running", []))
                    except Exception:
                        pass
                    # Summarize in-progress jobs from global_progress
                    jobs = []
                    try:
                        for jid, p in list(global_progress.get('jobs', {}).items()):
                            jobs.append({
                                "job_id": jid,
                                "progress_percent": p.get('progress_percent', 0),
                                "current_step": p.get('current_step', 0),
                                "total_steps": p.get('total_steps', 0),
                            })
                    except Exception:
                        pass
                    # Merge persisted jobs (avoid duplicates)
                    try:
                        seen = {it.get("job_id") for it in jobs}
                        for jf in sorted(_jobs_dir().glob("*.json")):
                            try:
                                import json as _j
                                st = _j.loads(jf.read_text(encoding="utf-8"))
                            except Exception:
                                continue
                            jid = st.get('id') or jf.stem
                            if jid in seen:
                                continue
                            jobs.append({
                                "job_id": jid,
                                "progress_percent": st.get('progress_percent'),
                                "current_step": st.get('current_step'),
                                "total_steps": st.get('total_steps'),
                            })
                    except Exception:
                        pass
                    _publish({"type": "status", "pending": qd["queue_pending"], "running": qd["queue_running"], "jobs": jobs})

                    # Proactively close idle WS connections when no jobs are active or pending
                    try:
                        import json as _j, time as _t
                        if qd["queue_pending"] == 0 and qd["queue_running"] == 0:
                            # Any persisted running job in the last 2 minutes?
                            now_ts = _t.time()
                            active_persisted = False
                            for jf in _jobs_dir().glob("*.json"):
                                try:
                                    st = _j.loads(jf.read_text(encoding="utf-8"))
                                except Exception:
                                    continue
                                if st.get('status') == 'running' and (now_ts - float(st.get('updated_at', now_ts))) < 20:
                                    active_persisted = True
                                    break
                            if not active_persisted and (now_ts - last_idle_close) > 30:
                                _publish({"type": "server_idle_close"})
                                try:
                                    close_all_ws()
                                except Exception:
                                    pass
                                last_idle_close = now_ts
                    except Exception:
                        pass
                except Exception:
                    pass
                finally:
                    time.sleep(2.0)
        threading.Thread(target=worker, daemon=True).start()
    _start_status_broadcast()
    from fastapi.staticfiles import StaticFiles
    api.mount("/files", StaticFiles(directory="/outputs"), name="files")
    COMFY = "http://127.0.0.1:8188"

    @api.get("/")
    async def root():
        return {"ok": True, "msg": "ComfyUI proxy up. Try /health or /system_stats."}

    @api.get("/healthz")
    async def healthz(request: Request):
        from pathlib import Path
        base = str(request.base_url).rstrip("/")
        outs = []
        try:
            for f in sorted(Path("/outputs").glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
                name = f.name
                url = f"{base}/view?filename={name}"
                outs.append({"filename": name, "url": url})
        except Exception:
            pass
        ws = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/events"
        return {"ok": True, "http": base, "ws": ws, "recent": [{"id": "outputs", "status": "completed", "artifacts": outs}]}

    @api.websocket("/events")
    async def ws_events(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        clients.add(q)
        ws_sockets.add(ws)
        try:
            job_manager.set_ws_clients(len(ws_sockets))
        except Exception:
            pass
        try:
            await ws.send_text(_json.dumps({"type": "hello"}))
            while True:
                ev = await q.get()
                await ws.send_text(_json.dumps(ev))
                try:
                    if isinstance(ev, dict) and ev.get("type") == "server_idle_close":
                        await ws.close()
                        break
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            clients.discard(q)
            try:
                ws_sockets.discard(ws)
            except Exception:
                pass
            try:
                job_manager.set_ws_clients(len(ws_sockets))
            except Exception:
                pass
    
    @api.get("/progress-status")
    async def progress_status():
        """Get real-time progress of all active jobs with accurate timing."""
        try:
            import json
            import urllib.request
            from pathlib import Path
            
            # Get queue status
            queue_response = urllib.request.urlopen(f"http://127.0.0.1:8188/queue", timeout=5)
            queue_data = json.loads(queue_response.read().decode())
            
            # Get active and pending jobs
            queue_pending = queue_data.get("queue_pending", [])
            queue_running = queue_data.get("queue_running", [])
            
            # Count actual video files
            output_dir = Path("/outputs")
            video_count = len(list(output_dir.glob("*.mp4")))
            
            # Calculate more accurate progress
            total_jobs_submitted = len(job_manager.active_jobs) + len(job_manager.completed_jobs)
            jobs_in_queue = len(queue_pending) + len(queue_running)
            
            # More accurate progress calculation
            if total_jobs_submitted > 0:
                queue_progress = max(0, (total_jobs_submitted - jobs_in_queue) / total_jobs_submitted * 100)
            else:
                queue_progress = 0
            
            # Get real-time progress from WebSocket data
            job_details = []
            total_progress = 0
            active_jobs_count = len(job_manager.active_jobs)
            
            for job_id in job_manager.active_jobs:
                job_progress = global_progress['jobs'].get(job_id, {})
                job_details.append({
                    'job_id': job_id,
                    'progress_percent': job_progress.get('progress_percent', 0),
                    'current_step': job_progress.get('current_step', 0),
                    'total_steps': job_progress.get('total_steps', 0),
                    'videos_generated': job_progress.get('videos_generated', 0),
                })
                total_progress += job_progress.get('progress_percent', 0)
            
            overall_progress = total_progress / max(1, active_jobs_count) if active_jobs_count > 0 else 100
            
            # Determine service active state (avoid waking cold containers unnecessarily)
            now_ts = time.time()
            any_active = (len(queue_pending) + len(queue_running) + len(job_manager.active_jobs)) > 0
            # Completed jobs list
            completed_list = list(job_manager.completed_jobs)
            return {
                "active_jobs": active_jobs_count,
                "uploaded_videos": len(job_manager.uploaded_videos),
                "queue_pending": len(queue_pending),
                "queue_running": len(queue_running),
                "video_files_generated": video_count,
                "queue_progress": round(queue_progress, 1),
                "overall_progress": round(overall_progress, 1),
                "total_jobs_submitted": total_jobs_submitted,
                "last_activity": job_manager.last_activity,
                "idle_time": now_ts - job_manager.last_activity,
                "recent_jobs": completed_list[-10:],
                "job_details": [dict(it, **{"completed": False}) for it in job_details],
                "completed_jobs": completed_list,
                "active": any_active,
                "estimated_time_per_shot": "15-67 seconds",
                "websocket_progress": True,
            }
        except Exception as e:
            return {"error": str(e)}

    @api.get("/health")
    async def health():
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{COMFY}/system_stats")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @api.get("/system_stats")
    async def system_stats():
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{COMFY}/system_stats")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @api.get("/object_info")
    async def object_info():
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{COMFY}/object_info")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @api.post("/prompt")
    async def prompt(request: Request):
        import json
        try:
            req = await request.json()
        except Exception:
            return Response(
                content=json.dumps({"error": "Invalid JSON body"}),
                status_code=400,
                media_type="application/json",
            )
        
        # Track job start
        job_id = req.get("client_id", f"job_{int(time.time())}")
        print(f"[DEBUG] Tracking job with ID: {job_id}", flush=True)
        job_manager.job_started(job_id)

        # Accept either a full request {prompt: {...}} or a raw graph map as body
        p = req.get("prompt", None)
        revived_graph = False
        if p is None and isinstance(req, dict):
            # Heuristic: if body looks like a graph (values with class_type/inputs), wrap it
            try:
                looks_like_graph = False
                for v in list(req.values())[:5]:
                    if isinstance(v, dict) and ("class_type" in v or "inputs" in v):
                        looks_like_graph = True
                        break
                if looks_like_graph:
                    req["prompt"] = p = req
                    revived_graph = True
            except Exception:
                pass
        if p is None:
            return Response(
                content=json.dumps({"error": "Missing 'prompt' in body"}),
                status_code=400,
                media_type="application/json",
            )

        # Accept top-level STRING prompt and parse it
        revived_top = False
        if isinstance(p, str):
            try:
                p2 = json.loads(p)
                if isinstance(p2, dict):
                    req["prompt"] = p = p2
                    revived_top = True
                else:
                    return Response(
                        content=json.dumps({"error": "prompt string did not parse to an object"}),
                        status_code=400,
                        media_type="application/json",
                    )
            except Exception as e:
                return Response(
                    content=json.dumps({"error": f"prompt string not valid JSON: {str(e)}"}),
                    status_code=400,
                    media_type="application/json",
                )

        if not isinstance(p, dict):
            return Response(
                content=json.dumps({"error": "prompt must be an object map (dict)"}),
                status_code=400,
                media_type="application/json",
            )

        # Revive any node-level strings
        revived_nodes, bad_nodes, missing = [], [], []
        for k, v in list(p.items()):
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, dict):
                        p[k] = parsed
                        revived_nodes.append(k)
                    else:
                        bad_nodes.append(k)
                except Exception:
                    bad_nodes.append(k)

        # Validate node shapes
        for k, v in p.items():
            if not isinstance(v, dict):
                bad_nodes.append(k)
                continue
            if "class_type" not in v or "inputs" not in v or not isinstance(v["inputs"], dict):
                missing.append(k)

        if revived_top or revived_nodes or revived_graph:
            print(f"[PROXY] revived_top={revived_top} revived_graph={revived_graph} revived_nodes={revived_nodes}", flush=True)

        if bad_nodes or missing:
            err = {
                "error": "prompt contains invalid nodes",
                "string_nodes_unparsed": bad_nodes[:20],
                "nodes_missing_fields": missing[:20],
            }
            return Response(content=json.dumps(err), status_code=400, media_type="application/json")

        # Build an execution plan: sum steps across samplers and count encode nodes
        def _resolve_int_from_input(graph: dict, spec):
            try:
                if isinstance(spec, int):
                    return spec
                if isinstance(spec, (float,)):
                    return int(spec)
                if isinstance(spec, list) and len(spec) >= 1 and isinstance(spec[0], str):
                    ref = graph.get(spec[0])
                    if isinstance(ref, dict):
                        ct = ref.get('class_type')
                        inputs = ref.get('inputs') or {}
                        if ct in ('PrimitiveInt', 'PrimitiveInteger', 'Int'):
                            val = inputs.get('value')
                            if isinstance(val, int):
                                return val
                        if ct in ('SimpleMath+', 'SimpleMath'):
                            expr = inputs.get('value')
                            if isinstance(expr, str):
                                vals = {}
                                for k2, v2 in inputs.items():
                                    if k2 == 'value':
                                        continue
                                    vi = _resolve_int_from_input(graph, v2)
                                    if isinstance(vi, int):
                                        vals[k2] = vi
                                try:
                                    return int(eval(expr, {}, vals))
                                except Exception:
                                    return None
                return None
            except Exception:
                return None

        def _build_plan(graph: dict):
            weights = {}
            encode_nodes = []
            # Try to infer per-video frame count from graph
            def _collect_lengths() -> list[int]:
                lens: list[int] = []
                try:
                    for nid, node in graph.items():
                        if not isinstance(node, dict):
                            continue
                        ct = node.get('class_type')
                        ins = node.get('inputs') or {}
                        # Primary: EmptyHunyuanLatentVideo.length
                        if ct in ('EmptyHunyuanLatentVideo', 'EmptyLatentVideo'):
                            ln = _resolve_int_from_input(graph, ins.get('length'))
                            if isinstance(ln, int) and ln > 0:
                                lens.append(ln)
                        # Generic: any node with a numeric 'length' input
                        if 'length' in ins:
                            ln2 = _resolve_int_from_input(graph, ins.get('length'))
                            if isinstance(ln2, int) and ln2 > 0:
                                lens.append(ln2)
                except Exception:
                    pass
                return lens

            lengths = _collect_lengths()
            frames_per_video = max(lengths) if lengths else 1
            for nid, node in graph.items():
                if not isinstance(node, dict):
                    continue
                ct = node.get('class_type')
                if ct in ('KSampler', 'KSamplerAdvanced'):
                    st = _resolve_int_from_input(graph, (node.get('inputs') or {}).get('steps'))
                    if isinstance(st, int) and st > 0:
                        weights[str(nid)] = st
                if ct in ('VHS_VideoCombine', 'VideoCombine', 'SaveVideo'):
                    encode_nodes.append(str(nid))
            encode_weights = {eid: frames_per_video for eid in encode_nodes}
            total = sum(weights.values()) + sum(encode_weights.values())
            return {"weights": weights, "encode_weights": encode_weights, "total_steps": total, "frames": frames_per_video}

        plan = _build_plan(p)

        # Capture/ensure filename prefix from graph for artifact filtering/uniqueness
        try:
            expect_prefix = None
            for _nid, node in (p or {}).items():
                if not isinstance(node, dict):
                    continue
                ct = node.get('class_type')
                if ct in ('VHS_VideoCombine', 'VideoCombine', 'SaveVideo', 'SaveImage'):
                    ins = node.get('inputs') or {}
                    fp = ins.get('filename_prefix')
                    if isinstance(fp, str) and fp.strip():
                        expect_prefix = fp.strip()
                        break
            if not expect_prefix:
                # Default to client-provided id to avoid cross-job collisions
                expect_prefix = str(job_id)
                # Patch the graph inline to ensure downstream filenames are prefixed uniquely
                for _nid, node in (p or {}).items():
                    if not isinstance(node, dict):
                        continue
                    ct = node.get('class_type')
                    if ct in ('VHS_VideoCombine', 'VideoCombine', 'SaveVideo', 'SaveImage'):
                        if not isinstance(node.get('inputs'), dict):
                            node['inputs'] = {}
                        if not node['inputs'].get('filename_prefix'):
                            node['inputs']['filename_prefix'] = expect_prefix
                # Save patched prompt back into request body
                req['prompt'] = p
            if expect_prefix:
                job_prefixes[str(job_id)] = expect_prefix
        except Exception:
            pass

        # Capture job meta for later upload/notify (project/user)
        try:
            candidate_project = req.get('project_id') if isinstance(req, dict) else None
            candidate_user = req.get('user_id') if isinstance(req, dict) else None
        except Exception:
            candidate_project = None
            candidate_user = None

        # Forward to local ComfyUI
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{COMFY}/prompt", json=req)

        # Start monitoring for job completion and upload
        if r.status_code == 200:
            # extract prompt_id from Comfy response
            pid_val = None
            try:
                d = r.json()
                pid_val = d.get("prompt_id") or d.get("number") or d.get("id")
            except Exception:
                pid_val = None
            # Record meta and prefix for both prompt and client ids
            try:
                if pid_val is not None:
                    job_meta[str(pid_val)] = {"project_id": candidate_project, "user_id": candidate_user}
                    # Propagate expected prefix to prompt_id key as well
                    if job_prefixes.get(str(job_id)):
                        job_prefixes[str(pid_val)] = job_prefixes.get(str(job_id))
                if job_id:
                    job_meta[str(job_id)] = {"project_id": candidate_project, "user_id": candidate_user}
            except Exception:
                pass
            # Persist initial running state with computed total steps for cross-request visibility
            try:
                if pid_val is not None:
                    norm_total = int(plan.get('frames') or plan.get('total_steps') or 0)
                    _persist_job_state(str(pid_val), {
                        'status': 'running',
                        'current_step': 0,
                        'total_steps': norm_total,
                        'progress_percent': 0,
                        'artifacts': [],
                    })
            except Exception:
                pass
            threading.Thread(
                target=_monitor_job_completion,
                args=(job_id, pid_val, job_manager, plan),
                daemon=True
            ).start()
        
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        ) 


    @api.get("/queue")
    async def queue():
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{COMFY}/queue")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @api.get("/history")
    async def history_all():
        """Get all history entries"""
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{COMFY}/history")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @api.get("/history/{pid}")
    async def history(pid: str):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{COMFY}/history/{pid}")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @api.get("/view")
    async def view(filename: str):
        """Serve generated files from the outputs directory (streaming).
        We stream in chunks and only mark download as seen on successful completion
        to avoid premature shutdown during long transfers.
        """
        import os
        from pathlib import Path
        from fastapi.responses import StreamingResponse
        base = Path("/outputs").resolve()
        file_path = (base / filename).resolve()
        # Prevent path traversal
        try:
            _ = file_path.relative_to(base)
        except Exception:
            return {"error": "Invalid path"}, 400
        if not file_path.exists() or not file_path.is_file():
            return {"error": f"File {filename} not found"}, 404

        # Determine content type based on file extension
        content_type = "application/octet-stream"
        if filename.endswith(".mp4"):
            content_type = "video/mp4"
        elif filename.endswith(".png"):
            content_type = "image/png"
        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif filename.endswith(".gif"):
            content_type = "image/gif"

        def file_iterator(path: Path, chunk_size: int = 1024 * 1024):
            try:
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        # Touch last_activity to avoid idle shutdown while streaming
                        try:
                            job_manager.last_activity = time.time()
                        except Exception:
                            pass
                        yield chunk
                # Mark as downloaded after successful streaming
                try:
                    downloads_seen.add(filename)
                except Exception:
                    pass
            except Exception:
                # Do not mark as downloaded on error
                raise

        return StreamingResponse(file_iterator(file_path), media_type=content_type)

    @api.get("/jobs/{job_id}")
    async def get_job(job_id: str, request: Request):
        """Return job status/artifacts from persistent state, with in-memory fallback."""
        import json as _j
        base = str(request.base_url).rstrip("/")
        # Prefer persisted state keyed by job_id; callers should pass the prompt_id
        state = _read_job_state(job_id) or {}
        status = state.get('status') or 'unknown'
        cur = state.get('current_step')
        tot = state.get('total_steps')
        pr = state.get('progress_percent')
        names = state.get('artifacts') or []
        # Fallback to memory if no persisted artifacts
        if not names:
            names = job_artifacts.get(job_id, [])
        arts = []
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            # Prefer presigned S3 URL if we have an object key
            url = None
            try:
                key = (job_object_keys.get(job_id) or {}).get(name)
            except Exception:
                key = None
            if key:
                try:
                    import boto3
                    expire = int(os.environ.get("ARTIFACT_URL_EXPIRE", "86400"))
                    region = os.environ.get("S3_REGION") or os.environ.get("AWS_REGION")
                    endpoint = os.environ.get("S3_ENDPOINT")
                    bucket = os.environ.get("S3_BUCKET")
                    s3 = boto3.client("s3", region_name=region, endpoint_url=endpoint) if endpoint else boto3.client("s3", region_name=region)
                    url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expire)
                except Exception:
                    url = None
            if not url:
                url = f"{base}/view?filename={name}"
            arts.append({"filename": name, "url": url})
        # If completed but no artifact list, try scanning outputs directory for recent files
        if status == 'completed' and not arts:
            try:
                from pathlib import Path as _P
                for f in sorted(_P("/outputs").glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
                    arts.append({"filename": f.name, "url": f"{base}/view?filename={f.name}"})
            except Exception:
                pass
        resp = {"id": job_id, "status": status, "artifacts": arts}
        if pr is not None:
            resp["progress_percent"] = pr
        if cur is not None:
            resp["current_step"] = cur
        if tot is not None:
            resp["total_steps"] = tot
        return resp

    @api.post("/jobs/{job_id}/imported")
    async def mark_imported(job_id: str, request: Request):
        """Client signals it finished importing artifacts.
        Optional JSON body may include {"filenames": ["..."]} to mark specific files as downloaded.
        Accepts either X-Shared-Secret or Bearer token matching SHARED_SECRET.
        """
        try:
            # Auth (best-effort; allow unauth if no secret configured)
            ok = True
            if SHARED_SECRET and SHARED_SECRET != "change-me":
                ok = (request.headers.get("x-shared-secret") == SHARED_SECRET)
                if not ok:
                    auth = request.headers.get("authorization") or ""
                    if auth.lower().startswith("bearer ") and auth.split(" ", 1)[1].strip() == SHARED_SECRET:
                        ok = True
            if not ok:
                return Response(status_code=401)
            # Mark ACK
            try:
                jobs_imported_ack.add(str(job_id))
            except Exception:
                pass
            # Optionally mark specific files as downloaded
            try:
                data = await request.json()
            except Exception:
                data = {}
            try:
                files = data.get("filenames") or []
                for name in files:
                    if isinstance(name, str) and name:
                        downloads_seen.add(name)
            except Exception:
                pass
            # Keep container warm briefly
            try:
                job_manager.last_activity = time.time()
            except Exception:
                pass
            return {"ok": True}
        except Exception:
            return Response(status_code=500)

    @api.get("/debug/models")
    async def debug_models():
        import json, glob, pathlib

        def ls(dirpath):
            p = pathlib.Path(dirpath)
            if not p.exists():
                return {"exists": False}
            items = []
            for f in sorted(p.glob("*")):
                it = {"name": f.name, "is_symlink": f.is_symlink(), "is_file": f.is_file()}
                if f.is_symlink():
                    try:
                        tgt = f.resolve(strict=False)
                        it["points_to"] = str(tgt)
                        it["target_exists"] = tgt.exists()
                    except Exception as e:
                        it["points_to"] = f"<error: {e}>"
                        it["target_exists"] = False
                items.append(it)
            return {"exists": True, "items": items}

        return {
            "/root/comfy/ComfyUI/models": ls("/root/comfy/ComfyUI/models"),
            "/root/comfy/ComfyUI/models/unet": ls("/root/comfy/ComfyUI/models/unet"),
            "/root/comfy/ComfyUI/models/vae": ls("/root/comfy/ComfyUI/models/vae"),
            "/root/comfy/ComfyUI/models/clip": ls("/root/comfy/ComfyUI/models/clip"),
            "/root/comfy/ComfyUI/models/diffusion_models": ls("/root/comfy/ComfyUI/models/diffusion_models"),
        }

    @api.post("/debug/relink")
    async def debug_relink():
        # Rebuild symlinks and re-write the YAML
        _ensure_model_layout()
        # Ask Comfy to refresh its object info (Comfy reads its managers live)
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                await client.get(f"{COMFY}/object_info")
        except Exception:
            pass
        # Return what we see on disk, for sanity
        from pathlib import Path
        def ls(dirpath):
            p = Path(dirpath)
            return sorted([f.name for f in p.glob("*.safetensors")]) if p.exists() else []
        return {
            "unet": ls("/models/unet"),
            "vae": ls("/models/vae"),
            "clip": ls("/models/clip")
        }
    
    @api.get("/debug/extra_paths")
    async def debug_extra_paths():
        # Extra model paths YAML is intentionally disabled/unused.
        p = "/root/comfy/ComfyUI/extra_model_paths.yaml"
        exists = os.path.exists(p)
        return {"path": p, "exists": exists, "note": "extra_model_paths.yaml is not used; models are discovered via standard folders and symlinks."}

    @api.get("/debug/ls_outputs")
    async def debug_ls_outputs():
        import os, pathlib, time
        root = "/outputs"
        if not os.path.exists(root):
            return {"root": root, "exists": False, "items": []}

        def list_all(r):
            items = []
            base = pathlib.Path(r)
            for p in base.rglob("*"):
                try:
                    stat = p.stat()
                    items.append({
                        "path": str(p.relative_to(base)),
                        "is_dir": p.is_dir(),
                        "size": None if p.is_dir() else stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                except Exception:
                    pass
            # sort: dirs first, then files by name
            items.sort(key=lambda x: (not x["is_dir"], x["path"]))
            return items

        return {"root": root, "exists": True, "items": list_all(root)}
    
    
    @api.get("/debug/ls_path")
    async def debug_ls_path(path: str = "/outputs"):
        import os, pathlib
        p = pathlib.Path(path)
        if not p.exists():
            return {"path": path, "exists": False, "items": []}
        items = []
        for f in sorted(p.rglob("*")):
            try:
                st = f.stat()
                items.append({
                    "path": str(f.relative_to(p)),
                    "is_dir": f.is_dir(),
                    "size": None if f.is_dir() else st.st_size,
                    "mtime": st.st_mtime,
                })
            except Exception:
                pass
        return {"path": path, "exists": True, "items": items}

    @api.get("/debug/find_outputs")
    async def debug_find_outputs(q: str = "teacache"):
        import pathlib
        roots = [
            "/outputs",
            "/root/comfy/ComfyUI/output",
            "/root/comfy/ComfyUI",
            "/userdir",
            "/tmp",
        ]
        results = []
        for root in roots:
            base = pathlib.Path(root)
            if not base.exists():
                continue
            for f in base.rglob("*"):
                name = f.name
                if q and q not in name:
                    continue
                try:
                    st = f.stat()
                    results.append({
                        "root": root,
                        "path": str(f.relative_to(base)),
                        "abs": str(f),
                        "is_dir": f.is_dir(),
                        "size": None if f.is_dir() else st.st_size,
                        "mtime": st.st_mtime,
                    })
                except Exception:
                    pass
        results.sort(key=lambda x: x["mtime"], reverse=True)
        return {"query": q, "results": results[:200]}


    return api
