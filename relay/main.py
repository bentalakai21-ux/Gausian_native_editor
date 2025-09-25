import os
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="Relay", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory registry: tenant -> websockets
TENANT_SOCKETS: Dict[str, Set[WebSocket]] = {}

RELAY_WEBHOOK_TOKEN = os.getenv("RELAY_WEBHOOK_TOKEN", "")
ACCEPT_CLIENT_BEARER = os.getenv("ACCEPT_CLIENT_BEARER", "")


def ensure_webhook_auth(x_relay_token: Optional[str] = Header(None)):
    if RELAY_WEBHOOK_TOKEN and x_relay_token != RELAY_WEBHOOK_TOKEN:
        raise HTTPException(status_code=401, detail="invalid relay webhook token")


def ensure_client_auth(authorization: Optional[str]) -> bool:
    # If a token is configured, require exact match
    if ACCEPT_CLIENT_BEARER:
        return authorization == f"Bearer {ACCEPT_CLIENT_BEARER}"
    # Otherwise, accept any (for development)
    return True


class Artifact(BaseModel):
    url: str
    filename: Optional[str] = "output.bin"


class ModalEvent(BaseModel):
    # One of: status, progress, job_completed
    type: str
    tenant: str
    job_id: Optional[str] = None
    # progress
    progress_percent: Optional[float] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    node_id: Optional[str] = None
    # batch status
    pending: Optional[int] = None
    running: Optional[int] = None
    jobs: Optional[List[Dict[str, Any]]] = None
    # completion
    artifacts: Optional[List[Artifact]] = Field(default_factory=list)


async def broadcast(tenant: str, message: Dict[str, Any]):
    conns = TENANT_SOCKETS.get(tenant)
    if not conns:
        return
    dead: List[WebSocket] = []
    for ws in list(conns):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        conns.discard(ws)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/webhook/modal")
async def modal_webhook(evt: ModalEvent, _auth=Depends(ensure_webhook_auth)):
    # Normalize pass-through: desktop app already understands this shape
    await broadcast(evt.tenant, evt.model_dump())
    return {"ok": True}


@app.websocket("/stream")
async def stream(ws: WebSocket, tenant: str = Query(..., description="Tenant id")):
    # Optional client auth via Authorization header
    auth = ws.headers.get("authorization")
    if not ensure_client_auth(auth):
        await ws.close(code=4401)
        return
    await ws.accept()
    conns = TENANT_SOCKETS.setdefault(tenant, set())
    conns.add(ws)
    try:
        while True:
            # Keep-alive: we don't expect client messages, but receive to detect disconnect
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        conns.discard(ws)

