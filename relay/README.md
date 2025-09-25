Relay (Live Events) — Minimal Scaffold

Overview
- Accepts webhooks from your cloud jobs (Modal or your backend) for progress and completion
- Broadcasts normalized events to desktop clients over WebSocket
- Keeps cost low by being lightweight and stateless; easy to run on serverless/edge (Fly.io, Render, Railway, Cloud Run, etc.)

Endpoints
- POST /webhook/modal
  - Auth: header `X-Relay-Token: <secret>` (set env RELAY_WEBHOOK_TOKEN)
  - Body (JSON):
    - type: "status" | "progress" | "job_completed"
    - tenant: string (per user/org)
    - job_id: string
    - Optional: progress_percent, current_step, total_steps, node_id
    - Optional: artifacts: [{ url, filename }]

- WS /stream?tenant=<id>
  - Desktop connects here to receive events for a tenant
  - Auth: header `Authorization: Bearer <token>` (optional; set env ACCEPT_CLIENT_BEARER to enforce)

Run locally
```
cd relay
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export RELAY_WEBHOOK_TOKEN=dev-secret
# Optional: enforce desktop WS token matching this value
# export ACCEPT_CLIENT_BEARER=desktop-dev-token
uvicorn main:app --host 0.0.0.0 --port 8000
```

Configure desktop app
- In Cloud (Modal) panel, set "Relay WS URL" to:
  - `ws://localhost:8000/stream?tenant=<your-tenant-id>` for local testing
  - or `wss://relay.yourdomain.com/stream?tenant=<tenant>` in prod
- Desktop sends Authorization: Bearer <API Key> by default; if you set ACCEPT_CLIENT_BEARER, use that token instead.

Send a test webhook
```
curl -X POST http://localhost:8000/webhook/modal \
  -H 'Content-Type: application/json' \
  -H 'X-Relay-Token: dev-secret' \
  -d '{
    "type": "progress",
    "tenant": "t1",
    "job_id": "job-123",
    "progress_percent": 37.5,
    "current_step": 3,
    "total_steps": 10
  }'
```

Docker
```
docker build -t relay .
docker run -p 8000:8000 -e RELAY_WEBHOOK_TOKEN=dev-secret relay
```

Notes
- Stateless: in-memory connection registry only; for multi-instance deployments add Redis pub/sub.
- SSE is easy to add if needed; for now WS keeps things simple.

