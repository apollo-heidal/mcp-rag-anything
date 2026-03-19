# Project Instructions

- Project deployment is done with Docker Compose, so all additions to the deployment or sidecar apps need to be provisioned using Docker Compose.

- Do not put fallback env vars for models in docker-compose.yml. Instead, add them to .env.example.

- Use native `docker compose` and standard Docker CLI commands for service lifecycle work in this repo. For builds, startup, shutdown, logs, stats, restarts, rebuilds, and verification, prefer direct Docker commands over custom wrappers.
- Treat Docker Desktop with Docker Model Runner enabled as the default local runtime for this project. If setup steps need to be documented or changed, update the README rather than introducing a new wrapper script.
- Use `uv run ...` for one-off local commands only, such as an isolated script, quick inspection, or a narrow verification step. Do not treat `uv run` as a substitute for container logs, container-based runtime validation, or repeatable integration checks for larger changes.

## MCP Curl Testing (Streamable HTTP)

### 1. Initialize session and get upload link
```bash
#!/usr/bin/env bash
set -euo pipefail

BASE=http://localhost:8000

# --- Initialize MCP session ---
curl -s -D /tmp/mcp_headers -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' \
  -o /dev/null

SID=$(grep -i mcp-session-id /tmp/mcp_headers | awk '{print $2}' | tr -d '\r\n')
echo "Session: $SID"

# --- Send initialized notification ---
curl -s -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' > /dev/null

# --- Create upload link (one-time grant token baked into the URL) ---
curl -s -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"create_upload_link","arguments":{"filename":"myfile.txt","collection_id":"default"}}}' \
  > /tmp/mcp_upload_resp.txt

UPLOAD_GRANT=$(grep -o 'api/upload/[^\\]*' /tmp/mcp_upload_resp.txt | head -1 | sed 's|api/upload/||')
echo "Upload grant: ${UPLOAD_GRANT:0:30}..."

# --- Upload a file (grant token in path, no other auth needed) ---
curl -f -X POST -F file=@myfile.txt "$BASE/api/upload/$UPLOAD_GRANT"
```

### 2. Get a UI session grant (for browsing files, checking status)
```bash
curl -s -X POST "$BASE/mcp" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"create_ui_session_link","arguments":{"requested_for":"curl-test","ttl_seconds":3600}}}' \
  > /tmp/mcp_ui_resp.txt

UI_GRANT=$(grep -o 'grant=[^\\]*' /tmp/mcp_ui_resp.txt | head -1 | sed 's|grant=||')
echo "UI grant: ${UI_GRANT:0:30}..."

# List files
curl -s "$BASE/api/files?grant=$UI_GRANT" | python3 -m json.tool

# Check file status (replace FILE_ID)
curl -s "$BASE/api/files/FILE_ID/status?grant=$UI_GRANT" | python3 -m json.tool

# Upload via UI session grant (alternative to one-time upload link)
curl -f -X POST -F file=@myfile.txt "$BASE/api/upload?grant=$UI_GRANT"
```

**Key notes:**
- Header is `Mcp-Session-Id` (not `Mcp-Session`)
- Must send `Accept: application/json, text/event-stream` on every request
- Response is SSE format: `event: message\ndata: {json}`
- Must send `notifications/initialized` before calling tools
- Two upload paths: `/api/upload/{grant_token}` (one-time upload link) or `/api/upload?grant=` (UI session)
- `/api/files` and `/api/files/{id}/status` require a UI session grant (`?grant=`)
