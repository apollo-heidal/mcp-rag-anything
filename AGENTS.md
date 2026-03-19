# Project Instructions

## MANDATORY: Use td for Task Management

You must run td usage --new-session at conversation start (or after /clear) to see current work.
Use td usage -q for subsequent reads.

- Use native `docker compose` and standard Docker CLI commands for service lifecycle work in this repo. For builds, startup, shutdown, logs, stats, restarts, rebuilds, and verification, prefer direct Docker commands over custom wrappers.
- Treat Docker Desktop with Docker Model Runner enabled as the default local runtime for this project. If setup steps need to be documented or changed, update the README rather than introducing a new wrapper script.
- Use `uv run ...` for one-off local commands only, such as an isolated script, quick inspection, or a narrow verification step. Do not treat `uv run` as a substitute for container logs, container-based runtime validation, or repeatable integration checks for larger changes.

## MCP Curl Testing (Streamable HTTP)

### 1. Initialize session
```bash
HEADERS=$(curl -si -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}')
SID=$(echo "$HEADERS" | grep -i 'mcp-session-id' | tr -d '\r' | awk '{print $2}')
echo "Session: $SID"

### 2. Send initialized notification
curl -s -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' > /dev/null

### 3. Get UI session grant
GRANT_RESP=$(curl -s -X POST http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"create_ui_session_link","arguments":{"requested_for":"curl-test","ttl_seconds":3600}}}')
GRANT=$(echo "$GRANT_RESP" | grep '^data:' | sed 's/^data: //' | python3 -c "import sys,json; print(json.loads(json.loads(sys.stdin.read())['result']['content'][0]['text'])['url'].split('grant=')[1])")
echo "Grant: $GRANT"

### 4. Use grant for API calls
curl -s "http://localhost:8000/api/files?grant=$GRANT" | python3 -m json.tool
curl -s -X POST -F file=@test.txt "http://localhost:8000/api/upload?grant=$GRANT"
curl -s "http://localhost:8000/api/files/{file_id}/status?grant=$GRANT"
```

**Key notes:**
- Header is `Mcp-Session-Id` (not `Mcp-Session`)
- Must send `Accept: application/json, text/event-stream` on every request
- Response is SSE format: `event: message\ndata: {json}`
- Must send `notifications/initialized` before calling tools
