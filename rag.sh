#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-help}"

pull() {
  docker model pull ai/qwen3.5:2b-instruct-Q4_K_M
  docker model pull ai/qwen3-embedding:0.6b-Q4_K_M
}

case "$CMD" in
  pull)
    pull
    ;;
  build)
    docker compose build
    ;;
  start)
    pull
    docker compose up --build
    ;;
  stop)
    docker compose down
    ;;
  verify)
    echo "==> Models available:"
    docker model list
    echo ""
    echo "==> Docker Model Runner API:"
    curl -s http://localhost:12434/engines/llama.cpp/v1/models | python3 -m json.tool
    ;;
  *)
    echo "Usage: $0 {pull|build|start|stop|verify}"
    exit 1
    ;;
esac
