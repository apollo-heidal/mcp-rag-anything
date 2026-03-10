#!/usr/bin/env bash
set -euo pipefail

CMD="${1:-help}"

setup() {
  docker desktop enable model-runner --tcp=12434
}

pull() {
  docker model pull hf.co/unsloth/Qwen3.5-2B-GGUF
  docker model pull hf.co/unsloth/Qwen3-Embedding-0.6B
}

case "$CMD" in
  setup)
    setup
    ;;
  pull)
    pull
    ;;
  build)
    docker compose build
    ;;
  start)
    setup
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
    curl -s http://localhost:12434/engines/v1/models | python3 -m json.tool
    ;;
  *)
    echo "Usage: $0 {setup|pull|build|start|stop|verify}"
    exit 1
    ;;
esac
