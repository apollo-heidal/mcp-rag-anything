FROM oven/bun:1 AS base
WORKDIR /app

# Copy Bun project
COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile

# Python + venv
RUN apt-get update && apt-get install -y python3 python3-venv && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# App source
COPY server.ts rag_bridge.py ./

CMD ["bun", "server.ts"]
