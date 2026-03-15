# Project Instructions

- Use native `docker compose` and standard Docker CLI commands for service lifecycle work in this repo. For builds, startup, shutdown, logs, stats, restarts, rebuilds, and verification, prefer direct Docker commands over custom wrappers.
- Treat Docker Desktop with Docker Model Runner enabled as the default local runtime for this project. If setup steps need to be documented or changed, update the README rather than introducing a new wrapper script.
- Use `uv run ...` for one-off local commands only, such as an isolated script, quick inspection, or a narrow verification step. Do not treat `uv run` as a substitute for container logs, container-based runtime validation, or repeatable integration checks for larger changes.
