---
name: Memory files location preference
description: Memory files for this project must be stored in doc/memory/ within the repo, not in ~/.claude/projects/
type: feedback
---

Memory files must be added under `doc/memory/` in the project repository, not in the default `~/.claude/projects/.../memory/` location.

**Why:** User wants memory to be version-controlled and co-located with the project.

**How to apply:** When creating or updating memory files for this project, always write them to `/Users/apollo/dev/rag-anywhere/doc/memory/`. Update MEMORY.md index with relative paths pointing there.
