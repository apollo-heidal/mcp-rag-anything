"""Content-hash computation."""

import hashlib
from pathlib import Path

import aiofiles


async def compute_file_hash(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream file in chunks and return SHA-256 hex digest."""
    sha = hashlib.sha256()
    async with aiofiles.open(path, "rb") as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()
