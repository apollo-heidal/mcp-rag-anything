"""ZIP extraction with __MACOSX / dotfile filtering."""

import asyncio
import zipfile
from pathlib import Path

ARCHIVE_EXTENSIONS = {".zip"}


def _should_skip(name: str) -> bool:
    """Return True for macOS resource fork files and hidden dotfiles."""
    parts = Path(name).parts
    for part in parts:
        if part.startswith("__MACOSX") or part.startswith("._"):
            return True
    return False


async def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a ZIP archive, skipping __MACOSX and ._ entries. Returns flat list of extracted file paths."""

    def _extract():
        extracted = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if _should_skip(info.filename):
                    continue
                target = dest_dir / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                extracted.append(target)
        return extracted

    return await asyncio.to_thread(_extract)


async def extract_recursive(archive_path: Path, dest_dir: Path) -> list[Path]:
    """Extract archive recursively — nested ZIPs are also extracted. Returns all leaf (non-zip) file paths."""
    files = await extract_zip(archive_path, dest_dir)
    result = []
    for f in files:
        if f.suffix.lower() in ARCHIVE_EXTENSIONS:
            nested_dest = dest_dir / f"{f.stem}_extracted"
            nested_dest.mkdir(parents=True, exist_ok=True)
            nested_files = await extract_recursive(f, nested_dest)
            result.extend(nested_files)
            f.unlink(missing_ok=True)
        else:
            result.append(f)
    return result
