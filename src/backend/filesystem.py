"""FilesystemBackend — disk-based backend implementing BackendProtocol.

All paths are resolved relative to a configurable ``root_dir`` to prevent
directory traversal.  Async variants delegate to sync (blocking I/O is
typically fast enough for local disks; wrap in ``asyncio.to_thread`` if
needed for high concurrency).

No langchain / langgraph dependency.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from pathlib import Path
from typing import Optional

from .protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)


class FilesystemBackend:
    """Backend that reads/writes the real filesystem under ``root_dir``."""

    def __init__(self, root_dir: str = ".", virtual_mode: bool = False) -> None:
        self.root = Path(root_dir).resolve()
        self.virtual_mode = virtual_mode  # restrict all ops to subtree

    def _resolve(self, path: str) -> Path:
        """Resolve *path* safely under root_dir.  Raises ValueError on traversal."""
        clean = path.replace("\\", "/").lstrip("/")
        resolved = (self.root / clean).resolve()
        if not str(resolved).startswith(str(self.root)):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    # ── ls_info ─────────────────────────────────────────────────

    def ls_info(self, path: str = "/") -> list[FileInfo]:
        target = self._resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(str(target))
        results: list[FileInfo] = []
        for entry in sorted(target.iterdir()):
            stat = entry.stat()
            results.append(FileInfo(
                name=entry.name,
                path=str(entry.relative_to(self.root)),
                is_dir=entry.is_dir(),
                size=stat.st_size if entry.is_file() else 0,
                modified_at=str(stat.st_mtime),
            ))
        return results

    async def als_info(self, path: str = "/") -> list[FileInfo]:
        return await asyncio.to_thread(self.ls_info, path)

    # ── read ────────────────────────────────────────────────────

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        target = self._resolve(file_path)
        if not target.is_file():
            raise FileNotFoundError(str(target))
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        selected = lines[offset : offset + limit]
        return "".join(selected)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    # ── write ───────────────────────────────────────────────────

    def write(self, file_path: str, content: str) -> WriteResult:
        target = self._resolve(file_path)
        created = not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return WriteResult(path=str(target.relative_to(self.root)),
                           bytes_written=len(content.encode()),
                           created=created)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return await asyncio.to_thread(self.write, file_path, content)

    # ── edit ────────────────────────────────────────────────────

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult:
        target = self._resolve(file_path)
        if not target.is_file():
            raise FileNotFoundError(str(target))
        text = target.read_text(encoding="utf-8")
        if replace_all:
            count = text.count(old_string)
            new_text = text.replace(old_string, new_string)
        else:
            count = 1 if old_string in text else 0
            new_text = text.replace(old_string, new_string, 1)
        target.write_text(new_text, encoding="utf-8")
        return EditResult(path=str(target.relative_to(self.root)), replacements=count)

    async def aedit(self, file_path: str, old_string: str, new_string: str,
                    replace_all: bool = False) -> EditResult:
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    # ── grep_raw ────────────────────────────────────────────────

    def grep_raw(self, pattern: str, path: Optional[str] = None,
                 glob: Optional[str] = None) -> list[GrepMatch]:
        regex = re.compile(pattern, re.IGNORECASE)
        search_root = self._resolve(path) if path else self.root
        matches: list[GrepMatch] = []
        for dirpath, _, filenames in os.walk(search_root):
            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                if glob and not fnmatch.fnmatch(fname, glob):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            m = regex.search(line)
                            if m:
                                matches.append(GrepMatch(
                                    path=str(fpath.relative_to(self.root)),
                                    line_number=i,
                                    line=line.rstrip("\n"),
                                    column=m.start(),
                                ))
                except (PermissionError, OSError):
                    continue
        return matches

    async def agrep_raw(self, pattern: str, path: Optional[str] = None,
                        glob: Optional[str] = None) -> list[GrepMatch]:
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    # ── glob_info ───────────────────────────────────────────────

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        search_root = self._resolve(path)
        results: list[FileInfo] = []
        for match in sorted(search_root.rglob(pattern)):
            stat = match.stat()
            results.append(FileInfo(
                name=match.name,
                path=str(match.relative_to(self.root)),
                is_dir=match.is_dir(),
                size=stat.st_size if match.is_file() else 0,
                modified_at=str(stat.st_mtime),
            ))
        return results

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return await asyncio.to_thread(self.glob_info, pattern, path)

    # ── upload / download ───────────────────────────────────────

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        results = []
        for fpath, data in files:
            wr = self.write(fpath, data.decode("utf-8", errors="replace"))
            results.append(FileUploadResponse(path=wr.path, bytes_written=wr.bytes_written))
        return results

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for p in paths:
            target = self._resolve(p)
            if target.is_file():
                results.append(FileDownloadResponse(
                    path=str(target.relative_to(self.root)),
                    content=target.read_bytes(),
                ))
            else:
                results.append(FileDownloadResponse(path=p, content=b""))
        return results

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return await asyncio.to_thread(self.download_files, paths)
