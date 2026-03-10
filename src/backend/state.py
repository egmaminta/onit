"""StateBackend — ephemeral in-memory file backend.

All files are stored as plain dicts in memory.  No disk I/O, no execution.
Ideal for testing and subagent sandboxing.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


@dataclass
class _FileData:
    """In-memory representation of a file."""
    lines: list[str] = field(default_factory=list)
    created_at: str = ""
    modified_at: str = ""


class StateBackend:
    """Ephemeral in-memory backend implementing ``BackendProtocol``."""

    def __init__(self) -> None:
        self._files: dict[str, _FileData] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalise(path: str) -> str:
        """Normalise path: always start with ``/``, no trailing ``/``."""
        p = path.replace("\\", "/")
        if not p.startswith("/"):
            p = "/" + p
        return p.rstrip("/") or "/"

    # ── ls_info ─────────────────────────────────────────────────

    def ls_info(self, path: str = "/") -> list[FileInfo]:
        path = self._normalise(path)
        prefix = path if path.endswith("/") else path + "/"
        seen_dirs: set[str] = set()
        result: list[FileInfo] = []

        for fpath, fdata in sorted(self._files.items()):
            if not fpath.startswith(prefix):
                continue
            relative = fpath[len(prefix):]
            if "/" in relative:
                dir_name = relative.split("/", 1)[0]
                if dir_name not in seen_dirs:
                    seen_dirs.add(dir_name)
                    result.append(FileInfo(
                        name=dir_name, path=prefix + dir_name, is_dir=True,
                    ))
            else:
                content = "\n".join(fdata.lines)
                result.append(FileInfo(
                    name=relative, path=fpath, is_dir=False,
                    size=len(content.encode()),
                    modified_at=fdata.modified_at, created_at=fdata.created_at,
                ))
        return result

    async def als_info(self, path: str = "/") -> list[FileInfo]:
        return self.ls_info(path)

    # ── read ────────────────────────────────────────────────────

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        file_path = self._normalise(file_path)
        fd = self._files.get(file_path)
        if fd is None:
            raise FileNotFoundError(file_path)
        lines = fd.lines[offset : offset + limit]
        return "\n".join(lines)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return self.read(file_path, offset, limit)

    # ── write ───────────────────────────────────────────────────

    def write(self, file_path: str, content: str) -> WriteResult:
        file_path = self._normalise(file_path)
        now = self._now()
        created = file_path not in self._files
        self._files[file_path] = _FileData(
            lines=content.split("\n"),
            created_at=now if created else self._files.get(file_path, _FileData()).created_at or now,
            modified_at=now,
        )
        return WriteResult(path=file_path, bytes_written=len(content.encode()), created=created)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    # ── edit ────────────────────────────────────────────────────

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult:
        file_path = self._normalise(file_path)
        fd = self._files.get(file_path)
        if fd is None:
            raise FileNotFoundError(file_path)
        text = "\n".join(fd.lines)
        if replace_all:
            count = text.count(old_string)
            text = text.replace(old_string, new_string)
        else:
            count = 1 if old_string in text else 0
            text = text.replace(old_string, new_string, 1)
        fd.lines = text.split("\n")
        fd.modified_at = self._now()
        return EditResult(path=file_path, replacements=count)

    async def aedit(self, file_path: str, old_string: str, new_string: str,
                    replace_all: bool = False) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    # ── grep_raw ────────────────────────────────────────────────

    def grep_raw(self, pattern: str, path: Optional[str] = None,
                 glob: Optional[str] = None) -> list[GrepMatch]:
        regex = re.compile(pattern, re.IGNORECASE)
        matches: list[GrepMatch] = []
        for fpath, fd in sorted(self._files.items()):
            if path:
                norm_path = self._normalise(path)
                if not fpath.startswith(norm_path):
                    continue
            if glob and not fnmatch.fnmatch(fpath, glob):
                continue
            for i, line in enumerate(fd.lines):
                m = regex.search(line)
                if m:
                    matches.append(GrepMatch(
                        path=fpath, line_number=i + 1,
                        line=line, column=m.start(),
                    ))
        return matches

    async def agrep_raw(self, pattern: str, path: Optional[str] = None,
                        glob: Optional[str] = None) -> list[GrepMatch]:
        return self.grep_raw(pattern, path, glob)

    # ── glob_info ───────────────────────────────────────────────

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        results: list[FileInfo] = []
        for fpath, fd in sorted(self._files.items()):
            if fnmatch.fnmatch(fpath, pattern):
                content = "\n".join(fd.lines)
                results.append(FileInfo(
                    name=fpath.rsplit("/", 1)[-1], path=fpath, is_dir=False,
                    size=len(content.encode()),
                    modified_at=fd.modified_at, created_at=fd.created_at,
                ))
        return results

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self.glob_info(pattern, path)

    # ── upload / download ───────────────────────────────────────

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        results = []
        for fpath, data in files:
            wr = self.write(fpath, data.decode("utf-8", errors="replace"))
            results.append(FileUploadResponse(path=wr.path, bytes_written=wr.bytes_written))
        return results

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for p in paths:
            p = self._normalise(p)
            fd = self._files.get(p)
            if fd:
                content = "\n".join(fd.lines).encode()
                results.append(FileDownloadResponse(path=p, content=content))
            else:
                results.append(FileDownloadResponse(path=p, content=b""))
        return results

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self.download_files(paths)
