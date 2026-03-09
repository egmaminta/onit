"""CompositeBackend — route paths to different backends by prefix.

Allows mounting multiple backends at different path prefixes, e.g.:
  - ``/memories/`` → StateBackend (ephemeral)
  - ``/``          → FilesystemBackend (disk)

Uses longest-prefix matching.

No langchain / langgraph dependency.
"""

from __future__ import annotations

from typing import Optional

from .protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)


class CompositeBackend:
    """Routes operations to backends based on path prefix."""

    def __init__(self, default: BackendProtocol) -> None:
        self._default = default
        self._routes: list[tuple[str, BackendProtocol]] = []

    def mount(self, prefix: str, backend: BackendProtocol) -> None:
        """Mount *backend* at *prefix* (e.g. ``/memories/``)."""
        prefix = prefix.rstrip("/") + "/"
        self._routes.append((prefix, backend))
        # Sort by prefix length descending for longest-prefix match
        self._routes.sort(key=lambda r: len(r[0]), reverse=True)

    def _resolve(self, path: str) -> tuple[BackendProtocol, str]:
        """Find the backend for *path* and return (backend, local_path)."""
        for prefix, backend in self._routes:
            if path.startswith(prefix):
                local_path = path[len(prefix) - 1:]  # keep leading /
                return backend, local_path or "/"
        return self._default, path

    # ── ls_info ─────────────────────────────────────────────────

    def ls_info(self, path: str = "/") -> list[FileInfo]:
        backend, local = self._resolve(path)
        results = backend.ls_info(local)
        # At root level, also show route prefixes as virtual directories
        if path == "/" or path == "":
            prefix_dirs = {r[0].strip("/").split("/")[0] for r in self._routes}
            existing_names = {fi.name for fi in results}
            for d in sorted(prefix_dirs):
                if d not in existing_names:
                    results.append(FileInfo(name=d, path=f"/{d}", is_dir=True))
        return results

    async def als_info(self, path: str = "/") -> list[FileInfo]:
        backend, local = self._resolve(path)
        results = await backend.als_info(local)
        if path == "/" or path == "":
            prefix_dirs = {r[0].strip("/").split("/")[0] for r in self._routes}
            existing_names = {fi.name for fi in results}
            for d in sorted(prefix_dirs):
                if d not in existing_names:
                    results.append(FileInfo(name=d, path=f"/{d}", is_dir=True))
        return results

    # ── read ────────────────────────────────────────────────────

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        backend, local = self._resolve(file_path)
        return backend.read(local, offset, limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        backend, local = self._resolve(file_path)
        return await backend.aread(local, offset, limit)

    # ── write ───────────────────────────────────────────────────

    def write(self, file_path: str, content: str) -> WriteResult:
        backend, local = self._resolve(file_path)
        return backend.write(local, content)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        backend, local = self._resolve(file_path)
        return await backend.awrite(local, content)

    # ── edit ────────────────────────────────────────────────────

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult:
        backend, local = self._resolve(file_path)
        return backend.edit(local, old_string, new_string, replace_all)

    async def aedit(self, file_path: str, old_string: str, new_string: str,
                    replace_all: bool = False) -> EditResult:
        backend, local = self._resolve(file_path)
        return await backend.aedit(local, old_string, new_string, replace_all)

    # ── grep_raw ────────────────────────────────────────────────

    def grep_raw(self, pattern: str, path: Optional[str] = None,
                 glob: Optional[str] = None) -> list[GrepMatch]:
        if path:
            backend, local = self._resolve(path)
            return backend.grep_raw(pattern, local, glob)
        # Search ALL backends when no path given
        results: list[GrepMatch] = []
        results.extend(self._default.grep_raw(pattern, None, glob))
        for prefix, backend in self._routes:
            for m in backend.grep_raw(pattern, None, glob):
                # Re-prefix paths so callers see the composite path
                results.append(GrepMatch(
                    path=prefix.rstrip("/") + "/" + m.path.lstrip("/"),
                    line_number=m.line_number,
                    line=m.line,
                    column=m.column,
                ))
        return results

    async def agrep_raw(self, pattern: str, path: Optional[str] = None,
                        glob: Optional[str] = None) -> list[GrepMatch]:
        if path:
            backend, local = self._resolve(path)
            return await backend.agrep_raw(pattern, local, glob)
        results: list[GrepMatch] = []
        results.extend(await self._default.agrep_raw(pattern, None, glob))
        for prefix, backend in self._routes:
            for m in await backend.agrep_raw(pattern, None, glob):
                results.append(GrepMatch(
                    path=prefix.rstrip("/") + "/" + m.path.lstrip("/"),
                    line_number=m.line_number,
                    line=m.line,
                    column=m.column,
                ))
        return results

    # ── glob_info ───────────────────────────────────────────────

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        backend, local = self._resolve(path)
        return backend.glob_info(pattern, local)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        backend, local = self._resolve(path)
        return await backend.aglob_info(pattern, local)

    # ── upload / download ───────────────────────────────────────

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        results = []
        for fpath, data in files:
            backend, local = self._resolve(fpath)
            res = backend.upload_files([(local, data)])
            results.extend(res)
        return results

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        results = []
        for fpath, data in files:
            backend, local = self._resolve(fpath)
            res = await backend.aupload_files([(local, data)])
            results.extend(res)
        return results

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for p in paths:
            backend, local = self._resolve(p)
            res = backend.download_files([local])
            results.extend(res)
        return results

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for p in paths:
            backend, local = self._resolve(p)
            res = await backend.adownload_files([local])
            results.extend(res)
        return results

    # ── execute (delegate to default if it supports it) ─────────

    def execute(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse:
        if isinstance(self._default, SandboxBackendProtocol):
            return self._default.execute(command, timeout)
        raise NotImplementedError("Default backend does not support execution")

    async def aexecute(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse:
        if isinstance(self._default, SandboxBackendProtocol):
            return await self._default.aexecute(command, timeout)
        raise NotImplementedError("Default backend does not support execution")
