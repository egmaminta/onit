"""ContainerSandbox — Docker-based execution backend.

Implements ``SandboxBackendProtocol`` by executing all operations inside
a Docker container.  File operations use shell commands via ``docker exec``.

Only available when Docker is installed on the host.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import shlex
import shutil
import subprocess
from typing import Optional

from .protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300  # 5 minutes


def docker_available() -> bool:
    """Check if Docker CLI is available on the host."""
    return shutil.which("docker") is not None


class ContainerSandbox:
    """Backend that runs all operations inside a Docker container."""

    def __init__(
        self,
        container_id: str,
        workdir: str = "/workspace",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        if not docker_available():
            raise RuntimeError("Docker is not available on this system")
        self.container_id = container_id
        self.workdir = workdir
        self.timeout = timeout

    # ── Low-level exec ──────────────────────────────────────────

    def _docker_exec(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse:
        """Run a command inside the container."""
        t = timeout or self.timeout
        try:
            result = subprocess.run(
                ["docker", "exec", "-w", self.workdir, self.container_id, "sh", "-c", command],
                capture_output=True, text=True, timeout=t,
            )
            return ExecuteResponse(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(exit_code=-1, stderr=f"Timed out after {t}s", timed_out=True)
        except Exception as e:
            return ExecuteResponse(exit_code=-1, stderr=str(e))

    async def _adocker_exec(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse:
        return await asyncio.to_thread(self._docker_exec, command, timeout)

    # ── execute (SandboxBackendProtocol) ────────────────────────

    def execute(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse:
        return self._docker_exec(command, timeout)

    async def aexecute(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse:
        return await self._adocker_exec(command, timeout)

    # ── ls_info ─────────────────────────────────────────────────

    def ls_info(self, path: str = "/") -> list[FileInfo]:
        resp = self._docker_exec(f"ls -la {shlex.quote(path)} 2>/dev/null || echo 'NOT_FOUND'")
        if resp.exit_code != 0 or "NOT_FOUND" in resp.stdout:
            return []
        results: list[FileInfo] = []
        for line in resp.stdout.strip().split("\n")[1:]:  # skip "total" line
            parts = line.split(None, 8)
            if len(parts) < 9:
                continue
            name = parts[8]
            if name in (".", ".."):
                continue
            is_dir = parts[0].startswith("d")
            size = int(parts[4]) if not is_dir else 0
            results.append(FileInfo(name=name, path=f"{path.rstrip('/')}/{name}", is_dir=is_dir, size=size))
        return results

    async def als_info(self, path: str = "/") -> list[FileInfo]:
        return await asyncio.to_thread(self.ls_info, path)

    # ── read ────────────────────────────────────────────────────

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        start = offset + 1
        end = offset + limit
        resp = self._docker_exec(f"sed -n '{start},{end}p' {shlex.quote(file_path)}")
        if resp.exit_code != 0:
            raise FileNotFoundError(f"{file_path}: {resp.stderr}")
        return resp.stdout

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    # ── write ───────────────────────────────────────────────────

    def write(self, file_path: str, content: str) -> WriteResult:
        # Use base64 + heredoc to avoid shell injection and ARG_MAX
        encoded = base64.b64encode(content.encode()).decode()
        resp = self._docker_exec(
            f"mkdir -p $(dirname {shlex.quote(file_path)}) && echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(file_path)}"
        )
        if resp.exit_code != 0:
            raise IOError(f"Write failed: {resp.stderr}")
        return WriteResult(path=file_path, bytes_written=len(content.encode()), created=True)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return await asyncio.to_thread(self.write, file_path, content)

    # ── edit ────────────────────────────────────────────────────

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult:
        # Read, replace in Python, write back (avoids sed escaping issues)
        content = self.read(file_path, 0, 100000)
        if replace_all:
            count = content.count(old_string)
            new_content = content.replace(old_string, new_string)
        else:
            count = 1 if old_string in content else 0
            new_content = content.replace(old_string, new_string, 1)
        self.write(file_path, new_content)
        return EditResult(path=file_path, replacements=count)

    async def aedit(self, file_path: str, old_string: str, new_string: str,
                    replace_all: bool = False) -> EditResult:
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    # ── grep_raw ────────────────────────────────────────────────

    def grep_raw(self, pattern: str, path: Optional[str] = None,
                 glob: Optional[str] = None) -> list[GrepMatch]:
        search_path = path or self.workdir
        cmd = f"grep -rn -i {shlex.quote(pattern)} {shlex.quote(search_path)}"
        if glob:
            cmd += f" --include={shlex.quote(glob)}"
        resp = self._docker_exec(cmd)
        matches: list[GrepMatch] = []
        for line in resp.stdout.strip().split("\n"):
            if not line or ":" not in line:
                continue
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append(GrepMatch(
                    path=parts[0], line_number=int(parts[1]) if parts[1].isdigit() else 0,
                    line=parts[2],
                ))
        return matches

    async def agrep_raw(self, pattern: str, path: Optional[str] = None,
                        glob: Optional[str] = None) -> list[GrepMatch]:
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    # ── glob_info ───────────────────────────────────────────────

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        resp = self._docker_exec(f"find {path} -name '{pattern}' -maxdepth 5 2>/dev/null")
        results: list[FileInfo] = []
        for fpath in resp.stdout.strip().split("\n"):
            if not fpath:
                continue
            name = fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath
            results.append(FileInfo(name=name, path=fpath, is_dir=False))
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
            content = self.read(p, 0, 100000)
            results.append(FileDownloadResponse(path=p, content=content.encode()))
        return results

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return await asyncio.to_thread(self.download_files, paths)
