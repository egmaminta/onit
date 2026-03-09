"""Backend Protocol — abstract interface for file/execution backends.

All backends implement the same protocol so the agent can work with
in-memory state, local filesystem, Docker containers, or remote
services interchangeably.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# ────────────────────────────────────────────────────────────────
# Result dataclasses
# ────────────────────────────────────────────────────────────────

@dataclass
class FileInfo:
    """Metadata about a file or directory."""
    name: str
    path: str
    is_dir: bool = False
    size: int = 0
    modified_at: str = ""
    created_at: str = ""


@dataclass
class WriteResult:
    """Result of a write operation."""
    path: str
    bytes_written: int = 0
    created: bool = False  # True if file was newly created


@dataclass
class EditResult:
    """Result of an edit (search-and-replace) operation."""
    path: str
    replacements: int = 0
    snippet: str = ""  # context around the replacement


@dataclass
class GrepMatch:
    """A single match from a grep/search operation."""
    path: str
    line_number: int
    line: str
    column: int = 0


@dataclass
class ExecuteResponse:
    """Result of a command execution."""
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass
class FileUploadResponse:
    """Result of uploading a file."""
    path: str
    bytes_written: int = 0


@dataclass
class FileDownloadResponse:
    """Result of downloading a file."""
    path: str
    content: bytes = b""
    mime_type: str = "application/octet-stream"


# ────────────────────────────────────────────────────────────────
# BackendProtocol
# ────────────────────────────────────────────────────────────────

@runtime_checkable
class BackendProtocol(Protocol):
    """Abstract protocol for file-system operations.

    Every method has both a sync and async variant.  Implementations may
    raise ``NotImplementedError`` for the variant they don't support.
    """

    # ── List / browse ───────────────────────────────────────────

    def ls_info(self, path: str = "/") -> list[FileInfo]: ...
    async def als_info(self, path: str = "/") -> list[FileInfo]: ...

    # ── Read ────────────────────────────────────────────────────

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str: ...
    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str: ...

    # ── Write ───────────────────────────────────────────────────

    def write(self, file_path: str, content: str) -> WriteResult: ...
    async def awrite(self, file_path: str, content: str) -> WriteResult: ...

    # ── Edit (search-and-replace) ───────────────────────────────

    def edit(self, file_path: str, old_string: str, new_string: str,
             replace_all: bool = False) -> EditResult: ...
    async def aedit(self, file_path: str, old_string: str, new_string: str,
                    replace_all: bool = False) -> EditResult: ...

    # ── Grep / search ──────────────────────────────────────────

    def grep_raw(self, pattern: str, path: Optional[str] = None,
                 glob: Optional[str] = None) -> list[GrepMatch]: ...
    async def agrep_raw(self, pattern: str, path: Optional[str] = None,
                        glob: Optional[str] = None) -> list[GrepMatch]: ...

    # ── Glob ────────────────────────────────────────────────────

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]: ...
    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]: ...

    # ── Bulk upload / download ──────────────────────────────────

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]: ...
    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]: ...

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]: ...
    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]: ...


@runtime_checkable
class SandboxBackendProtocol(BackendProtocol, Protocol):
    """Extended backend that can execute shell commands."""

    def execute(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse: ...
    async def aexecute(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse: ...
