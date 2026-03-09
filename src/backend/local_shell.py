"""LocalShellBackend — filesystem + command execution.

Extends ``FilesystemBackend`` with ``SandboxBackendProtocol.execute()``
for running shell commands under the root directory.

Safety: blocks known-dangerous patterns and truncates large outputs.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from typing import Optional

from .filesystem import FilesystemBackend
from .protocol import ExecuteResponse

logger = logging.getLogger(__name__)

# Patterns that are too dangerous to run even from an agent
_BLOCKED_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r"\bsudo\s+reboot\b", re.IGNORECASE),
    re.compile(r"\bsudo\s+shutdown\b", re.IGNORECASE),
    re.compile(r">\s*/dev/(sd|null|zero)", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+.*\bof=/dev/", re.IGNORECASE),
    re.compile(r":()\{\s*:\|:&\s*\};:", re.IGNORECASE),  # fork bomb
]

DEFAULT_OUTPUT_LIMIT = 50_000  # 50 KB


class LocalShellBackend(FilesystemBackend):
    """Filesystem backend with shell execution capability."""

    def __init__(
        self,
        root_dir: str = ".",
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(root_dir)
        self.output_limit = output_limit
        self.env = env

    # ── execute ─────────────────────────────────────────────────

    def execute(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse:
        # Safety check
        for pattern in _BLOCKED_PATTERNS:
            if pattern.search(command):
                return ExecuteResponse(
                    exit_code=1,
                    stderr=f"Blocked: command matches unsafe pattern: {pattern.pattern}",
                )

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self.env,
            )
            stdout = result.stdout
            stderr = result.stderr

            # Truncate large outputs
            if len(stdout) > self.output_limit:
                stdout = stdout[: self.output_limit] + f"\n... (truncated at {self.output_limit} bytes)"
            if len(stderr) > self.output_limit:
                stderr = stderr[: self.output_limit] + f"\n... (truncated at {self.output_limit} bytes)"

            return ExecuteResponse(
                exit_code=result.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                exit_code=-1,
                stderr=f"Command timed out after {timeout} seconds",
                timed_out=True,
            )
        except Exception as e:
            logger.error("execute() error: %s", e)
            return ExecuteResponse(exit_code=-1, stderr=str(e))

    async def aexecute(self, command: str, timeout: Optional[int] = None) -> ExecuteResponse:
        return await asyncio.to_thread(self.execute, command, timeout)
