"""MemoryMiddleware — persistent agent memory from AGENTS.md and memory dirs.

Loads repository-scoped instructions from AGENTS.md files and user/session
memory from configured directories.  All injected read-only into the
system prompt.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)


def _read_file_safe(path: str, max_bytes: int = 32768) -> str | None:
    """Read a text file, returning None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(max_bytes)
    except Exception:
        return None


class MemoryStore:
    """Read-only memory loader from filesystem directories."""

    def __init__(
        self,
        agents_md_paths: list[str] | None = None,
        memory_dirs: list[str] | None = None,
    ) -> None:
        self._agents_md_paths = agents_md_paths or []
        self._memory_dirs = memory_dirs or []
        self._repo_instructions: str = ""
        self._memories: dict[str, str] = {}  # filename → content
        self._loaded = False

    def load(self) -> None:
        """Load all memory sources."""
        # Load AGENTS.md files (repo instructions)
        repo_parts = []
        for path in self._agents_md_paths:
            if os.path.isfile(path):
                content = _read_file_safe(path)
                if content:
                    repo_parts.append(content.strip())
        self._repo_instructions = "\n\n".join(repo_parts)

        # Load memory files from memory directories
        self._memories.clear()
        for mem_dir in self._memory_dirs:
            if not os.path.isdir(mem_dir):
                continue
            for entry in sorted(os.listdir(mem_dir)):
                fpath = os.path.join(mem_dir, entry)
                if os.path.isfile(fpath) and entry.endswith(".md"):
                    content = _read_file_safe(fpath)
                    if content:
                        self._memories[entry] = content.strip()
        self._loaded = True

    @property
    def repo_instructions(self) -> str:
        if not self._loaded:
            self.load()
        return self._repo_instructions

    @property
    def memories(self) -> dict[str, str]:
        if not self._loaded:
            self.load()
        return self._memories


class MemoryMiddleware(AgentMiddleware):
    """Inject persistent memory and repo instructions into system prompt."""

    def __init__(
        self,
        agents_md_paths: list[str] | None = None,
        memory_dirs: list[str] | None = None,
    ) -> None:
        self.store = MemoryStore(agents_md_paths, memory_dirs)

    async def initialize(self, state: AgentState) -> None:
        self.store.load()

    async def before_model(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[0], dict):
            return state
        if messages[0].get("role") != "system":
            return state

        parts: list[str] = []

        # Repository instructions
        repo = self.store.repo_instructions
        if repo:
            parts.append(f"<repoInstructions>\n{repo}\n</repoInstructions>")

        # User/session memories
        memories = self.store.memories
        if memories:
            mem_text = "\n".join(
                f"### {name}\n{content}" for name, content in memories.items()
            )
            parts.append(f"<memory>\n{mem_text}\n</memory>")

        if parts:
            addition = "\n\n" + "\n\n".join(parts)
            content = messages[0].get("content", "")
            if isinstance(content, str) and "<repoInstructions>" not in content and "<memory>" not in content:
                messages[0]["content"] = content + addition

        return state
