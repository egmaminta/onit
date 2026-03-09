"""FilesystemMiddleware — bridge backend protocol to agent tools.

Exposes the BackendProtocol operations (ls, read_file, write_file,
edit_file, glob, grep, execute) as virtual tools so the agent can
interact with the filesystem/sandbox through the middleware chain.

Large outputs are truncated to keep context manageable.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from ..backend.protocol import BackendProtocol, SandboxBackendProtocol
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from backend.protocol import BackendProtocol, SandboxBackendProtocol
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 30_000
TRUNCATION_MSG = "\n\n... [output truncated at {n} chars]"

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_LS_TOOL = {
    "type": "function",
    "function": {
        "name": "backend_ls",
        "description": "List directory contents in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (relative)."},
            },
            "required": [],
        },
    },
}

_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "backend_read_file",
        "description": "Read a file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative)."},
            },
            "required": ["path"],
        },
    },
}

_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "backend_write_file",
        "description": "Write content to a file (create or overwrite).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative)."},
                "content": {"type": "string", "description": "File content."},
            },
            "required": ["path", "content"],
        },
    },
}

_EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "backend_edit_file",
        "description": "Replace exact text in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (relative)."},
                "old_text": {"type": "string", "description": "Exact text to find and replace."},
                "new_text": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
}

_GREP_TOOL = {
    "type": "function",
    "function": {
        "name": "backend_grep",
        "description": "Search for a pattern across files in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex or plain)."},
                "path": {"type": "string", "description": "Optional directory to scope the search."},
            },
            "required": ["pattern"],
        },
    },
}

_GLOB_TOOL = {
    "type": "function",
    "function": {
        "name": "backend_glob",
        "description": "Find files matching a glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')."},
            },
            "required": ["pattern"],
        },
    },
}

_EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "backend_execute",
        "description": "Execute a shell command in the sandbox.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)."},
            },
            "required": ["command"],
        },
    },
}

_ALL_TOOLS = [_LS_TOOL, _READ_TOOL, _WRITE_TOOL, _EDIT_TOOL, _GREP_TOOL, _GLOB_TOOL]


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + TRUNCATION_MSG.format(n=MAX_OUTPUT_CHARS)
    return text


class FilesystemMiddleware(AgentMiddleware):
    """Expose BackendProtocol as virtual tools for the agent."""

    def __init__(self, backend: BackendProtocol | None = None) -> None:
        self._backend = backend

    def _get_backend(self, state: AgentState) -> BackendProtocol | None:
        return self._backend or state.get("backend")

    async def before_model(self, state: AgentState) -> AgentState:
        backend = self._get_backend(state)
        if backend is None:
            return state

        tools = state.get("tools", [])
        existing_names = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}

        for tool_def in _ALL_TOOLS:
            if tool_def["function"]["name"] not in existing_names:
                tools.append(tool_def)

        # Add execute tool only if backend supports sandbox
        if isinstance(backend, SandboxBackendProtocol) and "backend_execute" not in existing_names:
            tools.append(_EXECUTE_TOOL)

        state["tools"] = tools
        return state

    async def handle_tool(self, state: AgentState, tool_name: str, args: dict) -> Optional[str]:
        if not tool_name.startswith("backend_"):
            return None
        backend = self._get_backend(state)
        if backend is None:
            return "No backend configured."

        try:
            if tool_name == "backend_ls":
                infos = await backend.als_info(args.get("path", "."))
                lines = [f"{'d ' if fi.is_dir else '  '}{fi.name}  ({fi.size}B)" for fi in infos]
                return _truncate("\n".join(lines) if lines else "(empty directory)")

            elif tool_name == "backend_read_file":
                content = await backend.aread(args["path"])
                return _truncate(content)

            elif tool_name == "backend_write_file":
                result = await backend.awrite(args["path"], args["content"])
                return f"Wrote {result.bytes_written} bytes to {result.path}"

            elif tool_name == "backend_edit_file":
                result = await backend.aedit(args["path"], args["old_text"], args["new_text"])
                return f"Edited {result.path}: {result.replacements} replacement(s)"

            elif tool_name == "backend_grep":
                matches = await backend.agrep_raw(args["pattern"], args.get("path"))
                lines = [f"{m.path}:{m.line_number}: {m.line_text}" for m in matches]
                return _truncate("\n".join(lines) if lines else "(no matches)")

            elif tool_name == "backend_glob":
                infos = await backend.aglob_info(args["pattern"])
                lines = [f"{'d ' if fi.is_dir else '  '}{fi.name}" for fi in infos]
                return _truncate("\n".join(lines) if lines else "(no matches)")

            elif tool_name == "backend_execute":
                if not isinstance(backend, SandboxBackendProtocol):
                    return "Backend does not support execution."
                resp = await backend.aexecute(
                    args["command"],
                    timeout=args.get("timeout", 30),
                )
                output = f"exit_code={resp.exit_code}\n"
                if resp.stdout:
                    output += f"--- stdout ---\n{resp.stdout}\n"
                if resp.stderr:
                    output += f"--- stderr ---\n{resp.stderr}\n"
                return _truncate(output)

        except FileNotFoundError as e:
            return f"File not found: {e}"
        except PermissionError as e:
            return f"Permission denied: {e}"
        except Exception as e:
            logger.error("FilesystemMiddleware error in %s: %s", tool_name, e)
            return f"Error: {e}"

        return None
