"""WorktreeMiddleware — git worktree management tools.

Injects worktree tools only if git is available. Provides create, list,
status, run, remove, keep, and events tools, plus task-bind.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from ..agent.worktree import WorktreeManager
    from ..agent.event_bus import EventBus
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from agent.worktree import WorktreeManager
    from agent.event_bus import EventBus
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_WT_CREATE = {
    "type": "function",
    "function": {
        "name": "worktree_create",
        "description": "Create a new git worktree, optionally linked to a task.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Worktree name (1-40 chars, alphanumeric/._-)."},
                "task_id": {"type": "integer", "description": "Optional task ID to bind."},
                "base_ref": {"type": "string", "description": "Git ref to branch from (default: HEAD)."},
            },
            "required": ["name"],
        },
    },
}

_WT_LIST = {
    "type": "function",
    "function": {
        "name": "worktree_list",
        "description": "List all git worktrees.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_WT_STATUS = {
    "type": "function",
    "function": {
        "name": "worktree_status",
        "description": "Get git status for a worktree.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Worktree name."},
            },
            "required": ["name"],
        },
    },
}

_WT_RUN = {
    "type": "function",
    "function": {
        "name": "worktree_run",
        "description": "Execute a command inside a worktree directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Worktree name."},
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 300)."},
            },
            "required": ["name", "command"],
        },
    },
}

_WT_REMOVE = {
    "type": "function",
    "function": {
        "name": "worktree_remove",
        "description": "Remove a git worktree.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Worktree name."},
                "force": {"type": "boolean", "description": "Force removal."},
                "complete_task": {"type": "boolean", "description": "Mark bound task as completed."},
            },
            "required": ["name"],
        },
    },
}

_WT_KEEP = {
    "type": "function",
    "function": {
        "name": "worktree_keep",
        "description": "Mark a worktree as kept (prevent auto-removal).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Worktree name."},
            },
            "required": ["name"],
        },
    },
}

_WT_EVENTS = {
    "type": "function",
    "function": {
        "name": "worktree_events",
        "description": "List recent worktree lifecycle events.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max events to show (default: 20)."},
            },
            "required": [],
        },
    },
}

_TASK_BIND_WT = {
    "type": "function",
    "function": {
        "name": "task_bind_worktree",
        "description": "Bind a task to a worktree and set owner.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "Task ID."},
                "worktree": {"type": "string", "description": "Worktree name."},
                "owner": {"type": "string", "description": "Owner name."},
            },
            "required": ["task_id", "worktree"],
        },
    },
}

_ALL_WT_TOOLS = [_WT_CREATE, _WT_LIST, _WT_STATUS, _WT_RUN, _WT_REMOVE, _WT_KEEP, _WT_EVENTS, _TASK_BIND_WT]


class WorktreeMiddleware(AgentMiddleware):
    """Git worktree management tools — only active if git is available."""

    def __init__(
        self,
        worktree_manager: WorktreeManager | None = None,
        event_bus: EventBus | None = None,
        task_manager: Any = None,
        repo_root: str = ".",
    ) -> None:
        self._wt = worktree_manager
        self._events = event_bus
        self._task_manager = task_manager
        self._repo_root = repo_root

    async def initialize(self, state: AgentState) -> None:
        if self._wt is None:
            self._events = self._events or EventBus()
            self._wt = WorktreeManager(
                event_bus=self._events,
                repo_root=self._repo_root,
            )

    async def before_model(self, state: AgentState) -> AgentState:
        if self._wt is None or not self._wt.git_available:
            return state

        tools = state.get("tools", [])
        existing = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
        for td in _ALL_WT_TOOLS:
            if td["function"]["name"] not in existing:
                tools.append(td)
        state["tools"] = tools
        return state

    async def handle_tool(self, state: AgentState, tool_name: str, args: dict) -> Optional[str]:
        if self._wt is None:
            return None

        if tool_name == "worktree_create":
            return self._wt.create(
                name=args.get("name", ""),
                task_id=args.get("task_id"),
                base_ref=args.get("base_ref", "HEAD"),
            )

        elif tool_name == "worktree_list":
            wts = self._wt.list_all()
            if not wts:
                return "No worktrees."
            lines = []
            for w in wts:
                parts = [f"{w['name']} ({w['branch']})"]
                if w.get("task_id"):
                    parts.append(f"task=#{w['task_id']}")
                if w.get("kept"):
                    parts.append("[kept]")
                lines.append(" ".join(parts))
            return "\n".join(lines)

        elif tool_name == "worktree_status":
            return self._wt.status(args.get("name", ""))

        elif tool_name == "worktree_run":
            return self._wt.run(
                name=args.get("name", ""),
                command=args.get("command", ""),
                timeout=args.get("timeout", 300),
            )

        elif tool_name == "worktree_remove":
            return self._wt.remove(
                name=args.get("name", ""),
                force=args.get("force", False),
                complete_task=args.get("complete_task", False),
            )

        elif tool_name == "worktree_keep":
            return self._wt.keep(args.get("name", ""))

        elif tool_name == "worktree_events":
            if self._events is None:
                return "No event bus configured."
            return self._events.list_recent(limit=args.get("limit", 20))

        elif tool_name == "task_bind_worktree":
            if self._task_manager is None:
                return "Task manager not available."
            task_id = args.get("task_id")
            worktree = args.get("worktree", "")
            owner = args.get("owner", "")
            task = self._task_manager.bind_worktree(int(task_id), worktree, owner)
            if task is None:
                return f"Task #{task_id} not found."
            return f"Task #{task_id} bound to worktree '{worktree}'"

        return None
