"""TaskMiddleware — persistent task board exposed as virtual tools.

Connects ``TaskManager`` (file-based persistence) to the middleware chain
with tools: ``task_create``, ``task_list``, ``task_get``, ``task_update``.

Coexists with TodoListMiddleware (todos = in-memory quick tracking;
tasks = persistent project tracking with deps and ownership).

No langchain / langgraph dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from ..agent.tasks import TaskManager, TaskStatus
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from agent.tasks import TaskManager, TaskStatus
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TASK_CREATE = {
    "type": "function",
    "function": {
        "name": "task_create",
        "description": "Create a new persistent task on the task board.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Brief task title."},
                "description": {"type": "string", "description": "Detailed description."},
            },
            "required": ["subject"],
        },
    },
}

_TASK_LIST = {
    "type": "function",
    "function": {
        "name": "task_list",
        "description": "List all tasks on the task board.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

_TASK_GET = {
    "type": "function",
    "function": {
        "name": "task_get",
        "description": "Get details of a specific task by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "Task ID."},
            },
            "required": ["task_id"],
        },
    },
}

_TASK_UPDATE = {
    "type": "function",
    "function": {
        "name": "task_update",
        "description": "Update a task's status, owner, or description.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "Task ID."},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"],
                    "description": "New status.",
                },
                "owner": {"type": "string", "description": "New owner."},
                "description": {"type": "string", "description": "Updated description."},
            },
            "required": ["task_id"],
        },
    },
}

_TASK_DEPEND = {
    "type": "function",
    "function": {
        "name": "task_add_dependency",
        "description": "Add a dependency: task_id is blocked by blocked_by_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "Task that will be blocked."},
                "blocked_by_id": {"type": "integer", "description": "Task that blocks it."},
            },
            "required": ["task_id", "blocked_by_id"],
        },
    },
}

_ALL_TASK_TOOLS = [_TASK_CREATE, _TASK_LIST, _TASK_GET, _TASK_UPDATE, _TASK_DEPEND]


def _format_task(task) -> str:
    blocked = f" blocked_by={task.blocked_by}" if task.blocked_by else ""
    wt = f" worktree={task.worktree}" if task.worktree else ""
    owner = f" owner={task.owner}" if task.owner else ""
    return (
        f"[{task.status.value}] #{task.id}: {task.subject}{owner}{wt}{blocked}\n"
        f"  {task.description}" if task.description else
        f"[{task.status.value}] #{task.id}: {task.subject}{owner}{wt}{blocked}"
    )


class TaskMiddleware(AgentMiddleware):
    """Persistent task board tools."""

    def __init__(self, tasks_dir: str = ".tasks") -> None:
        self.manager = TaskManager(tasks_dir)

    async def before_model(self, state: AgentState) -> AgentState:
        tools = state.get("tools", [])
        existing = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
        for tool_def in _ALL_TASK_TOOLS:
            if tool_def["function"]["name"] not in existing:
                tools.append(tool_def)
        state["tools"] = tools
        return state

    async def handle_tool(self, state: AgentState, tool_name: str, args: dict) -> Optional[str]:
        if tool_name == "task_create":
            task = self.manager.create(
                subject=args.get("subject", "Untitled"),
                description=args.get("description", ""),
            )
            return f"Created task #{task.id}: {task.subject}"

        elif tool_name == "task_list":
            tasks = self.manager.list_all()
            if not tasks:
                return "No tasks on the board."
            return "\n".join(_format_task(t) for t in tasks)

        elif tool_name == "task_get":
            task_id = args.get("task_id")
            if task_id is None:
                return "Error: task_id is required."
            task = self.manager.get(int(task_id))
            if task is None:
                return f"Task #{task_id} not found."
            return _format_task(task)

        elif tool_name == "task_update":
            task_id = args.get("task_id")
            if task_id is None:
                return "Error: task_id is required."
            status = None
            s = args.get("status")
            if s:
                try:
                    status = TaskStatus(s)
                except ValueError:
                    return f"Invalid status: {s}. Use: pending, in_progress, completed."
            task = self.manager.update(
                task_id=int(task_id),
                status=status,
                owner=args.get("owner"),
                description=args.get("description"),
            )
            if task is None:
                return f"Task #{task_id} not found."
            return f"Updated task #{task.id}: {_format_task(task)}"

        elif tool_name == "task_add_dependency":
            task_id = args.get("task_id")
            blocked_by = args.get("blocked_by_id")
            if task_id is None or blocked_by is None:
                return "Error: both task_id and blocked_by_id are required."
            task = self.manager.add_dependency(int(task_id), int(blocked_by))
            if task is None:
                return "One or both task IDs not found."
            return f"Task #{task.id} now blocked by #{blocked_by}"

        return None
