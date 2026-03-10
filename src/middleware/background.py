"""BackgroundMiddleware — background task execution.

Provides ``background_run`` and ``check_background`` virtual tools so
the agent can launch long-running tasks and check on them later.

Notifications from completed background tasks are drained into the
conversation via the ``before_model`` hook.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundTask:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)


class BackgroundManager:
    """Manages background async tasks."""

    MAX_TASKS = 20

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._notifications: list[str] = []

    async def run_task(
        self,
        description: str,
        coro: Any,
    ) -> str:
        """Schedule a coroutine as a background task. Returns task ID."""
        if len(self._tasks) >= self.MAX_TASKS:
            # Remove oldest completed tasks to make room
            completed = [
                tid for tid, t in self._tasks.items()
                if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            ]
            for tid in completed[:5]:
                del self._tasks[tid]

        task_id = uuid.uuid4().hex[:8]
        bg = BackgroundTask(id=task_id, description=description)
        self._tasks[task_id] = bg

        async def _wrapper():
            bg.status = TaskStatus.RUNNING
            try:
                result = await coro
                bg.result = str(result) if result else "(completed)"
                bg.status = TaskStatus.COMPLETED
                self._notifications.append(
                    f"Background task '{bg.description}' ({bg.id}) completed successfully."
                )
            except Exception as e:
                bg.error = str(e)
                bg.status = TaskStatus.FAILED
                self._notifications.append(
                    f"Background task '{bg.description}' ({bg.id}) failed: {e}"
                )
            finally:
                bg.completed_at = time.time()

        bg._task = asyncio.create_task(_wrapper())
        return task_id

    def check_task(self, task_id: str) -> str | None:
        """Get status summary for a task."""
        bg = self._tasks.get(task_id)
        if bg is None:
            return None
        elapsed = (bg.completed_at or time.time()) - bg.created_at
        lines = [
            f"Task: {bg.description}",
            f"ID: {bg.id}",
            f"Status: {bg.status.value}",
            f"Elapsed: {elapsed:.1f}s",
        ]
        if bg.status == TaskStatus.COMPLETED:
            lines.append(f"Result: {bg.result[:2000]}")
        elif bg.status == TaskStatus.FAILED:
            lines.append(f"Error: {bg.error[:1000]}")
        return "\n".join(lines)

    def list_tasks(self) -> str:
        """Summary of all tasks."""
        if not self._tasks:
            return "No background tasks."
        lines = []
        for bg in self._tasks.values():
            lines.append(f"[{bg.status.value}] {bg.id}: {bg.description}")
        return "\n".join(lines)

    def drain_notifications(self) -> list[str]:
        """Return and clear pending notifications."""
        notifs = list(self._notifications)
        self._notifications.clear()
        return notifs


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_BG_RUN_TOOL = {
    "type": "function",
    "function": {
        "name": "background_run",
        "description": "Launch a command in the background sandbox. Returns a task ID for later checking.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "description": {"type": "string", "description": "Brief description of the task."},
            },
            "required": ["command"],
        },
    },
}

_BG_CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "check_background",
        "description": "Check status of a background task by ID, or list all tasks if no ID given.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to check (omit to list all)."},
            },
            "required": [],
        },
    },
}


class BackgroundMiddleware(AgentMiddleware):
    """Background task execution via virtual tools."""

    def __init__(self) -> None:
        self.manager = BackgroundManager()

    async def before_model(self, state: AgentState) -> AgentState:
        # Drain notifications into messages
        notifications = self.manager.drain_notifications()
        if notifications:
            messages = state.get("messages", [])
            # Inject as system-level messages so the model doesn't confuse them with user input
            for notif in notifications:
                messages.append({
                    "role": "system",
                    "content": f"[Background notification] {notif}",
                })

        # Inject tools
        tools = state.get("tools", [])
        existing_names = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
        if "background_run" not in existing_names:
            tools.append(_BG_RUN_TOOL)
        if "check_background" not in existing_names:
            tools.append(_BG_CHECK_TOOL)
        state["tools"] = tools

        return state

    async def handle_tool(self, state: AgentState, tool_name: str, args: dict) -> Optional[str]:
        if tool_name == "background_run":
            command = args.get("command", "")
            desc = args.get("description", command[:60])
            if not command:
                return "Error: command is required."

            # Use backend if available for execution
            backend = state.get("backend")

            try:
                from ..backend.protocol import SandboxBackendProtocol
            except (ImportError, ValueError):
                try:
                    from backend.protocol import SandboxBackendProtocol
                except ImportError:
                    SandboxBackendProtocol = None

            if SandboxBackendProtocol and isinstance(backend, SandboxBackendProtocol):
                coro = backend.aexecute(command, timeout=300)
            else:
                # Fallback: run via asyncio subprocess
                async def _run_subprocess():
                    proc = await asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
                    return f"exit={proc.returncode}\nstdout:\n{stdout.decode(errors='replace')[:5000]}\nstderr:\n{stderr.decode(errors='replace')[:2000]}"
                coro = _run_subprocess()

            task_id = await self.manager.run_task(desc, coro)
            return f"Background task launched: {task_id}\nDescription: {desc}\nUse check_background(task_id='{task_id}') to monitor."

        elif tool_name == "check_background":
            task_id = args.get("task_id", "")
            if task_id:
                result = self.manager.check_task(task_id)
                return result if result else f"Unknown task ID: {task_id}"
            return self.manager.list_tasks()

        return None
