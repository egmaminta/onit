"""TodoListMiddleware — lightweight in-memory todo tracking.

Injects a ``write_todos`` virtual tool and renders a checklist summary
in the system prompt footer.  Nags the model if todos haven't been
updated in N turns.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)

MAX_ITEMS = 20
NAG_INTERVAL = 3  # turns between nag reminders


class TodoStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class TodoManager:
    """Manages the in-memory todo list with validation constraints."""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self._turns_since_update = 0

    def write(self, todos: list[dict]) -> str:
        """Replace the entire todo list.  Returns validation message."""
        if len(todos) > MAX_ITEMS:
            return f"Error: max {MAX_ITEMS} todo items allowed."

        # Validate statuses and single in_progress constraint
        in_progress_count = 0
        clean = []
        for item in todos:
            status = item.get("status", "pending")
            if status not in {s.value for s in TodoStatus}:
                return f"Error: invalid status '{status}'. Use: pending, in_progress, completed."
            if status == "in_progress":
                in_progress_count += 1
            if in_progress_count > 1:
                return "Error: only one todo can be in_progress at a time."
            clean.append({
                "id": item.get("id", len(clean) + 1),
                "title": item.get("title", "Untitled"),
                "status": status,
            })

        self.items = clean
        self._turns_since_update = 0
        # Return full list so the UI shows the complete state
        lines = ["Todos updated."]
        for t in clean:
            mark = {"completed": "✓", "in_progress": "→", "pending": "○"}.get(t["status"], "?")
            lines.append(f"  {mark} [{t['status']}] {t['title']}")
        return "\n".join(lines)

    def tick(self) -> None:
        """Called each turn to track nag timing."""
        self._turns_since_update += 1

    @property
    def should_nag(self) -> bool:
        return bool(self.items) and self._turns_since_update >= NAG_INTERVAL

    def render(self) -> str:
        """Render the checklist as text."""
        if not self.items:
            return ""
        lines = []
        completed = sum(1 for i in self.items if i["status"] == "completed")
        total = len(self.items)
        for item in self.items:
            status = item["status"]
            if status == "completed":
                marker = "[x]"
            elif status == "in_progress":
                marker = "[>]"
            else:
                marker = "[ ]"
            lines.append(f"{marker} {item['title']}")
        lines.append(f"({completed}/{total} completed)")
        return "\n".join(lines)


# Workflow instruction injected into the system prompt
_TODO_WORKFLOW_INSTRUCTION = (
    "<taskWorkflow>\n"
    "IMPORTANT: For any task that involves multiple steps, tool usage, or non-trivial work:\n"
    "1. FIRST call write_todos to create a plan with clear, actionable items before calling any other tools\n"
    "2. Mark the current step as in_progress before starting work on it\n"
    "3. Complete the work for that step using the appropriate tools\n"
    "4. Update todos to mark it completed, then move to the next step\n"
    "For simple greetings, factual questions, or brief conversational responses, respond directly without todos.\n"
    "</taskWorkflow>"
)

# Tool definition injected into the tools list
_WRITE_TODOS_TOOL = {
    "type": "function",
    "function": {
        "name": "write_todos",
        "description": (
            "Update the todo list to track task progress. Each item has: "
            "id (number), title (string), status (pending|in_progress|completed). "
            "Only one item can be in_progress at a time. Max 20 items."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "title": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        },
                        "required": ["id", "title", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
}


class TodoListMiddleware(AgentMiddleware):
    """Inject todo tracking into the agent loop."""

    def __init__(self) -> None:
        self.manager = TodoManager()

    async def before_model(self, state: AgentState) -> AgentState:
        # Inject write_todos tool
        tools = state.get("tools", [])
        tool_names = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
        if "write_todos" not in tool_names:
            tools.append(_WRITE_TODOS_TOOL)
            state["tools"] = tools

        # Inject workflow instruction and todo summary into system prompt
        messages = state.get("messages", [])
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            sys_content = messages[0].get("content", "")
            if isinstance(sys_content, str) and "<taskWorkflow>" not in sys_content:
                messages[0]["content"] = sys_content + "\n\n" + _TODO_WORKFLOW_INSTRUCTION

        # Append todo checklist to system prompt
        checklist = self.manager.render()
        if checklist:
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                sys_content = messages[0].get("content", "")
                if isinstance(sys_content, str) and "<todos>" not in sys_content:
                    messages[0]["content"] = sys_content + f"\n\n<todos>\n{checklist}\n</todos>"

        # Nag reminder
        self.manager.tick()
        if self.manager.should_nag:
            messages = state.get("messages", [])
            messages.append({
                "role": "user",
                "content": "<reminder>Update your todos to reflect current progress.</reminder>",
            })

        return state

    async def handle_tool(self, state: AgentState, tool_name: str, args: dict) -> Optional[str]:
        if tool_name != "write_todos":
            return None
        todos = args.get("todos", [])
        return self.manager.write(todos)
