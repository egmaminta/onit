"""TeamMiddleware — team coordination tools for lead and teammate agents.

Lead agents get: spawn_teammate, list_teammates, send_message, read_inbox,
broadcast, shutdown_request, check_shutdown_status, plan_review.

Teammate agents get: send_message, read_inbox, shutdown_response,
plan_approval, idle, claim_task.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from ..agent.message_bus import MessageBus
    from ..agent.teammate import TeammateManager
    from ..agent.protocols import ShutdownProtocol, PlanApprovalProtocol
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from agent.message_bus import MessageBus
    from agent.teammate import TeammateManager
    from agent.protocols import ShutdownProtocol, PlanApprovalProtocol
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions — Lead tools
# ---------------------------------------------------------------------------

_SPAWN_TEAMMATE = {
    "type": "function",
    "function": {
        "name": "spawn_teammate",
        "description": "Spawn a new teammate agent with a role and instructions.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique teammate name."},
                "role": {"type": "string", "description": "Role description."},
                "prompt": {"type": "string", "description": "Initial instructions for the teammate."},
            },
            "required": ["name", "role", "prompt"],
        },
    },
}

_LIST_TEAMMATES = {
    "type": "function",
    "function": {
        "name": "list_teammates",
        "description": "List all teammates and their status.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_SEND_MESSAGE = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": "Send a message to a teammate.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient name."},
                "content": {"type": "string", "description": "Message content."},
            },
            "required": ["to", "content"],
        },
    },
}

_READ_INBOX = {
    "type": "function",
    "function": {
        "name": "read_inbox",
        "description": "Read and drain all messages from your inbox.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_BROADCAST = {
    "type": "function",
    "function": {
        "name": "broadcast_message",
        "description": "Broadcast a message to all teammates.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Message to broadcast."},
            },
            "required": ["content"],
        },
    },
}

_SHUTDOWN_REQUEST = {
    "type": "function",
    "function": {
        "name": "shutdown_request",
        "description": "Request a teammate to shut down.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Teammate to shut down."},
            },
            "required": ["target"],
        },
    },
}

_CHECK_SHUTDOWN = {
    "type": "function",
    "function": {
        "name": "check_shutdown_status",
        "description": "Check status of a shutdown request.",
        "parameters": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "description": "Request ID."},
            },
            "required": ["request_id"],
        },
    },
}

_PLAN_REVIEW = {
    "type": "function",
    "function": {
        "name": "plan_review",
        "description": "Review a plan submitted by a teammate.",
        "parameters": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "description": "Plan request ID."},
                "approve": {"type": "boolean", "description": "Whether to approve."},
                "feedback": {"type": "string", "description": "Feedback for the teammate."},
            },
            "required": ["request_id", "approve"],
        },
    },
}

# ---------------------------------------------------------------------------
# Tool definitions — Teammate tools
# ---------------------------------------------------------------------------

_SHUTDOWN_RESPONSE = {
    "type": "function",
    "function": {
        "name": "shutdown_response",
        "description": "Respond to a shutdown request.",
        "parameters": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "description": "Request ID."},
                "approve": {"type": "boolean", "description": "Whether to approve shutdown."},
            },
            "required": ["request_id", "approve"],
        },
    },
}

_PLAN_APPROVAL = {
    "type": "function",
    "function": {
        "name": "plan_approval",
        "description": "Submit a plan to the lead for approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "The plan description."},
            },
            "required": ["plan"],
        },
    },
}

_IDLE_TOOL = {
    "type": "function",
    "function": {
        "name": "idle",
        "description": "Signal that you have no more work. Enters idle phase.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_CLAIM_TASK = {
    "type": "function",
    "function": {
        "name": "claim_task",
        "description": "Claim an unclaimed pending task from the task board.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "Task ID to claim."},
            },
            "required": ["task_id"],
        },
    },
}

LEAD_TOOLS = [
    _SPAWN_TEAMMATE, _LIST_TEAMMATES, _SEND_MESSAGE, _READ_INBOX,
    _BROADCAST, _SHUTDOWN_REQUEST, _CHECK_SHUTDOWN, _PLAN_REVIEW,
]

TEAMMATE_TOOLS = [
    _SEND_MESSAGE, _READ_INBOX, _SHUTDOWN_RESPONSE,
    _PLAN_APPROVAL, _IDLE_TOOL, _CLAIM_TASK,
]


class TeamMiddleware(AgentMiddleware):
    """Team coordination tools."""

    def __init__(
        self,
        bus: MessageBus,
        teammate_manager: TeammateManager,
        shutdown_protocol: ShutdownProtocol,
        plan_protocol: PlanApprovalProtocol,
        agent_name: str = "lead",
        is_lead: bool = True,
        task_manager: Any = None,
    ) -> None:
        self._bus = bus
        self._tm = teammate_manager
        self._shutdown = shutdown_protocol
        self._plan = plan_protocol
        self._name = agent_name
        self._is_lead = is_lead
        self._task_manager = task_manager

    async def before_model(self, state: AgentState) -> AgentState:
        tools = state.get("tools", [])
        existing = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}

        tool_set = LEAD_TOOLS if self._is_lead else TEAMMATE_TOOLS
        for td in tool_set:
            if td["function"]["name"] not in existing:
                tools.append(td)
        state["tools"] = tools
        return state

    async def handle_tool(self, state: AgentState, tool_name: str, args: dict) -> Optional[str]:
        name = state.get("metadata", {}).get("teammate_name", self._name)

        if tool_name == "spawn_teammate":
            return self._tm.spawn(
                name=args.get("name", ""),
                role=args.get("role", ""),
                prompt=args.get("prompt", ""),
            )

        elif tool_name == "list_teammates":
            members = self._tm.list_all()
            if not members:
                return "No teammates."
            lines = [f"[{m['status']}] {m['name']} ({m['role']}) — up {m['uptime']}" for m in members]
            return "\n".join(lines)

        elif tool_name == "send_message":
            to = args.get("to", "")
            content = args.get("content", "")
            msg_id = self._bus.send(name, to, content)
            return f"Message sent to {to} (id={msg_id})"

        elif tool_name == "read_inbox":
            messages = self._bus.read_inbox(name)
            if not messages:
                return "Inbox empty."
            lines = []
            for m in messages:
                lines.append(f"[{m.get('type', 'message')}] from {m.get('from', '?')}: {m.get('content', '')}")
            return "\n".join(lines)

        elif tool_name == "broadcast_message":
            content = args.get("content", "")
            members = self._tm.list_all()
            teammate_names = [m["name"] for m in members]
            return self._bus.broadcast(name, content, teammate_names)

        elif tool_name == "shutdown_request":
            target = args.get("target", "")
            request_id = self._shutdown.request_shutdown(name, target)
            return f"Shutdown requested for '{target}' (request_id={request_id})"

        elif tool_name == "check_shutdown_status":
            request_id = args.get("request_id", "")
            return self._shutdown.check_status(request_id)

        elif tool_name == "plan_review":
            request_id = args.get("request_id", "")
            approve = args.get("approve", False)
            feedback = args.get("feedback", "")
            return self._plan.review_plan(name, request_id, approve, feedback)

        elif tool_name == "shutdown_response":
            request_id = args.get("request_id", "")
            approve = args.get("approve", True)
            self._shutdown.respond_shutdown(name, "lead", request_id, approve)
            return f"Shutdown {'approved' if approve else 'rejected'}"

        elif tool_name == "plan_approval":
            plan = args.get("plan", "")
            request_id = self._plan.submit_plan(name, "lead", plan)
            return f"Plan submitted for approval (request_id={request_id})"

        elif tool_name == "idle":
            return "Entering idle phase. Will check inbox periodically."

        elif tool_name == "claim_task":
            task_id = args.get("task_id")
            if task_id is None:
                return "Error: task_id required."
            if self._task_manager is None:
                return "Task manager not available."
            try:
                from ..agent.tasks import TaskStatus
            except (ImportError, ValueError):
                from agent.tasks import TaskStatus
            task = self._task_manager.get(int(task_id))
            if task is None:
                return f"Task #{task_id} not found."
            if task.owner:
                return f"Task #{task_id} already claimed by {task.owner}."
            if task.blocked_by:
                return f"Task #{task_id} is blocked by: {task.blocked_by}"
            updated = self._task_manager.update(int(task_id), status=TaskStatus.IN_PROGRESS, owner=name)
            if updated:
                return f"Claimed task #{task_id}: {updated.subject}"
            return f"Failed to claim task #{task_id}."

        return None
