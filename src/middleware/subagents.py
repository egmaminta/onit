"""SubAgentMiddleware — spawn isolated sub-agent loops.

Injects a ``task`` virtual tool that spawns a child AgentLoop with:
- Fresh messages (no parent context)
- Subset of middleware (no recursive sub-agents)
- All parent tools except ``task`` itself
- Safety limit on iterations

No langchain / langgraph dependency.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from . import AgentMiddleware, MiddlewareChain
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from middleware import AgentMiddleware, MiddlewareChain

logger = logging.getLogger(__name__)

MAX_SUBAGENT_ITERATIONS = 30

_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate_task",
        "description": (
            "Delegate a subtask to an independent sub-agent that runs autonomously "
            "with its own conversation context and full tool access. Use this when "
            "a task can be cleanly separated from the main conversation — e.g. "
            "researching a topic, processing data, generating a file, running analysis, "
            "or any work that would benefit from a focused, isolated agent. "
            "The sub-agent cannot see the parent conversation. Returns its final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short label for the subtask (shown in logs).",
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "Complete, self-contained instructions for the sub-agent. "
                        "Include all necessary context since it cannot see the parent conversation."
                    ),
                },
            },
            "required": ["prompt"],
        },
    },
}


class SubAgentMiddleware(AgentMiddleware):
    """Inject ``task`` tool for spawning sub-agents."""

    def __init__(self, child_middleware: list[AgentMiddleware] | None = None) -> None:
        self._child_middleware = child_middleware or []

    async def before_model(self, state: AgentState) -> AgentState:
        tools = state.get("tools", [])
        tool_names = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
        if "delegate_task" not in tool_names:
            tools.append(_TASK_TOOL)
            state["tools"] = tools
        return state

    async def handle_tool(self, state: AgentState, tool_name: str, args: dict) -> Optional[str]:
        if tool_name != "delegate_task":
            return None

        prompt = args.get("prompt", "")
        name = args.get("name", "subtask")

        if not prompt:
            return "Error: prompt is required for task tool."

        logger.info("Spawning sub-agent: %s", name)

        # Lazy import to avoid circular dependency
        try:
            from ..agent.loop import AgentLoop
        except (ImportError, ValueError):
            from agent.loop import AgentLoop

        # Build child tools (exclude 'task' to prevent recursion)
        parent_tools = state.get("tools", [])
        child_tools = [
            t for t in parent_tools
            if isinstance(t, dict) and t.get("function", {}).get("name") != "delegate_task"
        ]

        # Build child state
        child_state: AgentState = {
            "messages": [
                {"role": "system", "content": state.get("system_prompt", "You are a helpful assistant.")},
                {"role": "user", "content": prompt},
            ],
            "system_prompt": state.get("system_prompt", "You are a helpful assistant."),
            "tools": child_tools,
            "host": state.get("host", ""),
            "host_key": state.get("host_key", "EMPTY"),
            "model": state.get("model"),
            "think": state.get("think", False),
            "stream": False,  # sub-agents don't stream to UI
            "max_tokens": state.get("max_tokens", 8192),
            "timeout": state.get("timeout"),
            "safety_queue": state.get("safety_queue") or asyncio.Queue(),
            "max_iterations": MAX_SUBAGENT_ITERATIONS,
            "verbose": state.get("verbose", False),
            "data_path": state.get("data_path", ""),
        }

        # Build child middleware chain
        child_chain = MiddlewareChain(middlewares=list(self._child_middleware))

        # Build tool_registry reference from parent
        # The child AgentLoop needs the same tool_registry for MCP tool lookup
        try:
            from ..agent.loop import AgentLoop as _AL
        except (ImportError, ValueError):
            from agent.loop import AgentLoop as _AL

        # Get parent's tool_registry from state
        tool_registry = state.get("tool_registry")

        child_loop = _AL(tool_registry=tool_registry, middleware_chain=child_chain)

        try:
            result = await child_loop.run(child_state)
            if result:
                logger.info("Sub-agent '%s' completed: %s", name, result[:200])
                return result
            return f"Sub-agent '{name}' returned no response."
        except Exception as e:
            logger.error("Sub-agent '%s' error: %s", name, e)
            return f"Sub-agent '{name}' encountered an error: {e}"
