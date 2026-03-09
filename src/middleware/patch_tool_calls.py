"""PatchToolCallsMiddleware \u2014 fix dangling tool calls in message history.

before_model: detect assistant messages with tool_calls but no matching
tool result \u2192 inject synthetic error tool_result.

Note: raw JSON tool-call extraction is handled by the AgentLoop itself
(not via after_model), so this middleware only patches history.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ..type.agent_state import AgentState
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)


class PatchToolCallsMiddleware(AgentMiddleware):
    """Ensure message history has valid tool_call / tool_result pairing."""

    async def before_model(self, state: AgentState) -> AgentState:
        """Inject synthetic tool results for dangling tool_calls."""
        messages = state.get("messages", [])
        # Collect IDs of tool_results we have
        result_ids = set()
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                tcid = msg.get("tool_call_id")
                if tcid:
                    result_ids.add(tcid)

        # Find dangling assistant tool_calls
        inserts: list[tuple[int, list[dict]]] = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                continue
            missing = []
            for tc in tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                tc_name = None
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    tc_name = fn.get("name", "unknown") if isinstance(fn, dict) else "unknown"
                else:
                    tc_name = getattr(getattr(tc, "function", None), "name", "unknown")
                if tc_id and tc_id not in result_ids:
                    missing.append((tc_id, tc_name))
            if missing:
                synthetics = []
                for tc_id, tc_name in missing:
                    synthetics.append({
                        "role": "tool",
                        "content": f"Error: tool call {tc_name} was not executed (context lost).",
                        "tool_call_id": tc_id,
                        "name": tc_name,
                    })
                    result_ids.add(tc_id)
                inserts.append((i + 1, synthetics))

        # Apply inserts in reverse order to maintain indices
        for idx, synthetics in reversed(inserts):
            for j, s in enumerate(synthetics):
                messages.insert(idx + j, s)

        state["messages"] = messages
        return state
