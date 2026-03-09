"""Middleware framework for the OnIt agent loop.

Provides the abstract ``AgentMiddleware`` base class and the
``MiddlewareChain`` that composes a list of middleware and orchestrates
their hook invocations in deterministic order.

Middleware is the primary extension point for the agent: every cross-cutting
concern (todo tracking, summarization, subagents, skills, memory, patching,
filesystem ops, background tasks, teams, …) is implemented as a middleware
that plugs into this chain.

No langchain / langgraph dependency — pure Python + asyncio.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ..type.agent_state import AgentState
except (ImportError, ValueError):
    from type.agent_state import AgentState

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Abstract base
# ────────────────────────────────────────────────────────────────

class AgentMiddleware:
    """Base class for all OnIt middleware.

    Subclasses override any combination of the four hooks below.  Default
    implementations are pass-throughs so middleware only needs to implement
    the hooks it cares about.

    Hooks execute in *chain order* (the order they appear in the list passed
    to ``MiddlewareChain``).  ``before_*`` hooks run first-to-last;
    ``after_*`` hooks run last-to-first (onion model).
    """

    name: str = "base"

    # ── lifecycle ───────────────────────────────────────────────

    async def initialize(self, state: AgentState) -> None:
        """Called once before the agent loop starts.

        Use this for one-time setup: scanning skill directories, loading
        AGENTS.md, etc.  The default is a no-op.
        """

    async def shutdown(self, state: AgentState) -> None:
        """Called once after the agent loop ends (normal or aborted)."""

    # ── per-turn hooks ──────────────────────────────────────────

    async def before_model(self, state: AgentState) -> AgentState:
        """Fires before each LLM call.

        Can modify ``state`` in place (append to messages, inject tools,
        mutate system_prompt, etc.).  Must return the (possibly mutated)
        state.
        """
        return state

    async def after_model(self, state: AgentState, response: Any) -> Any:
        """Fires after the LLM response is received.

        ``response`` is the raw completion object (streaming or not).
        Return the (possibly modified) response.
        """
        return response

    async def before_tool(
        self,
        state: AgentState,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Fires before a tool is executed.

        Return ``(tool_name, args)`` — potentially modified.  A middleware
        can raise an exception to block the tool call.
        """
        return tool_name, args

    async def after_tool(
        self,
        state: AgentState,
        tool_name: str,
        result: str,
    ) -> str:
        """Fires after a tool execution returns.

        Return the (possibly modified) result string.
        """
        return result

    # ── virtual tool handling ───────────────────────────────────

    async def handle_tool(
        self,
        state: AgentState,
        tool_name: str,
        args: dict[str, Any],
    ) -> str | None:
        """Handle a *virtual* tool call injected by this middleware.

        If this middleware injected a tool definition (e.g. ``write_todos``,
        ``load_skill``, ``task``, etc.) it should implement this method to
        execute the call.  Return the result string, or ``None`` if this
        middleware does not own the tool — the chain will try the next
        middleware.
        """
        return None


# ────────────────────────────────────────────────────────────────
# Middleware chain
# ────────────────────────────────────────────────────────────────

class MiddlewareChain:
    """Composes an ordered list of ``AgentMiddleware`` instances.

    Hook execution order:
    * ``before_model`` / ``before_tool``: first → last  (outermost first)
    * ``after_model``  / ``after_tool`` : last → first  (onion unwinding)

    Virtual tool dispatch (``handle_tool``) tries each middleware in order
    and returns the first non-``None`` result.
    """

    def __init__(self, middlewares: list[AgentMiddleware] | None = None):
        self.middlewares: list[AgentMiddleware] = list(middlewares or [])

    def add(self, middleware: AgentMiddleware) -> "MiddlewareChain":
        """Append a middleware to the chain (builder pattern)."""
        self.middlewares.append(middleware)
        return self

    # ── lifecycle ───────────────────────────────────────────────

    async def initialize(self, state: AgentState) -> None:
        for mw in self.middlewares:
            try:
                await mw.initialize(state)
            except Exception:
                logger.exception("Middleware %s.initialize() failed", mw.name)

    async def shutdown(self, state: AgentState) -> None:
        for mw in reversed(self.middlewares):
            try:
                await mw.shutdown(state)
            except Exception:
                logger.exception("Middleware %s.shutdown() failed", mw.name)

    # ── per-turn hooks ──────────────────────────────────────────

    async def before_model(self, state: AgentState) -> AgentState:
        for mw in self.middlewares:
            state = await mw.before_model(state)
        return state

    async def after_model(self, state: AgentState, response: Any) -> Any:
        for mw in reversed(self.middlewares):
            response = await mw.after_model(state, response)
        return response

    async def before_tool(
        self,
        state: AgentState,
        tool_name: str,
        args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        for mw in self.middlewares:
            tool_name, args = await mw.before_tool(state, tool_name, args)
        return tool_name, args

    async def after_tool(
        self,
        state: AgentState,
        tool_name: str,
        result: str,
    ) -> str:
        for mw in reversed(self.middlewares):
            result = await mw.after_tool(state, tool_name, result)
        return result

    # ── virtual tool dispatch ───────────────────────────────────

    async def handle_tool(
        self,
        state: AgentState,
        tool_name: str,
        args: dict[str, Any],
    ) -> str | None:
        """Try each middleware for a virtual tool handler.

        Returns the first non-``None`` result, or ``None`` if no middleware
        handles the tool (meaning it should go to the MCP ToolRegistry).
        """
        for mw in self.middlewares:
            result = await mw.handle_tool(state, tool_name, args)
            if result is not None:
                return result
        return None
