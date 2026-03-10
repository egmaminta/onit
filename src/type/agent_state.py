"""AgentState — shared state passed through the middleware chain and agent loop.

This is a plain TypedDict (not Pydantic) to keep it lightweight in the hot
loop.  Middleware and the AgentLoop read/mutate fields directly.
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Mutable state bag threaded through every middleware hook and agent turn."""

    # ── conversation ────────────────────────────────────────────
    messages: list[dict[str, Any]]
    """OpenAI-format message list (system, user, assistant, tool)."""

    system_prompt: str
    """System-level instruction injected as the first message."""

    tools: list[dict[str, Any]]
    """OpenAI-format tool definitions available for the current turn."""

    # ── iteration tracking ──────────────────────────────────────
    iteration: int
    """Current iteration count of the agent loop (1-based)."""

    max_iterations: int
    """Hard cap on loop iterations (default 100)."""

    max_repeated_tool_calls: int
    """Threshold for repeated identical tool calls before aborting."""

    # ── todo tracking (in-memory scratch-pad) ───────────────────
    todos: list[dict[str, Any]]
    """Quick in-memory todo items managed by TodoListMiddleware."""

    # ── middleware / plugin state ────────────────────────────────
    metadata: dict[str, Any]
    """Arbitrary middleware-owned state (skills cache, memory contents, …)."""

    # ── configuration ───────────────────────────────────────────
    config: dict[str, Any]
    """Resolved OnIt configuration dict (serving, middleware, paths, …)."""

    data_path: str
    """Per-session data directory for file outputs."""

    session_path: str
    """Path to the JSONL session history file."""

    # ── safety / control ────────────────────────────────────────
    safety_queue: asyncio.Queue
    """Queue checked each turn — a non-empty value aborts the loop."""

    # ── backend (Phase 2) ───────────────────────────────────────
    backend: Any
    """Optional BackendProtocol instance for filesystem/sandbox ops."""

    # ── UI / streaming ──────────────────────────────────────────
    chat_ui: Any
    """ChatUI / StreamingAdapter / None — for streaming tokens to the user."""

    verbose: bool
    """Whether to print debug information to stdout."""

    # ── model configuration ─────────────────────────────────────
    host: str
    """OpenAI-compatible API base URL."""

    host_key: str
    """API key for the model host."""

    model: str
    """Resolved model identifier."""

    think: bool
    """Whether to enable chain-of-thought / reasoning tokens."""

    stream: bool
    """Whether to use streaming completions."""

    max_tokens: int
    """Maximum tokens per completion."""

    timeout: int | None
    """Per-tool-call timeout in seconds (None = no timeout)."""

    prompt_intro: str
    """Short identity string prepended to system prompt."""

    # ── tool call history (for repeated-call detection) ─────────
    tool_call_history: list[tuple[str, str]]
    """List of (tool_name, args_json) seen so far in this loop."""
