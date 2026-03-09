"""SummarizationMiddleware — multi-tier context compaction.

Tier 1 (Micro-compact): every turn, replace old tool results with short
placeholders while keeping the last N intact.

Tier 2 (Auto-compact): when estimated tokens exceed a threshold, save
the full transcript and replace messages with a summary + recent tail.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _messages_text(messages: list) -> str:
    """Concatenate all message content for token estimation."""
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        parts.append(part.get("text", ""))
    return "\n".join(parts)


COMPACT_COOLDOWN = 2  # turns before re-compaction is allowed


class SummarizationMiddleware(AgentMiddleware):
    """Multi-tier context compaction."""

    def __init__(
        self,
        keep_last_n: int = 3,
        token_threshold: int = 50_000,
        tail_fraction: float = 0.1,
        context_ratio: float = 0.85,
    ) -> None:
        self.keep_last_n = keep_last_n
        self._configured_threshold = token_threshold
        self.tail_fraction = tail_fraction
        self.context_ratio = context_ratio
        self._turns_since_compact: int | None = None  # None = never compacted
        self._resolved = False
        self.token_threshold = token_threshold

    def _resolve_threshold(self, state) -> None:
        """Set threshold to 85% of max_tokens from the agent state on first call."""
        if self._resolved:
            return
        self._resolved = True
        max_tokens = state.get("max_tokens", 0)
        if max_tokens and max_tokens > 0:
            dynamic = int(max_tokens * self.context_ratio)
            # Use the higher of dynamic vs configured so config can override
            self.token_threshold = max(dynamic, self._configured_threshold)
            logger.info("Summarization threshold: %d tokens (%.0f%% of %d)",
                        self.token_threshold, self.context_ratio * 100, max_tokens)

    async def before_model(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])

        # Resolve dynamic threshold on first call
        self._resolve_threshold(state)

        # Tier 1: Micro-compact — replace old tool results with placeholders
        messages = self._micro_compact(messages)

        # Tier 2: Auto-compact — compress if tokens exceed threshold
        total_tokens = estimate_tokens(_messages_text(messages))
        if total_tokens > self.token_threshold:
            if self._turns_since_compact is None or self._turns_since_compact >= COMPACT_COOLDOWN:
                messages = await self._auto_compact(messages, state)
                self._turns_since_compact = 0
            else:
                self._turns_since_compact += 1
        else:
            # Reset once tokens drop below threshold so future
            # accumulation can trigger compaction again.
            self._turns_since_compact = None

        state["messages"] = messages
        return state

    def _micro_compact(self, messages: list) -> list:
        """Keep last N tool results intact, replace older ones with placeholders."""
        # Find indices of tool-result messages
        tool_indices = []
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "tool":
                tool_indices.append(i)

        if len(tool_indices) <= self.keep_last_n:
            return messages

        # Indices to compact (all but the last N)
        to_compact = set(tool_indices[:-self.keep_last_n])

        for i in to_compact:
            msg = messages[i]
            tool_name = msg.get("name", "tool")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part content (e.g., with images) — strip to text placeholder
                msg["content"] = f"[Previously used {tool_name}]"
            elif isinstance(content, str) and len(content) > 200:
                msg["content"] = f"[Previously used {tool_name}]"

        return messages

    async def _auto_compact(self, messages: list, state: AgentState) -> list:
        """Save transcript and compress conversation."""
        data_path = state.get("data_path", "")

        # Save full transcript
        if data_path:
            transcript_dir = os.path.join(data_path, ".transcripts")
            os.makedirs(transcript_dir, exist_ok=True)
            transcript_path = os.path.join(transcript_dir, f"transcript_{int(time.time())}.jsonl")
            try:
                with open(transcript_path, "w", encoding="utf-8") as f:
                    for msg in messages:
                        if isinstance(msg, dict):
                            f.write(json.dumps(msg, default=str) + "\n")
                logger.info("Saved transcript to %s", transcript_path)
            except Exception as e:
                logger.warning("Failed to save transcript: %s", e)
                transcript_path = "unavailable"
        else:
            transcript_path = "unavailable"

        # Build summary of conversation so far
        summary_parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Just track the shape — actual LLM summarization would need
                # a model call which we defer to future enhancement
                if role == "user":
                    summary_parts.append(f"User: {content[:200]}")
                elif role == "assistant":
                    summary_parts.append(f"Assistant: {content[:200]}")
                elif role == "tool":
                    name = msg.get("name", "tool")
                    summary_parts.append(f"Tool({name}): {content[:100]}")

        summary = "\n".join(summary_parts[-30:])  # last 30 entries max

        # Keep system message + recent tail
        system_msg = messages[0] if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system" else None
        tail_count = max(4, int(len(messages) * self.tail_fraction))
        tail = messages[-tail_count:]

        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append({
            "role": "user",
            "content": f"[Compressed conversation. Transcript: {transcript_path}]\n\nSummary of prior context:\n{summary}",
        })
        new_messages.extend(tail)

        logger.info("Auto-compacted: %d messages → %d messages", len(messages), len(new_messages))
        return new_messages

    async def emergency_compact(self, state: AgentState) -> AgentState:
        """Aggressive compaction for token-limit recovery.

        Keeps only the system message and the last few messages.
        Called by the AgentLoop when an API token-limit error occurs.
        """
        messages = state.get("messages", [])
        if len(messages) <= 5:
            return state

        system = []
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            system = [messages[0]]

        tail = messages[-4:]
        new_messages = system + [
            {"role": "user", "content": "[Prior conversation truncated due to context length limit. Continuing from recent context.]"}
        ] + tail

        state["messages"] = new_messages
        self._turns_since_compact = 0
        logger.info("Emergency compacted: %d messages → %d messages", len(messages), len(new_messages))
        return state

    async def handle_tool(self, state: AgentState, tool_name: str, args: dict) -> Optional[str]:
        """Handle manual compress tool."""
        if tool_name != "compress":
            return None
        messages = state.get("messages", [])
        self._turns_since_compact = None  # allow re-compression
        new_messages = await self._auto_compact(messages, state)
        state["messages"] = new_messages
        self._turns_since_compact = 0
        return "Conversation compressed successfully."
