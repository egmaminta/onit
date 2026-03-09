"""EventBus — append-only JSONL observability log.

Provides structured event logging for worktree lifecycle, task state
changes, and other system events.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EventBus:
    """Append-only JSONL event log."""

    def __init__(self, events_dir: str = ".worktrees") -> None:
        self._dir = events_dir
        self._path = os.path.join(events_dir, "events.jsonl")
        self._lock = threading.Lock()
        os.makedirs(events_dir, exist_ok=True)

    def emit(
        self,
        event: str,
        task: int | str | None = None,
        worktree: str | None = None,
        error: str | None = None,
        extra: dict | None = None,
    ) -> None:
        """Append a structured event to the log."""
        entry: dict[str, Any] = {
            "event": event,
            "timestamp": time.time(),
        }
        if task is not None:
            entry["task"] = task
        if worktree is not None:
            entry["worktree"] = worktree
        if error is not None:
            entry["error"] = error
        if extra:
            entry["extra"] = extra

        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        logger.debug("Event: %s", event)

    def list_recent(self, limit: int = 20) -> str:
        """Return last N events as formatted text."""
        if not os.path.isfile(self._path):
            return "No events recorded."

        lines: list[str] = []
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)

        # Take last N
        recent = lines[-limit:]
        result = []
        for raw in recent:
            try:
                entry = json.loads(raw)
                ts = time.strftime("%H:%M:%S", time.localtime(entry.get("timestamp", 0)))
                event = entry.get("event", "?")
                parts = [f"[{ts}] {event}"]
                if entry.get("task"):
                    parts.append(f"task={entry['task']}")
                if entry.get("worktree"):
                    parts.append(f"wt={entry['worktree']}")
                if entry.get("error"):
                    parts.append(f"error={entry['error']}")
                result.append(" ".join(parts))
            except json.JSONDecodeError:
                pass

        return "\n".join(result) if result else "No events recorded."
