"""MessageBus — JSONL append-only inbox-based messaging between agents.

Team members communicate via per-agent inbox files stored as JSONL in
``.team/inbox/{name}.jsonl``.  Thread-safe with per-inbox locking.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

VALID_MSG_TYPES = frozenset({
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval",
    "plan_approval_response",
})


class MessageBus:
    """JSONL-based message bus with per-inbox files."""

    def __init__(self, team_dir: str = ".team") -> None:
        self._dir = os.path.join(team_dir, "inbox")
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        os.makedirs(self._dir, exist_ok=True)

    def _get_lock(self, name: str) -> threading.Lock:
        with self._global_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            return self._locks[name]

    def _inbox_path(self, name: str) -> str:
        # Validate name to prevent path traversal
        safe = "".join(c for c in name if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"Invalid inbox name: {name}")
        return os.path.join(self._dir, f"{safe}.jsonl")

    def send(
        self,
        sender: str,
        to: str,
        content: str,
        msg_type: str = "message",
        extra: dict | None = None,
    ) -> str:
        """Send a message to a recipient's inbox. Returns message ID."""
        if msg_type not in VALID_MSG_TYPES:
            raise ValueError(f"Invalid message type: {msg_type}. Valid: {VALID_MSG_TYPES}")

        msg_id = uuid.uuid4().hex[:12]
        msg = {
            "id": msg_id,
            "from": sender,
            "to": to,
            "type": msg_type,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg["extra"] = extra

        lock = self._get_lock(to)
        with lock:
            with open(self._inbox_path(to), "a", encoding="utf-8") as f:
                f.write(json.dumps(msg) + "\n")

        return msg_id

    def read_inbox(self, name: str) -> list[dict]:
        """Read all messages and drain (clear) the inbox."""
        lock = self._get_lock(name)
        with lock:
            path = self._inbox_path(name)
            if not os.path.isfile(path):
                return []
            messages = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning("Skipping malformed message in %s", name)
            # Drain: truncate the file
            with open(path, "w", encoding="utf-8") as f:
                pass
            return messages

    def peek_inbox(self, name: str) -> list[dict]:
        """Read messages without draining."""
        lock = self._get_lock(name)
        with lock:
            path = self._inbox_path(name)
            if not os.path.isfile(path):
                return []
            messages = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return messages

    def broadcast(self, sender: str, content: str, teammates: list[str]) -> str:
        """Send to all teammates except sender. Returns count."""
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, msg_type="broadcast")
                count += 1
        return f"Broadcast sent to {count} teammate(s)"
