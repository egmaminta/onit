"""TeammateManager — spawn and manage teammate agent loops.

Each teammate runs its own AgentLoop in a daemon thread with WORK and
IDLE phases, inbox checking, and auto-shutdown on idle timeout.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    from .message_bus import MessageBus
    from .protocols import ShutdownProtocol
except (ImportError, ValueError):
    from agent.message_bus import MessageBus
    from agent.protocols import ShutdownProtocol

logger = logging.getLogger(__name__)


class MemberStatus(str, Enum):
    STARTING = "starting"
    WORKING = "working"
    IDLE = "idle"
    STOPPED = "stopped"


@dataclass
class TeamMember:
    name: str
    role: str
    prompt: str
    status: MemberStatus = MemberStatus.STARTING
    thread: threading.Thread | None = field(default=None, repr=False)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    started_at: float = field(default_factory=time.time)


class TeammateManager:
    """Manages a team of agent loops running in daemon threads."""

    MAX_WORK_ITERATIONS = 50
    IDLE_POLL_INTERVAL = 5.0  # seconds
    IDLE_TIMEOUT = 60.0  # seconds

    def __init__(
        self,
        bus: MessageBus,
        shutdown_protocol: ShutdownProtocol,
        team_dir: str = ".team",
        agent_loop_factory: Any = None,
        tool_registry: Any = None,
        middleware_chain: Any = None,
        default_model: str = "",
        default_host: str = "",
        default_host_key: str = "",
    ) -> None:
        self._bus = bus
        self._shutdown = shutdown_protocol
        self._team_dir = team_dir
        self._agent_loop_factory = agent_loop_factory
        self._tool_registry = tool_registry
        self._middleware_chain = middleware_chain
        self._default_model = default_model
        self._default_host = default_host
        self._default_host_key = default_host_key
        self._members: dict[str, TeamMember] = {}
        self._lock = threading.Lock()
        os.makedirs(team_dir, exist_ok=True)

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """Spawn a new teammate. Returns confirmation."""
        with self._lock:
            if name in self._members:
                return f"Teammate '{name}' already exists (status={self._members[name].status.value})"

            member = TeamMember(name=name, role=role, prompt=prompt)
            self._members[name] = member

        # Save config
        self._save_config()

        # Start daemon thread
        t = threading.Thread(
            target=self._teammate_thread,
            args=(name,),
            daemon=True,
            name=f"teammate-{name}",
        )
        member.thread = t
        t.start()

        return f"Spawned teammate '{name}' (role: {role})"

    def _teammate_thread(self, name: str) -> None:
        """Entry point for teammate daemon thread — runs async loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._teammate_loop(name))
        except Exception as e:
            logger.error("Teammate '%s' crashed: %s", name, e)
        finally:
            loop.close()
            with self._lock:
                if name in self._members:
                    self._members[name].status = MemberStatus.STOPPED

    async def _teammate_loop(self, name: str) -> None:
        """Main loop for a teammate: WORK → IDLE → auto-shutdown."""
        member = self._members.get(name)
        if not member:
            return

        member.status = MemberStatus.WORKING

        # Build initial messages
        system_prompt = (
            f"You are '{name}', a team member with role: {member.role}.\n"
            f"Instructions: {member.prompt}\n\n"
            f"You can communicate with the team using send_message and read_inbox tools.\n"
            f"Check your inbox regularly for messages and instructions."
        )

        state = {
            "messages": [{"role": "system", "content": system_prompt}],
            "system_prompt": system_prompt,
            "tools": [],
            "iteration": 0,
            "max_iterations": self.MAX_WORK_ITERATIONS,
            "model": self._default_model,
            "host": self._default_host,
            "host_key": self._default_host_key,
            "stream": False,
            "verbose": False,
            "metadata": {"teammate_name": name, "is_teammate": True},
        }

        # Check if we have agent loop factory
        if self._agent_loop_factory is None:
            logger.warning("No agent_loop_factory for teammate '%s'", name)
            with self._lock:
                member.status = MemberStatus.STOPPED
            return

        agent_loop = self._agent_loop_factory(
            tool_registry=self._tool_registry,
            middleware_chain=self._middleware_chain,
        )

        # WORK phase
        try:
            await agent_loop.run(state)
        except Exception as e:
            logger.error("Teammate '%s' work phase error: %s", name, e)

        # IDLE phase — poll for messages
        member.status = MemberStatus.IDLE
        idle_start = time.time()

        while not member.stop_event.is_set():
            elapsed = time.time() - idle_start
            if elapsed > self.IDLE_TIMEOUT:
                logger.info("Teammate '%s' idle timeout", name)
                break

            # Check inbox
            messages = self._bus.peek_inbox(name)
            if messages:
                # New messages — re-enter WORK phase
                member.status = MemberStatus.WORKING
                state["iteration"] = 0
                state["messages"].append({
                    "role": "user",
                    "content": f"[Inbox check] You have {len(messages)} new message(s). Use read_inbox to process them."
                })
                try:
                    await agent_loop.run(state)
                except Exception as e:
                    logger.error("Teammate '%s' resumed work error: %s", name, e)
                member.status = MemberStatus.IDLE
                idle_start = time.time()
            else:
                await asyncio.sleep(self.IDLE_POLL_INTERVAL)

        member.status = MemberStatus.STOPPED
        logger.info("Teammate '%s' stopped", name)

    def list_all(self) -> list[dict]:
        """List all team members with status."""
        result = []
        for name, member in self._members.items():
            result.append({
                "name": name,
                "role": member.role,
                "status": member.status.value,
                "uptime": f"{time.time() - member.started_at:.0f}s",
            })
        return result

    def stop(self, name: str) -> str:
        """Signal a teammate to stop."""
        member = self._members.get(name)
        if not member:
            return f"Teammate '{name}' not found."
        member.stop_event.set()
        return f"Stop signal sent to '{name}'"

    def _save_config(self) -> None:
        """Save team config to disk."""
        config = {
            "members": [
                {"name": m.name, "role": m.role, "status": m.status.value}
                for m in self._members.values()
            ]
        }
        path = os.path.join(self._team_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
