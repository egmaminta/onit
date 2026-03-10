"""Protocol handlers — Shutdown and Plan Approval protocols.

These protocols facilitate structured coordination between lead and
teammate agents via the MessageBus.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import logging
import threading
import uuid
from enum import Enum
from typing import Any

try:
    from .message_bus import MessageBus
except (ImportError, ValueError):
    from agent.message_bus import MessageBus

logger = logging.getLogger(__name__)


class RequestStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ShutdownProtocol:
    """Request-response shutdown coordination."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self._requests: dict[str, dict] = {}  # request_id → {target, status}
        self._lock = threading.Lock()

    def request_shutdown(self, sender: str, target: str) -> str:
        """Lead requests a teammate to shut down. Returns request_id."""
        request_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._requests[request_id] = {
                "target": target,
                "status": RequestStatus.PENDING,
            }
            self._bus.send(
                sender=sender,
                to=target,
                content=f"Shutdown requested (request_id={request_id})",
                msg_type="shutdown_request",
                extra={"request_id": request_id},
            )
        return request_id

    def respond_shutdown(self, sender: str, lead: str, request_id: str, approve: bool) -> str:
        """Teammate responds to shutdown request."""
        status = "approved" if approve else "rejected"
        self._bus.send(
            sender=sender,
            to=lead,
            content=f"Shutdown {status} (request_id={request_id})",
            msg_type="shutdown_response",
            extra={"request_id": request_id, "approve": approve},
        )
        return f"Shutdown {status}"

    def check_status(self, request_id: str) -> str:
        """Check status of a shutdown request."""
        with self._lock:
            req = self._requests.get(request_id)
            if not req:
                return f"Unknown request ID: {request_id}"
            return f"Shutdown request {request_id}: target={req['target']}, status={req['status'].value}"

    def process_response(self, request_id: str, approve: bool) -> None:
        """Update internal tracking based on a response."""
        with self._lock:
            if request_id in self._requests:
                self._requests[request_id]["status"] = (
                    RequestStatus.APPROVED if approve else RequestStatus.REJECTED
                )


class PlanApprovalProtocol:
    """Teammate submits plans for lead approval."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self._plans: dict[str, dict] = {}  # request_id → {sender, plan, status, feedback}
        self._lock = threading.Lock()

    def submit_plan(self, sender: str, lead: str, plan: str) -> str:
        """Teammate submits a plan for approval. Returns request_id."""
        request_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._plans[request_id] = {
                "sender": sender,
                "plan": plan,
                "status": RequestStatus.PENDING,
                "feedback": "",
            }
            self._bus.send(
                sender=sender,
                to=lead,
                content=f"Plan for approval (request_id={request_id}):\n{plan}",
                msg_type="plan_approval",
                extra={"request_id": request_id, "plan": plan},
            )
        return request_id

    def review_plan(
        self, lead: str, request_id: str, approve: bool, feedback: str = ""
    ) -> str:
        """Lead reviews a plan submission."""
        with self._lock:
            plan_info = self._plans.get(request_id)
            if not plan_info:
                return f"Unknown plan request: {request_id}"
            plan_info["status"] = RequestStatus.APPROVED if approve else RequestStatus.REJECTED
            plan_info["feedback"] = feedback
            sender = plan_info["sender"]

            status = "approved" if approve else "rejected"
            self._bus.send(
                sender=lead,
                to=sender,
                content=f"Plan {status}: {feedback}" if feedback else f"Plan {status}",
                msg_type="plan_approval_response",
                extra={
                    "request_id": request_id,
                    "approve": approve,
                    "feedback": feedback,
                },
            )
        return f"Plan {request_id} {status}"

    def check_plan(self, request_id: str) -> str:
        """Check status of a plan submission."""
        with self._lock:
            plan = self._plans.get(request_id)
            if not plan:
                return f"Unknown plan request: {request_id}"
            return (
                f"Plan {request_id}: sender={plan['sender']}, "
                f"status={plan['status'].value}, feedback={plan['feedback']}"
            )
