"""OnIt agent core — loop, tool execution, tasks, teams, worktrees."""

from .loop import AgentLoop  # noqa: F401
from .tool_executor import ToolExecutor  # noqa: F401
from .tasks import TaskManager, Task, TaskStatus  # noqa: F401
from .event_bus import EventBus  # noqa: F401
from .message_bus import MessageBus  # noqa: F401
from .teammate import TeammateManager, TeamMember, MemberStatus  # noqa: F401
from .protocols import ShutdownProtocol, PlanApprovalProtocol, RequestStatus  # noqa: F401
from .worktree import WorktreeManager, WorktreeInfo  # noqa: F401

__all__ = [
    "AgentLoop",
    "ToolExecutor",
    "TaskManager",
    "Task",
    "TaskStatus",
    "EventBus",
    "MessageBus",
    "TeammateManager",
    "TeamMember",
    "MemberStatus",
    "ShutdownProtocol",
    "PlanApprovalProtocol",
    "RequestStatus",
    "WorktreeManager",
    "WorktreeInfo",
]
