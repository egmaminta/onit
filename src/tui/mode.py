"""Plan/Agent mode system for Onit TUI.

Agent mode: Full access to all tools (default).
Plan mode: Read-only tools allowed; write/execute tools blocked.
"""

from enum import Enum


class OnitMode(str, Enum):
    AGENT = "agent"
    PLAN = "plan"


# Tools that are blocked in plan mode (write/execute operations).
# Any tool whose name contains one of these substrings will be filtered out.
PLAN_MODE_BLOCKED_TOOLS = frozenset({
    "bash",
    "shell",
    "execute",
    "write_file",
    "create_file",
    "edit_file",
    "delete",
    "remove",
    "move_file",
    "rename_file",
    "mkdir",
    "create_directory",
})


def is_tool_allowed(tool_name: str, mode: OnitMode) -> bool:
    """Check if a tool is allowed in the given mode.

    In AGENT mode, all tools are allowed.
    In PLAN mode, tools matching PLAN_MODE_BLOCKED_TOOLS are blocked.
    """
    if mode == OnitMode.AGENT:
        return True
    name_lower = tool_name.lower()
    return not any(blocked in name_lower for blocked in PLAN_MODE_BLOCKED_TOOLS)


PLAN_MODE_PROMPT_PREFIX = (
    "You are in PLAN mode. You must NOT execute, modify, create, or delete anything. "
    "Only analyze, research, strategize, and produce a detailed step-by-step plan. "
    "You may use search and read tools to gather information, but you must not run "
    "any commands, write any files, or make any changes to the filesystem. "
    "Output a clear, numbered plan with specific action items."
)
