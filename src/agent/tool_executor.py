"""ToolExecutor — unified tool execution for the OnIt agent loop.

Extracted and deduplicated from the two tool-execution paths that were
previously inlined in ``chat()``.  Handles:
* Tool lookup in the MCP ``ToolRegistry``
* Virtual tool dispatch via the ``MiddlewareChain``
* Timeout protection
* Base64 file extraction from tool responses
* Image stripping from stale messages
* ChatUI notification hooks (spinners, results, logs)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
except (ImportError, ValueError):
    from type.agent_state import AgentState

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Helpers (moved from chat.py)
# ────────────────────────────────────────────────────────────────

def extract_base64_file(tool_response: str, data_path: str) -> tuple[str, str | None, str | None]:
    """Detect base64-encoded file data in a tool response and save it to disk.

    Returns ``(cleaned_json_str, image_base64_or_None, mime_type_or_None)``.
    When the file is an image the base64 data and mime_type are returned so
    callers can inject the image into the conversation for VLM processing.
    """
    try:
        data = json.loads(tool_response)
    except (json.JSONDecodeError, TypeError):
        return tool_response, None, None

    if not isinstance(data, dict) or "file_data_base64" not in data:
        return tool_response, None, None

    file_data_b64 = data.pop("file_data_base64")
    mime_type = data.get("mime_type", "application/octet-stream")

    _ext_map = {
        "image/jpeg": ".jpg", "image/png": ".png",
        "image/gif": ".gif", "image/webp": ".webp",
    }
    ext = _ext_map.get(mime_type, ".bin")
    file_name = data.get("file_name", f"{uuid.uuid4()}{ext}")
    safe_name = os.path.basename(file_name)
    filepath = os.path.join(data_path, safe_name)
    os.makedirs(data_path, exist_ok=True)

    file_bytes = base64.b64decode(file_data_b64)
    fd = os.open(filepath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(file_bytes)

    data["saved_path"] = filepath
    data["download_url"] = f"/uploads/{safe_name}"
    data["file_size_bytes"] = len(file_bytes)

    image_b64 = file_data_b64 if mime_type.startswith("image/") else None
    return json.dumps(data), image_b64, mime_type if image_b64 else None


def strip_old_images(messages: list[dict]) -> None:
    """Replace base64 image payloads in stale tool messages with a placeholder.

    Keeps only the most-recently-added image-bearing tool message intact.
    """
    last_image_idx = -1
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
            if any(part.get("type") == "image_url" for part in msg["content"]):
                last_image_idx = i

    if last_image_idx == -1:
        return

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if i == last_image_idx:
            continue
        if msg.get("role") == "tool" and isinstance(msg.get("content"), list):
            if any(part.get("type") == "image_url" for part in msg["content"]):
                text = next(
                    (p["text"] for p in msg["content"] if p.get("type") == "text"), ""
                )
                msg["content"] = text + "\n[image omitted — already analyzed]"


# ────────────────────────────────────────────────────────────────
# ToolExecutor
# ────────────────────────────────────────────────────────────────

class ToolExecutor:
    """Execute a single tool call — virtual (middleware) or MCP."""

    def __init__(
        self,
        tool_registry: Any,
        middleware_chain: Any | None = None,
    ):
        self.tool_registry = tool_registry
        self.middleware_chain = middleware_chain

    async def execute(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any]:
        """Execute a tool and return the tool message dict for the conversation.

        Steps:
        1. Run ``middleware_chain.before_tool`` hooks
        2. Try ``middleware_chain.handle_tool`` (virtual tools)
        3. Fallback to MCP ToolRegistry lookup
        4. Run ``middleware_chain.after_tool`` hooks
        5. Handle base64 file extraction / vision injection

        Returns a dict ready to append to ``state['messages']``.
        """
        chat_ui = state.get("chat_ui")
        verbose = state.get("verbose", False)
        data_path = state.get("data_path", "")
        timeout = state.get("timeout")

        # UI: show tool start
        if chat_ui:
            chat_ui.add_tool_call(tool_name, tool_args)
            chat_ui.show_tool_start(tool_name, tool_args)
            chat_ui.start_tool_spinner(tool_name, tool_args)
        elif verbose:
            print(f"{tool_name}({tool_args})")

        # 1. before_tool hooks
        if self.middleware_chain:
            tool_name, tool_args = await self.middleware_chain.before_tool(
                state, tool_name, tool_args
            )

        tool_response: str = ""
        success = True
        try:
            # 2. Try virtual tool dispatch (middleware-owned tools)
            virtual_result: str | None = None
            if self.middleware_chain:
                virtual_result = await self.middleware_chain.handle_tool(
                    state, tool_name, tool_args
                )

            if virtual_result is not None:
                tool_response = virtual_result
            else:
                # 3. MCP ToolRegistry fallback
                tool_response = await self._execute_mcp_tool(
                    tool_name, tool_args, timeout, chat_ui, verbose
                )

        except Exception as e:
            tool_response = f"Error: {e}"
            success = False
            if chat_ui:
                chat_ui.stop_tool_spinner()
                chat_ui.show_tool_done(tool_name, str(e), success=False)
                chat_ui.add_log(f"{tool_name} error: {e}", level="error")
            elif verbose:
                print(f"{tool_name} encountered an error: {e}")

        # 4. after_tool hooks
        if self.middleware_chain and success:
            tool_response = await self.middleware_chain.after_tool(
                state, tool_name, tool_response
            )

        # 5. Handle base64 file data / vision injection
        _vision_b64, _vision_mime = None, None
        if data_path and "file_data_base64" in tool_response:
            tool_response, _vision_b64, _vision_mime = extract_base64_file(
                tool_response, data_path
            )

        if _vision_b64:
            tool_content: Any = [
                {"type": "text", "text": tool_response},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{_vision_mime};base64,{_vision_b64}"
                    },
                },
            ]
        else:
            tool_content = tool_response

        # UI: show result
        if chat_ui and success:
            chat_ui.stop_tool_spinner()
            chat_ui.add_tool_result(tool_name, tool_response)
            chat_ui.show_tool_done(tool_name, tool_response)
        elif verbose and success:
            truncated = (
                tool_response[:500] + "..."
                if len(tool_response) > 500
                else tool_response
            )
            print(f"{tool_name} returned: {truncated}")

        return {
            "role": "tool",
            "content": tool_content,
            "name": tool_name,
            "parameters": tool_args,
            "tool_call_id": tool_call_id,
        }

    # ── MCP tool execution ──────────────────────────────────────

    async def _execute_mcp_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        timeout: int | None,
        chat_ui: Any,
        verbose: bool,
    ) -> str:
        """Look up and call a tool via the MCP ToolRegistry."""
        if tool_name not in self.tool_registry.tools:
            return f"Error: tool {tool_name} not found"

        tool_handler = self.tool_registry[tool_name]
        try:
            tool_response = await asyncio.wait_for(
                tool_handler(**tool_args),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            msg = (
                f"- tool call timed out after {timeout} seconds. "
                "Tool might have succeeded but no response was received. "
                "Check expected output."
            )
            if chat_ui:
                chat_ui.add_log(f"{tool_name} timed out after {timeout}s", level="warning")
            elif verbose:
                print(f"{tool_name} timed out after {timeout}s")
            return msg

        if chat_ui:
            chat_ui.stop_tool_spinner()

        return "" if tool_response is None else str(tool_response)
