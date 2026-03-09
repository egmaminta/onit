"""Message display widgets for Onit TUI.

Provides UserMessage, AssistantMessage, ToolCallMessage, and ToolResultMessage
widgets for rendering chat history in the Textual UI.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


def _timestamp() -> str:
    return datetime.now().strftime("%I:%M %p")


class UserMessage(Static):
    """Display widget for a user message."""

    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        padding: 0 1;
        margin: 1 0 1 0;
        border-left: thick $primary;
    }
    """

    def __init__(self, text: str, **kwargs: Any) -> None:
        self._text = text
        self._time = _timestamp()
        super().__init__(self._build(), **kwargs)

    def _build(self) -> Text:
        t = Text()
        t.append(f"👤 You ", style="bold cyan")
        t.append(f"[{self._time}]\n", style="dim")
        t.append(self._text, style="white")
        return t


class AssistantMessage(Static):
    """Display widget for an assistant message with streaming support."""

    DEFAULT_CSS = """
    AssistantMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 0 0;
        border-left: thick #C32148;
    }
    """

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        self._content = text
        self._time = _timestamp()
        self._elapsed = ""
        self._tag_buf = ""
        super().__init__(self._build(), **kwargs)

    def _filter_tags(self, text: str) -> str:
        """Strip <answer></answer> wrapper tags."""
        return text.replace("<answer>", "").replace("</answer>", "")

    def _build(self) -> Text:
        t = Text()
        t.append("🤖 • 💡 OnIt ", style="bold #C32148")
        t.append(f"[{self._time}]", style="dim")
        if self._elapsed:
            t.append(f" — {self._elapsed}", style="dim")
        t.append("\n")
        display = self._filter_tags(self._content).strip()
        t.append(display, style="white")
        return t

    def append_content(self, token: str) -> None:
        """Append a streamed answer token and re-render."""
        # Buffer partial tags across tokens
        buf = self._tag_buf + token
        buf = buf.replace("<answer>", "").replace("</answer>", "")
        if buf.endswith("<"):
            self._tag_buf = "<"
            buf = buf[:-1]
        else:
            self._tag_buf = ""
        self._content += buf
        self.update(self._build())

    def set_content(self, text: str) -> None:
        """Set the full content (non-streaming path)."""
        self._content = text
        self.update(self._build())

    def set_elapsed(self, elapsed: str) -> None:
        """Set the elapsed time and re-render."""
        self._elapsed = elapsed
        self.update(self._build())

    def get_content(self) -> str:
        """Return the raw content (without tags)."""
        return self._filter_tags(self._content)


class ThinkingMessage(Static):
    """Display widget for model thinking/reasoning tokens."""

    DEFAULT_CSS = """
    ThinkingMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 0 0;
        border-left: thick #6b7280;
    }
    """

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        self._content = text
        self._time = _timestamp()
        self._finished = False
        super().__init__(self._build(), **kwargs)

    def _build(self) -> Text:
        t = Text()
        t.append("\U0001F916 \u2022 \U0001F9E0 OnIt ", style="bold #6b7280")
        t.append(f"[{self._time}]\n", style="dim")
        t.append(self._content, style="dim italic")
        return t

    def append_thinking(self, token: str) -> None:
        """Append a thinking token and re-render."""
        self._content += token
        self.update(self._build())

    def end_thinking(self) -> None:
        """Mark thinking phase as complete, stripping trailing whitespace."""
        self._finished = True
        self._content = self._content.rstrip()
        self.update(self._build())

    def set_content(self, text: str) -> None:
        """Set full thinking content (non-streaming path)."""
        self._content = text
        self._finished = True
        self.update(self._build())


class ToolCallMessage(Static):
    """Display widget for a tool invocation."""

    DEFAULT_CSS = """
    ToolCallMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 0 0;
        border-left: thick #B85A3C;
    }
    """

    def __init__(self, name: str, arguments: dict, **kwargs: Any) -> None:
        self._name = name
        self._arguments = arguments
        self._time = _timestamp()
        super().__init__(self._build(), **kwargs)

    def _build(self) -> Text:
        t = Text()
        t.append(f"⚙️  {self._name} ", style="bold #B85A3C")
        t.append(f"[{self._time}]\n", style="dim")
        args_str = json.dumps(self._arguments, ensure_ascii=False)
        if len(args_str) > 200:
            args_str = args_str[:200] + "…"
        t.append(args_str, style="dim white")
        return t


class ToolResultMessage(Static):
    """Display widget for a tool result."""

    DEFAULT_CSS = """
    ToolResultMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 0 0;
        border-left: thick #DE7356;
    }
    """

    def __init__(self, name: str, result: str, success: bool = True, **kwargs: Any) -> None:
        self._name = name
        self._result = result
        self._success = success
        self._time = _timestamp()
        super().__init__(self._build(), **kwargs)

    def _build(self) -> Text:
        t = Text()
        icon = "✓" if self._success else "✗"
        style = "bold #DE7356" if self._success else "bold red"
        t.append(f"↩ {icon} {self._name} ", style=style)
        t.append(f"[{self._time}]\n", style="dim")
        # Never truncate todo list results
        if self._name == "write_todos":
            display = self._result
        else:
            display = self._result if len(self._result) <= 300 else self._result[:300] + "…"
        t.append(display, style="dim white")
        return t
