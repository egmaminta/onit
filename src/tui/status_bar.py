"""Status bar widget for Onit TUI."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static

from .mode import OnitMode

# Spinner frames (Claude Code style)
_SPINNER_CHARS = ["·", "✻", "✽", "✶", "✳", "✢"]

# Lobby messages that cycle while the agent is thinking
_THINKING_MESSAGES = [
    "Thinking",
    "On it",
    "Working on it",
    "Reasoning",
    "Processing",
    "Analyzing",
]


class ModelLabel(Widget):
    """Right-aligned model name with smart truncation.

    Drops provider prefix first, then left-truncates with ellipsis.
    """

    provider: reactive[str] = reactive("", layout=True)
    model: reactive[str] = reactive("", layout=True)

    def render(self) -> Text:
        width = self.content_size.width
        if not self.model or width <= 0:
            return Text("")
        full = f"{self.provider}:{self.model}" if self.provider else self.model
        if len(full) <= width:
            return Text(full, no_wrap=True, justify="right")
        if len(self.model) <= width:
            return Text(self.model, no_wrap=True, justify="right")
        if width > 1:
            return Text("\u2026" + self.model[-(width - 1):], no_wrap=True, justify="right")
        return Text("\u2026", no_wrap=True, justify="right")


class StatusBar(Horizontal):
    """Bottom status bar showing mode, streaming, cwd, tokens, and model."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: transparent;
        padding: 0 1;
    }

    StatusBar .sb-mode {
        width: auto;
        padding: 0 1;
    }

    StatusBar .sb-mode.agent {
        background: #10b981;
        color: black;
        text-style: bold;
    }

    StatusBar .sb-mode.plan {
        background: #f59e0b;
        color: black;
        text-style: bold;
    }

    StatusBar .sb-stream {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    StatusBar .sb-status {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    StatusBar .sb-status.thinking {
        color: #C32148;
    }

    StatusBar .sb-spacer {
        width: 1fr;
        min-width: 0;
        height: 1;
    }

    StatusBar .sb-cwd {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    StatusBar .sb-tokens {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }

    StatusBar ModelLabel {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }
    """

    mode: reactive[str] = reactive(OnitMode.AGENT.value, init=False)
    status_message: reactive[str] = reactive("", init=False)
    cwd: reactive[str] = reactive("", init=False)
    tokens: reactive[int] = reactive(0, init=False)
    max_tokens: reactive[int] = reactive(0, init=False)
    stream_enabled: reactive[bool] = reactive(False, init=False)

    def __init__(
        self,
        *,
        model_provider: str = "",
        model_name: str = "",
        stream_enabled: bool = False,
        max_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_provider = model_provider
        self._model_name_init = model_name
        self._initial_cwd = str(Path.cwd())
        self.stream_enabled = stream_enabled
        self.max_tokens = max_tokens
        self._spinner_timer: Timer | None = None
        self._spinner_frame: int = 0
        self._thinking_message: str = ""
        self._thinking_active: bool = False

    def compose(self):
        yield Static("AGENT", classes="sb-mode agent", id="sb-mode")
        stream_label = "stream" if self.stream_enabled else "no-stream"
        yield Static(stream_label, classes="sb-stream", id="sb-stream")
        yield Static("", classes="sb-status", id="sb-status")
        yield Static("", classes="sb-spacer")
        yield Static(self._format_cwd(self._initial_cwd), classes="sb-cwd", id="sb-cwd")
        yield Static("0 tokens", classes="sb-tokens", id="sb-tokens")
        label = ModelLabel(id="sb-model")
        label.provider = self._model_provider
        label.model = self._model_name_init
        yield label

    def _format_cwd(self, path: str) -> str:
        home = os.path.expanduser("~")
        if path.startswith(home):
            return path.replace(home, "~", 1)
        return path

    def _format_tokens(self, count: int) -> str:
        if count >= 1000:
            return f"{count / 1000:.1f}K tokens"
        return f"{count} tokens"

    # ── Reactive watchers ──────────────────────────────────────────

    def watch_mode(self, value: str) -> None:
        try:
            indicator = self.query_one("#sb-mode", Static)
        except Exception:
            return
        indicator.update(value.upper())
        indicator.remove_class("agent", "plan")
        indicator.add_class(value)

    def watch_status_message(self, value: str) -> None:
        try:
            widget = self.query_one("#sb-status", Static)
        except Exception:
            return
        widget.update(value)
        if value:
            widget.add_class("thinking")
        else:
            widget.remove_class("thinking")

    def watch_cwd(self, value: str) -> None:
        try:
            widget = self.query_one("#sb-cwd", Static)
        except Exception:
            return
        widget.update(self._format_cwd(value))

    def watch_tokens(self, value: int) -> None:
        try:
            widget = self.query_one("#sb-tokens", Static)
        except Exception:
            return
        if self.max_tokens > 0:
            pct = (value / self.max_tokens) * 100
            total_k = f"{value / 1000:.1f}K" if value >= 1000 else str(value)
            max_k = f"{self.max_tokens / 1000:.0f}K" if self.max_tokens >= 1000 else str(self.max_tokens)
            widget.update(f"{pct:.0f}% ({total_k}/{max_k})")
        else:
            widget.update(self._format_tokens(value))

    def watch_stream_enabled(self, value: bool) -> None:
        try:
            widget = self.query_one("#sb-stream", Static)
        except Exception:
            return
        widget.update("stream" if value else "no-stream")

    # ── Public API ─────────────────────────────────────────────────

    def set_mode(self, mode: OnitMode) -> None:
        self.mode = mode.value

    def set_status(self, message: str) -> None:
        if message:
            self.start_spinner(message)
        else:
            self.stop_spinner()

    def start_spinner(self, message: str = "Thinking") -> None:
        """Start the animated thinking spinner."""
        self._thinking_active = True
        self._thinking_message = message
        self._spinner_frame = 0
        self._update_spinner_display()
        if self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.15, self._tick_spinner)

    def stop_spinner(self) -> None:
        """Stop the spinner and clear status."""
        self._thinking_active = False
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
        self.status_message = ""

    def _tick_spinner(self) -> None:
        """Advance to next spinner frame."""
        self._spinner_frame += 1
        # Occasionally swap the lobby message
        if self._spinner_frame % 20 == 0:
            self._thinking_message = random.choice(_THINKING_MESSAGES)
        self._update_spinner_display()

    def _update_spinner_display(self) -> None:
        """Render the current spinner frame into the status widget."""
        char = _SPINNER_CHARS[self._spinner_frame % len(_SPINNER_CHARS)]
        self.status_message = f"{char} {self._thinking_message}…"

    def set_tokens(self, count: int) -> None:
        self.tokens = count

    def set_model(self, provider: str, name: str) -> None:
        try:
            label = self.query_one("#sb-model", ModelLabel)
        except Exception:
            return
        label.provider = provider
        label.model = name
