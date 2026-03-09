"""Onit TUI application — Textual-based interactive CLI.

This module provides:
- OnitApp: the main Textual application
- TuiChatUI: an adapter that implements the same interface as ChatUI
  so that onit.py and chat.py can work with the TUI without changes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.widgets import Static

from .banner import WelcomeBanner
from .chat_input import ChatSubmitted, OnitChatInput
from .messages import (
    AssistantMessage,
    ThinkingMessage,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)
from .mode import PLAN_MODE_PROMPT_PREFIX, OnitMode, is_tool_allowed
from .status_bar import StatusBar


# ── TuiChatUI adapter ─────────────────────────────────────────
# This class implements the same public interface as src.ui.text.ChatUI
# so that onit.py and chat.py can call stream_token(), add_message(), etc.
# It bridges those calls into Textual widget operations.

@dataclass
class _Message:
    """Lightweight mirror of src.ui.text.Message for compatibility."""
    role: str
    content: str
    timestamp: str
    elapsed: str = ""
    name: str = ""


class TuiChatUI:
    """Adapter between OnIt's ChatUI interface and the Textual TUI.

    onit.py / chat.py communicate with the UI through this object.
    Method calls are routed to the OnitApp via call_from_thread
    (since agent_session runs in an asyncio loop that may be different
    from Textual's internal loop).
    """

    def __init__(self, app: OnitApp) -> None:
        self._app = app
        self.messages: deque[_Message] = deque(maxlen=200)
        self.execution_logs: deque[dict] = deque(maxlen=100)
        self.console = None  # chat.py checks for this
        self._current_assistant: AssistantMessage | None = None
        self._current_thinking: ThinkingMessage | None = None
        self._stream_header_printed = False
        self._streaming_content = ""
        self._stream_pending = ""
        self._stream_think_started = False
        self._tag_buf = ""
        self._trail_buf = ""
        self._non_stream_reasoning: str | None = None  # set by chat.py in non-stream mode

    # ── Public interface (called by onit.py / chat.py) ─────────

    def get_user_input(self) -> str:
        """Block until the user submits a message (called from executor thread)."""
        future = asyncio.run_coroutine_threadsafe(
            self._app.wait_for_user_input(), self._app._loop
        )
        return future.result()

    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        response: str,
        elapsed: str = "",
    ) -> None:
        from datetime import datetime
        msg = _Message(
            role=role,
            content=response,
            timestamp=datetime.now().strftime("%I:%M %p"),
            elapsed=elapsed,
        )
        self.messages.append(msg)
        self._app.call_from_thread(self._app.mount_message, role, response, elapsed)

    def add_tool_call(self, name: str, arguments: dict) -> None:
        self._app.call_from_thread(self._app.mount_tool_call, name, arguments)

    def add_tool_result(self, name: str, result: str, truncate: int = 300) -> None:
        # Never truncate todo list updates — show the full state
        if name == "write_todos":
            display = result
        else:
            display = result if len(result) <= truncate else result[:truncate] + "…"
        success = True  # Default assumption
        self._app.call_from_thread(self._app.mount_tool_result, name, display, success)

    def add_log(self, message: str, level: str = "info") -> None:
        self.execution_logs.append({"message": message, "level": level})

    def report_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0) -> None:
        """Report token usage from the API response."""
        if total_tokens > 0:
            self._app._total_tokens = total_tokens
        else:
            self._app._total_tokens += prompt_tokens + completion_tokens
        self._app.call_from_thread(self._app.update_tokens, self._app._total_tokens)

    # ── Streaming interface (called by chat.py) ────────────────

    def stream_start(self) -> None:
        """Prepare for a new streamed response."""
        self._streaming_content = ""
        self._stream_header_printed = False
        self._stream_pending = ""
        self._stream_think_started = False
        self._tag_buf = ""
        self._trail_buf = ""
        self._current_assistant = None
        self._current_thinking = None
        # Don't create the assistant widget yet — it will be created
        # when the first answer token arrives (or when stream_end fires).

    def stream_think_token(self, token: str) -> None:
        if not self._app._show_thinking_tokens:
            return  # Suppress thinking tokens unless --show-thinking
        if self._current_thinking is None:
            self._app.call_from_thread(self._app._ensure_thinking_widget)
            self._current_thinking = self._app._current_thinking_widget
        self._stream_think_started = True
        if self._current_thinking:
            self._app.call_from_thread(self._current_thinking.append_thinking, token)

    def stream_think_end(self) -> None:
        if self._stream_think_started and self._current_thinking:
            self._app.call_from_thread(self._current_thinking.end_thinking)
            self._stream_think_started = False

    def stream_token(self, token: str) -> None:
        self._streaming_content += token
        if self._stream_think_started:
            self.stream_think_end()
        # Buffer tokens until we have visible (non-whitespace) content
        # to avoid briefly flashing an empty assistant widget when the
        # model emits whitespace before tool-call deltas.
        if self._current_assistant is None:
            if not self._streaming_content.strip():
                return  # still only whitespace — don't mount yet
            self._app.call_from_thread(self._app._ensure_assistant_widget)
            self._current_assistant = self._app._current_assistant_widget
            # Flush the buffered content, stripping leading/trailing whitespace
            if self._current_assistant:
                self._app.call_from_thread(
                    self._current_assistant.append_content, self._streaming_content.strip()
                )
            return
        if self._current_assistant:
            self._app.call_from_thread(self._current_assistant.append_content, token)

    def stream_end(self, elapsed: str = "") -> None:
        self.stream_think_end()
        if self._current_assistant:
            content = self._current_assistant.get_content()
            # Strip XML tags for storage
            content = re.sub(r"<[^>]+>", "", content).strip()
            if content:
                self._app.call_from_thread(self._current_assistant.set_elapsed, elapsed)
                from datetime import datetime
                self.messages.append(_Message(
                    role="assistant",
                    content=content,
                    timestamp=datetime.now().strftime("%I:%M %p"),
                    elapsed=elapsed,
                ))
            else:
                # Widget has no visible content (e.g. only whitespace
                # arrived before tool-call deltas) — remove the dead space.
                self._app.call_from_thread(self._current_assistant.remove)
        self._current_assistant = None
        self._app.call_from_thread(self._app._on_stream_end)

    # ── Tool display (called by chat.py in streaming mode) ─────

    def show_tool_start(self, name: str, arguments: dict) -> None:
        pass  # add_tool_call already displays the tool call

    def start_tool_spinner(self, name: str, arguments: dict) -> None:
        pass  # Widget border provides visual feedback

    def stop_tool_spinner(self) -> None:
        pass

    def show_tool_done(self, name: str, result: str, success: bool = True) -> None:
        pass  # add_tool_result already displays the result

    # ── Thinking spinner (called by onit.py) ───────────────────

    def start_thinking(self) -> None:
        self._app.call_from_thread(self._app._show_thinking)

    def stop_thinking(self) -> None:
        self._app.call_from_thread(self._app._hide_thinking)

    def start_status(self) -> None:
        self.start_thinking()

    def stop_status(self) -> None:
        self.stop_thinking()

    # ── Theme / initialize stubs ───────────────────────────────

    def set_theme(self, theme: str) -> None:
        pass  # Textual CSS handles theming

    def initialize(self) -> None:
        pass  # Banner handles welcome

    def render(self, thinking: bool = False):
        return None  # Not used in TUI mode

    def render_messages(self):
        return None

    def clear_messages(self, keep_last: int = 0) -> None:
        if keep_last > 0:
            msgs = list(self.messages)[-keep_last:]
            self.messages.clear()
            self.messages.extend(msgs)
        else:
            self.messages.clear()


# ── OnitApp ────────────────────────────────────────────────────

CSS_PATH = "app.tcss"


class OnitApp(App):
    """The main Onit Textual TUI application."""

    CSS_PATH = CSS_PATH

    BINDINGS = [
        Binding("escape", "interrupt", "Interrupt agent", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("shift+tab", "toggle_mode", "Toggle Plan/Agent mode", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear chat"),
    ]

    def __init__(
        self,
        onit: Any,
        *,
        stream_enabled: bool = False,
        show_thinking: bool = False,
        startup_warnings: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(ansi_color=True, **kwargs)
        self._onit = onit
        self._stream_enabled = stream_enabled
        self._show_thinking_tokens = show_thinking
        self._startup_warnings = startup_warnings or []
        self._mode = OnitMode.AGENT
        self._loop: asyncio.AbstractEventLoop | None = None
        self._user_input_event = threading.Event()
        self._user_input_text: str = ""
        self._current_assistant_widget: AssistantMessage | None = None
        self._current_thinking_widget: ThinkingMessage | None = None
        self._agent_running = False
        self._total_tokens = 0
        self._interrupt_flag = threading.Event()
        self._shutdown_flag = threading.Event()

        # Slash commands available to the user
        self._slash_commands = {
            "/exit": "Quit OnIt",
            "/clear": "Clear chat history",
            "/plan": "Switch to Plan mode",
            "/agent": "Switch to Agent mode",
            "/mode": "Toggle Plan/Agent mode",
            "/help": "Show available commands",
        }

        # Build the TuiChatUI adapter and attach it to OnIt
        self.tui_chat_ui = TuiChatUI(self)
        self._onit.chat_ui = self.tui_chat_ui

    def compose(self) -> ComposeResult:
        # Extract model info from OnIt config
        serving = self._onit.model_serving
        model_name = serving.get("model", "unknown")
        model_host = serving.get("host", "")
        max_tokens = serving.get("max_tokens", 0)

        # MCP server names
        mcp_names = [
            s.get("name", "Unknown")
            for s in self._onit.mcp_servers
            if s.get("enabled", True) and s.get("name") != "PromptsMCPServer"
        ]

        # Determine provider label for status bar
        provider = "local"
        if model_host:
            if "openrouter" in model_host.lower():
                provider = "openrouter"
            elif "localhost" in model_host or "127.0.0.1" in model_host:
                provider = "local"
            else:
                provider = "api"

        with VerticalScroll(id="chat"):
            yield WelcomeBanner(
                mcp_server_names=mcp_names,
                model_name=model_name,
                model_host=model_host,
                max_tokens=max_tokens,
                session_id=self._onit.session_id or "",
                stream_enabled=self._stream_enabled,
                id="welcome-banner",
            )
            yield Container(id="messages")

        yield OnitChatInput(id="input-area")
        yield StatusBar(
            model_provider=provider,
            model_name=model_name,
            stream_enabled=self._stream_enabled,
            max_tokens=max_tokens,
            id="status-bar",
        )

    async def on_mount(self) -> None:
        self._loop = asyncio.get_event_loop()
        # Display any startup warnings inside the TUI
        if self._startup_warnings:
            for warning in self._startup_warnings:
                self.mount_message("system", f"⚠ {warning}")
        # Start the agent loop as a background worker
        self._start_agent_loop()

    @work(thread=True)
    def _start_agent_loop(self) -> None:
        """Run the OnIt agent loop in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_agent())
        finally:
            loop.close()

    async def _run_agent(self) -> None:
        """The OnIt interactive loop, adapted from onit.client_to_agent."""
        onit = self._onit
        onit.input_queue = asyncio.Queue(maxsize=10)
        onit.output_queue = asyncio.Queue(maxsize=10)
        onit.safety_queue = asyncio.Queue(maxsize=10)
        onit.status = "running"

        from fastmcp import Client
        from ..model.serving.chat import chat

        prompt_client = Client(onit.prompt_url)
        STOP_TAG = "<stop></stop>"

        while not self._shutdown_flag.is_set():
            try:
                # Wait for user input (blocks until ChatSubmitted fires)
                task = await self._wait_for_input_from_agent()

                if self._shutdown_flag.is_set():
                    return

                if task.lower().strip() in onit.stop_commands:
                    self.call_from_thread(self._do_exit)
                    return

                if not task:
                    continue

                # Clear queues and interrupt flag
                self._interrupt_flag.clear()
                for q in (onit.input_queue, onit.output_queue, onit.safety_queue):
                    while not q.empty():
                        q.get_nowait()

                # Prompt engineering
                self.call_from_thread(self._show_thinking)
                async with prompt_client:
                    prompt_args = {
                        "task": task,
                        "data_path": onit.data_path,
                        "template_path": onit.template_path,
                        "file_server_url": onit.file_server_url,
                        "documents_path": onit.documents_path,
                        "topic": onit.topic,
                    }
                    instruction = await prompt_client.get_prompt("assistant", prompt_args)
                    instruction = instruction.messages[0].content.text

                # Inject plan mode prefix if in Plan mode
                if self._mode == OnitMode.PLAN:
                    instruction = PLAN_MODE_PROMPT_PREFIX + "\n\n" + instruction

                # Build chat kwargs
                stream = self._stream_enabled
                kwargs = {
                    "console": None,
                    "chat_ui": self.tui_chat_ui,
                    "cursor": "OnIt",
                    "memories": None,
                    "verbose": onit.verbose,
                    "data_path": onit.data_path,
                    "max_tokens": onit.model_serving.get("max_tokens", 262144),
                    "session_history": onit.load_session_history(),
                    "stream": stream,
                }
                if onit.prompt_intro:
                    kwargs["prompt_intro"] = onit.prompt_intro
                if hasattr(onit, 'middleware_chain') and onit.middleware_chain:
                    kwargs["middleware_chain"] = onit.middleware_chain

                # Filter tools in plan mode
                tool_registry = onit.tool_registry
                if self._mode == OnitMode.PLAN and tool_registry:
                    tool_registry = _filter_tool_registry(tool_registry, self._mode)

                start_time = time.monotonic()

                # Reset non-stream reasoning before each call
                self.tui_chat_ui._non_stream_reasoning = None

                # Check interrupt flag (set by Esc key from Textual thread)
                if self._interrupt_flag.is_set():
                    await onit.safety_queue.put(STOP_TAG)
                    self._interrupt_flag.clear()

                response = await chat(
                    host=onit.model_serving["host"],
                    host_key=onit.model_serving.get("host_key", "EMPTY"),
                    model=onit.model_serving["model"],
                    instruction=instruction,
                    tool_registry=tool_registry,
                    safety_queue=onit.safety_queue,
                    think=onit.model_serving.get("think", True),
                    timeout=onit.timeout,
                    **kwargs,
                )

                elapsed = f"{time.monotonic() - start_time:.2f}s"
                self.call_from_thread(self._hide_thinking)

                if response is None:
                    self.call_from_thread(
                        self.mount_message,
                        "system",
                        "Unable to get a response from the model.",
                        "",
                    )
                    continue

                response_text, thinking_text = _strip_thinking(response, show=self._show_thinking_tokens)

                # Use the reasoning field from the API (non-stream) if no
                # inline <think> tags were found
                if not thinking_text and self._show_thinking_tokens:
                    ns_reasoning = self.tui_chat_ui._non_stream_reasoning
                    if ns_reasoning:
                        thinking_text = ns_reasoning.strip()
                if not response_text:
                    self.call_from_thread(
                        self.mount_message,
                        "assistant",
                        "I wasn't able to generate a response. Please try again.",
                        elapsed,
                    )
                    continue

                if not stream:
                    # Non-streaming: mount full response
                    self.call_from_thread(
                        self.mount_message, "assistant", response_text, elapsed, thinking_text
                    )
                else:
                    # Streaming already pushed tokens via TuiChatUI;
                    # set the elapsed time on the last assistant widget
                    self.call_from_thread(self._set_last_elapsed, elapsed)

                # Save session
                try:
                    with open(onit.session_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "task": task,
                            "response": response_text,
                            "timestamp": time.time(),
                        }) + "\n")
                except Exception:
                    pass

            except asyncio.CancelledError:
                return
            except Exception as e:
                self.call_from_thread(
                    self.mount_message, "system", f"Error: {e}", ""
                )

    async def _wait_for_input_from_agent(self) -> str:
        """Wait for user input. Called from the agent thread's event loop."""
        # Signal the Textual app that we're ready for input
        self.call_from_thread(self._enable_input)

        # Block until the Textual main thread sets the event
        while not self._user_input_event.wait(timeout=0.05):
            if self._shutdown_flag.is_set():
                return ""
            if self._interrupt_flag.is_set():
                self._interrupt_flag.clear()
                return ""
        text = self._user_input_text
        self._user_input_event.clear()
        return text

    async def wait_for_user_input(self) -> str:
        """Wait for user input (called from the main Textual loop via run_coroutine_threadsafe)."""
        self._user_input_event.clear()
        # Offload blocking wait to a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._user_input_event.wait)
        return self._user_input_text

    # ── Message events from chat input ─────────────────────────

    @on(ChatSubmitted)
    def _on_chat_submitted(self, event: ChatSubmitted) -> None:
        """Handle user message submission from the chat input."""
        text = event.text

        # Handle slash commands locally
        cmd = text.strip().lower()
        if cmd == "/exit":
            self._do_exit()
            return
        if cmd == "/clear":
            self.action_clear_chat()
            return
        if cmd == "/plan":
            self._mode = OnitMode.PLAN
            try:
                self.query_one("#status-bar", StatusBar).set_mode(self._mode)
            except Exception:
                pass
            self.mount_message("system", "Switched to Plan mode.")
            return
        if cmd == "/agent":
            self._mode = OnitMode.AGENT
            try:
                self.query_one("#status-bar", StatusBar).set_mode(self._mode)
            except Exception:
                pass
            self.mount_message("system", "Switched to Agent mode.")
            return
        if cmd == "/mode":
            self.action_toggle_mode()
            return
        if cmd == "/help":
            lines = ["Available commands:"]
            for k, v in self._slash_commands.items():
                lines.append(f"  {k:12s} {v}")
            self.mount_message("system", "\n".join(lines))
            return

        # Mount user message widget
        self.mount_message("user", text)

        # Store the text and signal the agent loop
        self._user_input_text = text
        self._user_input_event.set()

        # Disable input while processing
        self._set_input_enabled(False)

    # ── Widget mounting ────────────────────────────────────────

    def mount_message(
        self, role: str, content: str, elapsed: str = "", thinking: str = ""
    ) -> None:
        """Mount a message widget into the messages container."""
        container = self.query_one("#messages", Container)
        if role == "user":
            widget = UserMessage(content)
            container.mount(widget)
        elif role == "assistant":
            # Mount thinking widget first if there's non-empty thinking content
            thinking_stripped = thinking.strip() if thinking else ""
            if thinking_stripped:
                think_widget = ThinkingMessage(thinking_stripped)
                container.mount(think_widget)
            widget = AssistantMessage(content)
            if elapsed:
                widget.set_elapsed(elapsed)
            container.mount(widget)
        elif role == "system":
            widget = Static(f"[dim italic]{content}[/]")
            container.mount(widget)
        else:
            widget = Static(content)
            container.mount(widget)
        self._scroll_to_bottom()

    def mount_tool_call(self, name: str, arguments: dict) -> None:
        container = self.query_one("#messages", Container)
        container.mount(ToolCallMessage(name, arguments))
        self._scroll_to_bottom()

    def mount_tool_result(self, name: str, result: str, success: bool = True) -> None:
        container = self.query_one("#messages", Container)
        container.mount(ToolResultMessage(name, result, success))
        self._scroll_to_bottom()

    def _ensure_assistant_widget(self) -> None:
        """Create an AssistantMessage widget for streaming if none exists."""
        if self._current_assistant_widget is None:
            widget = AssistantMessage("")
            self._current_assistant_widget = widget
            container = self.query_one("#messages", Container)
            container.mount(widget)
            self._scroll_to_bottom()

    def _ensure_thinking_widget(self) -> None:
        """Create a ThinkingMessage widget for streaming if none exists."""
        if self._current_thinking_widget is None:
            widget = ThinkingMessage("")
            self._current_thinking_widget = widget
            container = self.query_one("#messages", Container)
            container.mount(widget)
            self._scroll_to_bottom()

    def _on_stream_end(self) -> None:
        """Called when streaming is complete. Clean up empty widgets."""
        # Remove thinking widget if it has no visible content
        if self._current_thinking_widget is not None:
            content = getattr(self._current_thinking_widget, '_content', '')
            if not content or not content.strip():
                try:
                    self._current_thinking_widget.remove()
                except Exception:
                    pass
        self._current_assistant_widget = None
        self._current_thinking_widget = None
        self._scroll_to_bottom()

    def _set_last_elapsed(self, elapsed: str) -> None:
        """Set elapsed time on the most recent AssistantMessage widget."""
        try:
            container = self.query_one("#messages", Container)
            # Walk children in reverse to find the last AssistantMessage
            for child in reversed(list(container.children)):
                if isinstance(child, AssistantMessage):
                    child.set_elapsed(elapsed)
                    break
        except Exception:
            pass

    # ── Thinking / status ──────────────────────────────────────

    def _show_thinking(self) -> None:
        try:
            sb = self.query_one("#status-bar", StatusBar)
            sb.start_spinner("Thinking")
        except Exception:
            pass

    def _hide_thinking(self) -> None:
        try:
            sb = self.query_one("#status-bar", StatusBar)
            sb.stop_spinner()
        except Exception:
            pass

    # ── Input enable/disable ───────────────────────────────────

    def _enable_input(self) -> None:
        self._set_input_enabled(True)

    def _set_input_enabled(self, enabled: bool) -> None:
        try:
            input_area = self.query_one("#input-area", OnitChatInput)
            input_area.set_enabled(enabled)
            if enabled:
                input_area.focus_input()
        except Exception:
            pass

    # ── Scrolling ──────────────────────────────────────────────

    def _scroll_to_bottom(self) -> None:
        try:
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False)
            # Also schedule a deferred scroll to catch async mounts
            self.set_timer(0.1, self._deferred_scroll)
        except Exception:
            pass

    def _deferred_scroll(self) -> None:
        try:
            chat = self.query_one("#chat", VerticalScroll)
            chat.scroll_end(animate=False)
        except Exception:
            pass

    # ── Keybinding actions ─────────────────────────────────────

    def action_quit(self) -> None:
        """Quit the application cleanly."""
        self._do_exit()

    def _do_exit(self) -> None:
        """Forcefully shut down all threads and exit."""
        self._shutdown_flag.set()
        self._interrupt_flag.set()
        self._user_input_event.set()  # unblock agent thread if waiting
        self.exit()
        # Force-kill lingering threads after a short delay
        def _force_kill():
            time.sleep(0.5)
            os._exit(0)
        t = threading.Thread(target=_force_kill, daemon=True)
        t.start()

    def action_interrupt(self) -> None:
        """Interrupt the running agent via safety queue."""
        self._interrupt_flag.set()

    def action_toggle_mode(self) -> None:
        """Toggle between Plan and Agent mode."""
        if self._mode == OnitMode.AGENT:
            self._mode = OnitMode.PLAN
        else:
            self._mode = OnitMode.AGENT
        try:
            sb = self.query_one("#status-bar", StatusBar)
            sb.set_mode(self._mode)
        except Exception:
            pass

    def action_clear_chat(self) -> None:
        """Clear the messages container."""
        try:
            container = self.query_one("#messages", Container)
            container.remove_children()
        except Exception:
            pass

    # ── Token tracking ─────────────────────────────────────────

    def update_tokens(self, total: int) -> None:
        """Update the token count in status bar and banner."""
        self._total_tokens = total
        try:
            sb = self.query_one("#status-bar", StatusBar)
            sb.set_tokens(total)
        except Exception:
            pass
        try:
            banner = self.query_one("#welcome-banner", WelcomeBanner)
            banner.update_tokens(total)
        except Exception:
            pass


def _strip_thinking(text: str, show: bool = False) -> tuple[str, str]:
    """Remove or format <think>...</think> blocks from a response.

    Returns (response_text, thinking_text).  If *show* is False the
    thinking_text is always empty.  Otherwise the raw thinking content
    is returned separately so the caller can style it.
    """
    if not text:
        return (text, "")
    import re as _re
    think_pattern = _re.compile(r"<think>(.*?)</think>", _re.DOTALL)
    thinking_parts = think_pattern.findall(text)
    thinking_text = "\n".join(p.strip() for p in thinking_parts if p.strip()) if show else ""
    result = think_pattern.sub("", text)
    from ..lib.text import remove_tags
    result = remove_tags(result).strip()
    return (result, thinking_text)


def _filter_tool_registry(tool_registry: Any, mode: OnitMode) -> Any:
    """Create a filtered copy of the tool registry for the given mode.

    Returns a new registry-like object that only contains tools
    allowed in the specified mode.
    """
    if tool_registry is None:
        return None

    filtered = type(tool_registry)()
    for tool_name in tool_registry:
        if is_tool_allowed(tool_name, mode):
            # Copy all handlers for this tool (may span multiple URLs)
            for key, handler in tool_registry.handlers.items():
                if key.startswith(f"{tool_name}@"):
                    filtered.register(handler)
    return filtered
