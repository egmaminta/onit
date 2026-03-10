"""Chat input widget for Onit TUI.

Multiline input with Enter to submit, Ctrl+J/Shift+Enter for newline,
and Up/Down for input history navigation. Typing / shows command suggestions.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from textual import on
from textual.message import Message
from textual.widget import Widget
from textual.widgets import OptionList, TextArea
from textual.widgets.option_list import Option


# Available slash commands
SLASH_COMMANDS = {
    "/exit": "Quit OnIt",
    "/clear": "Clear chat history",
    "/plan": "Switch to Plan mode",
    "/agent": "Switch to Agent mode",
    "/mode": "Toggle Plan/Agent mode",
    "/help": "Show available commands",
}


class OnitTextArea(TextArea):
    """TextArea with custom key bindings for chat input."""

    class Submitted(Message):
        """Posted when the user submits input (Enter on single line or Ctrl+Enter)."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class HistoryPrevious(Message):
        """Request to navigate to previous history entry."""

    class HistoryNext(Message):
        """Request to navigate to next history entry."""

    BINDINGS = [
        ("ctrl+j", "newline", "Insert newline"),
        ("shift+enter", "newline", "Insert newline"),
    ]

    def _on_key(self, event) -> None:
        """Handle key events before they reach the default handler."""
        if event.key == "tab":
            # If suggestions are visible, let parent handle completion
            parent = self.parent
            if isinstance(parent, OnitChatInput) and parent._suggestions_visible:
                event.prevent_default()
                event.stop()
                parent._accept_suggestion()
                return

        if event.key == "enter":
            # If suggestions are visible, accept and submit the command
            parent = self.parent
            if isinstance(parent, OnitChatInput) and parent._suggestions_visible:
                event.prevent_default()
                event.stop()
                parent._accept_and_submit()
                return
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
            return

        if event.key == "escape":
            parent = self.parent
            if isinstance(parent, OnitChatInput) and parent._suggestions_visible:
                event.prevent_default()
                event.stop()
                parent._hide_suggestions()
                return

        if event.key == "up":
            parent = self.parent
            if isinstance(parent, OnitChatInput) and parent._suggestions_visible:
                event.prevent_default()
                event.stop()
                parent._move_suggestion(-1)
                return
            row, _col = self.cursor_location
            if row == 0:
                event.prevent_default()
                event.stop()
                self.post_message(self.HistoryPrevious())
                return

        if event.key == "down":
            parent = self.parent
            if isinstance(parent, OnitChatInput) and parent._suggestions_visible:
                event.prevent_default()
                event.stop()
                parent._move_suggestion(1)
                return
            row, _col = self.cursor_location
            if row == self.document.line_count - 1:
                event.prevent_default()
                event.stop()
                self.post_message(self.HistoryNext())
                return

    def action_newline(self) -> None:
        """Insert a literal newline."""
        self.insert("\n")


class OnitChatInput(Widget):
    """Chat input widget with history and slash command suggestions."""

    DEFAULT_CSS = """
    OnitChatInput {
        height: auto;
        min-height: 3;
        max-height: 12;
        padding: 0 1;
    }

    OnitChatInput TextArea {
        height: auto;
        min-height: 1;
        max-height: 10;
    }

    OnitChatInput OptionList {
        height: auto;
        max-height: 8;
        background: $surface;
        border: round $accent;
        display: none;
    }

    OnitChatInput OptionList.visible {
        display: block;
    }
    """

    def __init__(self, max_history: int = 100, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._history: deque[str] = deque(maxlen=max_history)
        self._history_index: int = -1
        self._saved_input: str = ""
        self._suggestions_visible: bool = False

    def compose(self):
        yield OptionList(id="slash-suggestions")
        yield OnitTextArea(id="chat-textarea")

    def on_mount(self) -> None:
        ta = self.query_one("#chat-textarea", OnitTextArea)
        ta.focus()
        # Override TextArea theme to have transparent background
        try:
            from textual.widgets.text_area import TextAreaTheme
            from rich.style import Style
            transparent = Style.null()
            theme = TextAreaTheme(
                name="onit-transparent",
                base_style=transparent,
                gutter_style=transparent,
                cursor_style=Style(reverse=True),
                cursor_line_style=transparent,
                cursor_line_gutter_style=transparent,
                bracket_matching_style=Style(bold=True, underline=True),
                selection_style=Style(bgcolor="grey37"),
            )
            ta.register_theme(theme)
            ta.theme = "onit-transparent"
        except Exception:
            pass

    @on(TextArea.Changed)
    def _on_text_changed(self, event: TextArea.Changed) -> None:
        """Show/hide suggestions as user types."""
        event.stop()
        text = event.text_area.text.strip()
        if text.startswith("/") and "\n" not in text:
            matches = [
                (cmd, desc) for cmd, desc in SLASH_COMMANDS.items()
                if cmd.startswith(text.lower())
            ]
            if matches:
                self._show_suggestions(matches)
                return
        self._hide_suggestions()

    def _show_suggestions(self, matches: list[tuple[str, str]]) -> None:
        ol = self.query_one("#slash-suggestions", OptionList)
        ol.clear_options()
        for cmd, desc in matches:
            ol.add_option(Option(f"{cmd}  {desc}", id=cmd))
        ol.add_class("visible")
        self._suggestions_visible = True
        if ol.option_count > 0:
            ol.highlighted = 0

    def _hide_suggestions(self) -> None:
        ol = self.query_one("#slash-suggestions", OptionList)
        ol.remove_class("visible")
        self._suggestions_visible = False

    def _accept_suggestion(self) -> None:
        ol = self.query_one("#slash-suggestions", OptionList)
        if ol.highlighted is not None and ol.option_count > 0:
            option = ol.get_option_at_index(ol.highlighted)
            cmd = option.id
            ta = self.query_one("#chat-textarea", OnitTextArea)
            ta.clear()
            ta.insert(cmd)
        self._hide_suggestions()

    def _accept_and_submit(self) -> None:
        """Accept the highlighted suggestion and submit it."""
        ol = self.query_one("#slash-suggestions", OptionList)
        cmd = None
        if ol.highlighted is not None and ol.option_count > 0:
            option = ol.get_option_at_index(ol.highlighted)
            cmd = option.id
        self._hide_suggestions()
        if cmd:
            ta = self.query_one("#chat-textarea", OnitTextArea)
            ta.clear()
            self.post_message(ChatSubmitted(cmd))

    def _move_suggestion(self, delta: int) -> None:
        ol = self.query_one("#slash-suggestions", OptionList)
        if ol.option_count == 0:
            return
        idx = (ol.highlighted or 0) + delta
        idx = max(0, min(idx, ol.option_count - 1))
        ol.highlighted = idx

    @on(OnitTextArea.Submitted)
    def _on_submitted(self, event: OnitTextArea.Submitted) -> None:
        """Bubble the submitted text up and record in history."""
        text = event.text
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._history_index = -1
        self._saved_input = ""
        # Re-post as our own message so the app can listen
        self.post_message(ChatSubmitted(text))

    @on(OnitTextArea.HistoryPrevious)
    def _on_history_prev(self, _event: OnitTextArea.HistoryPrevious) -> None:
        if not self._history:
            return
        ta = self.query_one("#chat-textarea", OnitTextArea)
        if self._history_index == -1:
            self._saved_input = ta.text
            self._history_index = len(self._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return
        ta.clear()
        ta.insert(self._history[self._history_index])

    @on(OnitTextArea.HistoryNext)
    def _on_history_next(self, _event: OnitTextArea.HistoryNext) -> None:
        if self._history_index == -1:
            return
        ta = self.query_one("#chat-textarea", OnitTextArea)
        self._history_index += 1
        if self._history_index >= len(self._history):
            self._history_index = -1
            ta.clear()
            ta.insert(self._saved_input)
        else:
            ta.clear()
            ta.insert(self._history[self._history_index])

    def focus_input(self) -> None:
        """Focus the text area."""
        ta = self.query_one("#chat-textarea", OnitTextArea)
        ta.focus()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the input area."""
        ta = self.query_one("#chat-textarea", OnitTextArea)
        ta.read_only = not enabled


class ChatSubmitted(Message):
    """Message posted when the user submits a chat message."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text
