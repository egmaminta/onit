"""ASCII welcome banner widget for Onit TUI."""

from __future__ import annotations

import os
import sys
from typing import Any

from rich.text import Text
from textual.widgets import Static


def _detect_unicode_support() -> bool:
    """Detect whether the terminal supports Unicode output."""
    env_override = os.environ.get("ONIT_CHARSET_MODE", "").lower()
    if env_override == "unicode":
        return True
    if env_override == "ascii":
        return False
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if "utf" in encoding.lower():
        return True
    for var in ("LANG", "LC_ALL", "LC_CTYPE"):
        val = os.environ.get(var, "")
        if "utf" in val.lower():
            return True
    return False


def _get_version() -> str:
    """Get the Onit package version."""
    try:
        from importlib.metadata import version
        return version("onit")
    except Exception:
        return "dev"


# ──── ASCII Art ───────────────────────────────────────────────────
# Unicode-capable terminals get the fancy banner; narrow/legacy get the simple one.

BANNER_UNICODE = r"""
  ██████╗ ███╗   ██╗██╗████████╗
 ██╔═══██╗████╗  ██║██║╚══██╔══╝
 ██║   ██║██╔██╗ ██║██║   ██║
 ██║   ██║██║╚██╗██║██║   ██║
 ╚██████╔╝██║ ╚████║██║   ██║
  ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝
"""

BANNER_ASCII = r"""
  ___  _   _ ___ _____
 / _ \| \ | |_ _|_   _|
| | | |  \| || |  | |
| |_| | |\  || |  | |
 \___/|_| \_|___| |_|
"""


class WelcomeBanner(Static):
    """Welcome banner displayed at the top of the chat area."""

    DEFAULT_CSS = """
    WelcomeBanner {
        height: auto;
        padding: 1 2;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        *,
        mcp_server_names: list[str] | None = None,
        model_name: str = "",
        model_host: str = "",
        max_tokens: int = 0,
        session_id: str = "",
        stream_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        self._mcp_server_names = mcp_server_names or []
        self._model_name = model_name
        self._model_host = model_host
        self._max_tokens = max_tokens
        self._session_id = session_id
        self._stream_enabled = stream_enabled
        self._total_tokens = 0
        self._use_unicode = _detect_unicode_support()

        super().__init__(self._build_banner(), **kwargs)

    def update_tokens(self, total_tokens: int) -> None:
        """Update the displayed token count and re-render."""
        self._total_tokens = total_tokens
        self.update(self._build_banner())

    def _build_banner(self) -> Text:
        banner_art = BANNER_UNICODE if self._use_unicode else BANNER_ASCII
        version = _get_version()

        text = Text()

        # ASCII art logo — strip only newlines to preserve leading spaces
        lines = banner_art.strip("\n").splitlines()
        art_width = max(len(line) for line in lines)
        for line in lines:
            text.append(line + "\n", style="bold #C32148")
        version_label = f"v{version}"
        text.append(f"{version_label:>{art_width}}\n", style="dim #C32148")
        text.append("  Developed by the University of the Philippines \u00b7 Ubiquitous Computing Laboratory\n", style="dim italic")
        text.append("\n")

        # MCP servers info
        count = len(self._mcp_server_names)
        check = "✓" if self._use_unicode else "[OK]"
        if count > 0:
            names = ", ".join(self._mcp_server_names)
            text.append(f"  {check} ", style="bold green")
            text.append(f"{count} MCP server{'s' if count != 1 else ''}: ", style="white")
            text.append(f"{names}\n", style="dim")
        else:
            text.append(f"  {check} ", style="bold yellow")
            text.append("No MCP servers loaded\n", style="dim yellow")

        # Model info
        provider = "local" if "localhost" in self._model_host or "127.0.0.1" in self._model_host else "API"
        text.append(f"  {check} ", style="bold green")
        text.append("Model: ", style="white")
        text.append(f"{self._model_name}", style="bold cyan")
        text.append(f" ({provider})\n", style="dim")

        # Token usage
        if self._max_tokens > 0:
            pct = (self._total_tokens / self._max_tokens) * 100 if self._max_tokens else 0
            total_k = f"{self._total_tokens / 1000:.1f}K" if self._total_tokens >= 1000 else str(self._total_tokens)
            max_k = f"{self._max_tokens / 1000:.0f}K" if self._max_tokens >= 1000 else str(self._max_tokens)
            text.append(f"  {check} ", style="bold green")
            text.append("Context: ", style="white")
            text.append(f"{pct:.0f}% ({total_k} out of {max_k} tokens)\n", style="dim")

        # Working directory
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        display_cwd = cwd.replace(home, "~") if cwd.startswith(home) else cwd
        text.append(f"  {check} ", style="bold green")
        text.append("Working dir: ", style="white")
        text.append(f"{display_cwd}\n", style="dim")

        # Session ID
        if self._session_id:
            short_id = self._session_id[:8]
            text.append(f"  {check} ", style="bold green")
            text.append("Session: ", style="white")
            text.append(f"{short_id}\n", style="dim")

        # Streaming mode
        stream_label = "enabled" if self._stream_enabled else "disabled"
        text.append(f"  {check} ", style="bold green")
        text.append("Streaming: ", style="white")
        text.append(f"{stream_label}\n", style="dim")

        text.append("\n")

        # Instructions (greyed out)
        bullet = "•" if self._use_unicode else "*"
        text.append(
            f"  Enter send {bullet} Ctrl+J newline {bullet} /commands\n",
            style="dim",
        )
        text.append(
            f"  Shift+Tab toggle Plan/Agent mode {bullet} Esc interrupt {bullet} /exit quit\n",
            style="dim",
        )

        return text
