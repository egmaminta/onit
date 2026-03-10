"""SkillsMiddleware — progressive skill disclosure.

Layer 1: Scan ``skills/`` directories for SKILL.md files and inject a
concise catalog into the system prompt.

Layer 2: Provide a ``load_skill`` virtual tool that returns the full
SKILL.md body on demand.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

try:
    from ..type.agent_state import AgentState
    from . import AgentMiddleware
except (ImportError, ValueError):
    from type.agent_state import AgentState
    from middleware import AgentMiddleware

logger = logging.getLogger(__name__)


def _parse_skill_md(content: str) -> tuple[dict, str]:
    """Parse YAML-ish frontmatter and body from a SKILL.md file.

    Returns (metadata_dict, body_text).  Uses simple regex parsing
    to avoid a PyYAML dependency for this lightweight feature.
    """
    meta: dict = {}
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            front = content[3:end].strip()
            body = content[end + 3:].strip()
            for line in front.split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip()
    return meta, body


class SkillLoader:
    """Lazy-loads skills from multiple directories."""

    def __init__(self, skill_dirs: list[str] | None = None) -> None:
        self._dirs = skill_dirs or []
        self._catalog: dict[str, dict] = {}  # name → {description, path, tags}
        self._loaded = False

    def scan(self) -> None:
        """Scan all skill directories and build the catalog."""
        self._catalog.clear()
        for skill_dir in self._dirs:
            if not os.path.isdir(skill_dir):
                continue
            for entry in os.listdir(skill_dir):
                skill_path = os.path.join(skill_dir, entry, "SKILL.md")
                if not os.path.isfile(skill_path):
                    continue
                name = entry.lower()
                # Validate name: 1-64 chars, lowercase alphanumeric + hyphens
                if not re.match(r"^[a-z0-9][a-z0-9-]{0,63}$", name):
                    logger.warning("Skipping skill with invalid name: %s", entry)
                    continue
                try:
                    with open(skill_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    meta, _ = _parse_skill_md(content)
                    self._catalog[name] = {
                        "name": meta.get("name", name),
                        "description": meta.get("description", ""),
                        "tags": meta.get("tags", ""),
                        "path": skill_path,
                    }
                except Exception as e:
                    logger.warning("Error loading skill %s: %s", entry, e)
        self._loaded = True

    @property
    def catalog(self) -> dict[str, dict]:
        if not self._loaded:
            self.scan()
        return self._catalog

    def load_skill(self, name: str) -> str | None:
        """Load the full SKILL.md body for a skill by name."""
        info = self.catalog.get(name)
        if not info:
            return None
        try:
            with open(info["path"], "r", encoding="utf-8") as f:
                content = f.read()
            _, body = _parse_skill_md(content)
            return f'<skill name="{name}">\n{body}\n</skill>'
        except Exception as e:
            return f"Error loading skill {name}: {e}"


_LOAD_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": "Load detailed instructions for a specific skill by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name from the catalog.",
                },
            },
            "required": ["name"],
        },
    },
}


class SkillsMiddleware(AgentMiddleware):
    """Progressive skill disclosure via system prompt + load_skill tool."""

    def __init__(self, skill_dirs: list[str] | None = None) -> None:
        self.loader = SkillLoader(skill_dirs)

    async def initialize(self, state: AgentState) -> None:
        """Scan skill directories on first run."""
        self.loader.scan()

    async def before_model(self, state: AgentState) -> AgentState:
        # Inject skill catalog into system prompt
        catalog = self.loader.catalog
        if catalog:
            catalog_text = "\n".join(
                f"- **{info['name']}**: {info['description']}"
                for info in catalog.values()
            )
            messages = state.get("messages", [])
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                sys_content = messages[0].get("content", "")
                if isinstance(sys_content, str) and "<skills>" not in sys_content:
                    messages[0]["content"] = sys_content + (
                        f"\n\n<skills>\nAvailable skills (use load_skill to get full instructions):\n"
                        f"{catalog_text}\n</skills>"
                    )

        # Inject load_skill tool
        tools = state.get("tools", [])
        tool_names = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
        if "load_skill" not in tool_names:
            tools.append(_LOAD_SKILL_TOOL)
            state["tools"] = tools

        return state

    async def handle_tool(self, state: AgentState, tool_name: str, args: dict) -> Optional[str]:
        if tool_name != "load_skill":
            return None
        name = args.get("name", "")
        result = self.loader.load_skill(name)
        if result is None:
            available = ", ".join(self.loader.catalog.keys())
            return f"Skill '{name}' not found. Available: {available}"
        return result
