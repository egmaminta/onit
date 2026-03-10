"""WorktreeManager — git worktree lifecycle management.

Creates, manages, and removes git worktrees with optional task binding
and event bus integration.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from .event_bus import EventBus
except (ImportError, ValueError):
    from agent.event_bus import EventBus

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")


def _git_available() -> bool:
    """Check if git is available on the system."""
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


def _is_git_repo(cwd: str = ".") -> bool:
    """Check if cwd is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            cwd=cwd,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


@dataclass
class WorktreeInfo:
    name: str
    path: str
    branch: str
    task_id: int | None = None
    kept: bool = False


class WorktreeManager:
    """Git worktree lifecycle manager."""

    def __init__(
        self,
        worktrees_dir: str = ".worktrees",
        event_bus: EventBus | None = None,
        repo_root: str = ".",
    ) -> None:
        self._dir = worktrees_dir
        self._events = event_bus or EventBus(worktrees_dir)
        self._repo_root = repo_root
        self._index_path = os.path.join(worktrees_dir, "index.json")
        self._lock = threading.Lock()
        self._worktrees: dict[str, WorktreeInfo] = {}
        self.git_available = _git_available() and _is_git_repo(repo_root)

        os.makedirs(worktrees_dir, exist_ok=True)
        self._load_index()

    def _load_index(self) -> None:
        """Load worktree index from disk."""
        if os.path.isfile(self._index_path):
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data.get("worktrees", []):
                    name = entry["name"]
                    self._worktrees[name] = WorktreeInfo(
                        name=name,
                        path=entry["path"],
                        branch=entry["branch"],
                        task_id=entry.get("task_id"),
                        kept=entry.get("kept", False),
                    )
            except Exception as e:
                logger.warning("Error loading worktree index: %s", e)

    def _save_index(self) -> None:
        """Save worktree index to disk."""
        data = {
            "worktrees": [
                {
                    "name": wt.name,
                    "path": wt.path,
                    "branch": wt.branch,
                    "task_id": wt.task_id,
                    "kept": wt.kept,
                }
                for wt in self._worktrees.values()
            ]
        }
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def create(
        self,
        name: str,
        task_id: int | None = None,
        base_ref: str = "HEAD",
    ) -> str:
        """Create a new worktree. Returns result message."""
        if not self.git_available:
            return "Git not available or not a git repository."
        if not _NAME_RE.match(name):
            return f"Invalid name '{name}'. Use 1-40 chars: [A-Za-z0-9._-]"

        with self._lock:
            if name in self._worktrees:
                return f"Worktree '{name}' already exists."

            wt_path = os.path.join(self._dir, name)
            branch = f"wt/{name}"

            self._events.emit("worktree.create.before", worktree=name, task=task_id)

            try:
                result = subprocess.run(
                    ["git", "worktree", "add", "-b", branch, wt_path, base_ref],
                    capture_output=True,
                    text=True,
                    cwd=self._repo_root,
                    timeout=30,
                )
                if result.returncode != 0:
                    self._events.emit(
                        "worktree.create.failed",
                        worktree=name,
                        error=result.stderr.strip(),
                    )
                    return f"Failed: {result.stderr.strip()}"
            except subprocess.TimeoutExpired:
                self._events.emit("worktree.create.failed", worktree=name, error="timeout")
                return "Worktree creation timed out."

            wt = WorktreeInfo(
                name=name,
                path=os.path.abspath(wt_path),
                branch=branch,
                task_id=task_id,
            )
            self._worktrees[name] = wt
            self._save_index()
            self._events.emit("worktree.create.after", worktree=name, task=task_id)
            return f"Created worktree '{name}' at {wt.path} (branch: {branch})"

    def status(self, name: str) -> str:
        """Get git status for a worktree."""
        wt = self._worktrees.get(name)
        if not wt:
            return f"Worktree '{name}' not found."
        try:
            result = subprocess.run(
                ["git", "status", "--short", "--branch"],
                capture_output=True,
                text=True,
                cwd=wt.path,
                timeout=10,
            )
            return result.stdout.strip() if result.stdout else "(clean)"
        except Exception as e:
            return f"Error: {e}"

    def run(self, name: str, command: str, timeout: int = 300) -> str:
        """Execute a command in the worktree directory."""
        wt = self._worktrees.get(name)
        if not wt:
            return f"Worktree '{name}' not found."
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=wt.path,
                timeout=timeout,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            output += f"\n[exit_code={result.returncode}]"
            # Truncate large output
            if len(output) > 50_000:
                output = output[:50_000] + "\n... [truncated]"
            return output.strip()
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s."
        except Exception as e:
            return f"Error: {e}"

    def remove(
        self,
        name: str,
        force: bool = False,
        complete_task: bool = False,
    ) -> str:
        """Remove a worktree."""
        with self._lock:
            wt = self._worktrees.get(name)
            if not wt:
                return f"Worktree '{name}' not found."

            self._events.emit("worktree.remove.before", worktree=name, task=wt.task_id)

            try:
                cmd = ["git", "worktree", "remove"]
                if force:
                    cmd.append("--force")
                cmd.append(wt.path)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self._repo_root,
                    timeout=30,
                )
                if result.returncode != 0:
                    self._events.emit(
                        "worktree.remove.failed",
                        worktree=name,
                        error=result.stderr.strip(),
                    )
                    return f"Failed: {result.stderr.strip()}"
            except Exception as e:
                self._events.emit("worktree.remove.failed", worktree=name, error=str(e))
                return f"Error: {e}"

            task_id = wt.task_id
            del self._worktrees[name]
            self._save_index()
            self._events.emit("worktree.remove.after", worktree=name, task=task_id)

            msg = f"Removed worktree '{name}'."
            if complete_task and task_id is not None:
                msg += f" Task #{task_id} should be marked completed."
            return msg

    def keep(self, name: str) -> str:
        """Mark a worktree as kept (don't auto-remove)."""
        with self._lock:
            wt = self._worktrees.get(name)
            if not wt:
                return f"Worktree '{name}' not found."
            wt.kept = True
            self._save_index()
            self._events.emit("worktree.keep", worktree=name)
            return f"Worktree '{name}' marked as kept."

    def list_all(self) -> list[dict]:
        """List all worktrees."""
        result = []
        for wt in self._worktrees.values():
            result.append({
                "name": wt.name,
                "path": wt.path,
                "branch": wt.branch,
                "task_id": wt.task_id,
                "kept": wt.kept,
            })
        return result
