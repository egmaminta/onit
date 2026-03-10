"""TaskManager — persistent file-based task board.

Stores tasks as JSON files in a ``.tasks/`` directory.  Supports CRUD,
dependency tracking, owner/worktree binding, and thread-safe writes.

No langchain / langgraph dependency.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Task:
    id: int
    subject: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    owner: str = ""
    worktree: str = ""
    blocked_by: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d["status"] = TaskStatus(d.get("status", "pending"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TaskManager:
    """File-based persistent task board.

    Tasks are stored as individual JSON files in ``{tasks_dir}/task_{id}.json``.
    """

    def __init__(self, tasks_dir: str = ".tasks") -> None:
        self._dir = tasks_dir
        self._lock = threading.Lock()
        self._next_id = 1
        self._ensure_dir()
        self._load_next_id()

    def _ensure_dir(self) -> None:
        os.makedirs(self._dir, exist_ok=True)

    def _task_path(self, task_id: int) -> str:
        return os.path.join(self._dir, f"task_{task_id}.json")

    def _load_next_id(self) -> None:
        """Scan existing files to determine next available ID."""
        max_id = 0
        if os.path.isdir(self._dir):
            for entry in os.listdir(self._dir):
                if entry.startswith("task_") and entry.endswith(".json"):
                    try:
                        tid = int(entry[5:-5])
                        max_id = max(max_id, tid)
                    except ValueError:
                        pass
        self._next_id = max_id + 1

    def _write_task(self, task: Task) -> None:
        path = self._task_path(task.id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2)

    def _read_task(self, task_id: int) -> Task | None:
        path = self._task_path(task_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Task.from_dict(json.load(f))
        except Exception as e:
            logger.warning("Error reading task %d: %s", task_id, e)
            return None

    def create(self, subject: str, description: str = "") -> Task:
        """Create a new task. Returns the task."""
        with self._lock:
            task = Task(
                id=self._next_id,
                subject=subject,
                description=description,
            )
            self._next_id += 1
            self._write_task(task)
            return task

    def get(self, task_id: int) -> Task | None:
        return self._read_task(task_id)

    def update(
        self,
        task_id: int,
        status: TaskStatus | None = None,
        owner: str | None = None,
        description: str | None = None,
    ) -> Task | None:
        """Update a task. Returns updated task or None if not found."""
        with self._lock:
            task = self._read_task(task_id)
            if task is None:
                return None
            if status is not None:
                task.status = status
                # If completing, remove from others' blocked_by
                if status == TaskStatus.COMPLETED:
                    self._unblock(task_id)
            if owner is not None:
                task.owner = owner
            if description is not None:
                task.description = description
            task.updated_at = time.time()
            self._write_task(task)
            return task

    def _unblock(self, completed_id: int) -> None:
        """Remove completed_id from all tasks' blocked_by lists."""
        for task in self._list_all_unsafe():
            if completed_id in task.blocked_by:
                task.blocked_by.remove(completed_id)
                task.updated_at = time.time()
                self._write_task(task)

    def list_all(self) -> list[Task]:
        with self._lock:
            return self._list_all_unsafe()

    def _list_all_unsafe(self) -> list[Task]:
        tasks = []
        if not os.path.isdir(self._dir):
            return tasks
        for entry in sorted(os.listdir(self._dir)):
            if entry.startswith("task_") and entry.endswith(".json"):
                try:
                    tid = int(entry[5:-5])
                    task = self._read_task(tid)
                    if task:
                        tasks.append(task)
                except ValueError:
                    pass
        return tasks

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> Task | None:
        """Bind a task to a worktree and optionally set owner."""
        with self._lock:
            task = self._read_task(task_id)
            if task is None:
                return None
            task.worktree = worktree
            task.status = TaskStatus.IN_PROGRESS
            if owner:
                task.owner = owner
            task.updated_at = time.time()
            self._write_task(task)
            return task

    def unbind_worktree(self, task_id: int) -> Task | None:
        """Remove worktree binding from a task."""
        with self._lock:
            task = self._read_task(task_id)
            if task is None:
                return None
            task.worktree = ""
            task.updated_at = time.time()
            self._write_task(task)
            return task

    def add_dependency(self, task_id: int, blocked_by_id: int) -> Task | None:
        """Add a dependency: task_id is blocked by blocked_by_id."""
        with self._lock:
            task = self._read_task(task_id)
            if task is None:
                return None
            blocker = self._read_task(blocked_by_id)
            if blocker is None:
                return None
            if blocked_by_id not in task.blocked_by:
                task.blocked_by.append(blocked_by_id)
                task.updated_at = time.time()
                self._write_task(task)
            return task
