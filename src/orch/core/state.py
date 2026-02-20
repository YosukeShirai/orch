"""SQLite-based state management for projects, tasks, and execution logs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from orch.agents.base import AgentResult, ProjectStatus, TaskStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    result_output TEXT,
    result_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (id, project_id),
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS dag_snapshots (
    project_id TEXT PRIMARY KEY,
    snapshot TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS execution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    task_id TEXT,
    event_type TEXT NOT NULL,
    data TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateManager:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("StateManager not initialized. Call initialize() first.")
        return self._db

    # --- Projects ---

    async def create_project(self, project_id: str, goal: str) -> None:
        now = _now()
        await self.db.execute(
            "INSERT INTO projects (id, goal, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (project_id, goal, ProjectStatus.ACTIVE.value, now, now),
        )
        await self.db.commit()

    async def update_project_status(self, project_id: str, status: ProjectStatus) -> None:
        await self.db.execute(
            "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, _now(), project_id),
        )
        await self.db.commit()

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_resumable_project(self) -> dict[str, Any] | None:
        """Find the most recent active or paused project."""
        async with self.db.execute(
            "SELECT * FROM projects WHERE status IN (?, ?) ORDER BY updated_at DESC LIMIT 1",
            (ProjectStatus.ACTIVE.value, ProjectStatus.PAUSED.value),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_projects(self) -> list[dict[str, Any]]:
        async with self.db.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # --- Tasks ---

    async def create_tasks(self, project_id: str, tasks: list[dict[str, str]]) -> None:
        now = _now()
        for task in tasks:
            await self.db.execute(
                "INSERT INTO tasks (id, project_id, title, description, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task["id"],
                    project_id,
                    task["title"],
                    task.get("description", ""),
                    TaskStatus.PENDING.value,
                    now,
                    now,
                ),
            )
        await self.db.commit()

    async def update_task_status(
        self,
        project_id: str,
        task_id: str,
        status: TaskStatus,
        result: AgentResult | None = None,
    ) -> None:
        if result:
            await self.db.execute(
                "UPDATE tasks SET status = ?, result_output = ?, result_error = ?, updated_at = ? "
                "WHERE id = ? AND project_id = ?",
                (status.value, result.output, result.error, _now(), task_id, project_id),
            )
        else:
            await self.db.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ? AND project_id = ?",
                (status.value, _now(), task_id, project_id),
            )
        await self.db.commit()

    async def get_tasks(self, project_id: str) -> list[dict[str, Any]]:
        async with self.db.execute(
            "SELECT * FROM tasks WHERE project_id = ? ORDER BY id", (project_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_completed_task_ids(self, project_id: str) -> set[str]:
        async with self.db.execute(
            "SELECT id FROM tasks WHERE project_id = ? AND status = ?",
            (project_id, TaskStatus.COMPLETED.value),
        ) as cursor:
            rows = await cursor.fetchall()
            return {row["id"] for row in rows}

    async def get_task_result(self, project_id: str, task_id: str) -> str | None:
        async with self.db.execute(
            "SELECT result_output FROM tasks WHERE project_id = ? AND id = ?",
            (project_id, task_id),
        ) as cursor:
            row = await cursor.fetchone()
            return row["result_output"] if row else None

    async def reset_running_tasks(self, project_id: str) -> int:
        """Reset RUNNING tasks to PENDING (crash recovery)."""
        cursor = await self.db.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE project_id = ? AND status = ?",
            (TaskStatus.PENDING.value, _now(), project_id, TaskStatus.RUNNING.value),
        )
        await self.db.commit()
        return cursor.rowcount

    # --- DAG Snapshots ---

    async def save_dag_snapshot(self, project_id: str, snapshot: str) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO dag_snapshots (project_id, snapshot, created_at) VALUES (?, ?, ?)",
            (project_id, snapshot, _now()),
        )
        await self.db.commit()

    async def get_dag_snapshot(self, project_id: str) -> str | None:
        async with self.db.execute(
            "SELECT snapshot FROM dag_snapshots WHERE project_id = ?", (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row["snapshot"] if row else None

    # --- Execution Log ---

    async def log_event(
        self,
        project_id: str,
        event_type: str,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO execution_log (project_id, task_id, event_type, data, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, task_id, event_type, json.dumps(data) if data else None, _now()),
        )
        await self.db.commit()
