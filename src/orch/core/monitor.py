"""Event monitoring and logging wrapper."""

from __future__ import annotations

import enum
from typing import Any

from orch.core.state import StateManager


class EventType(enum.Enum):
    PROJECT_CREATED = "project_created"
    DAG_GENERATED = "dag_generated"
    DAG_APPROVED = "dag_approved"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    PROJECT_COMPLETED = "project_completed"
    PROJECT_PAUSED = "project_paused"
    PROJECT_RESUMED = "project_resumed"
    CHECKPOINT = "checkpoint"


class Monitor:
    def __init__(self, state: StateManager) -> None:
        self._state = state

    async def log(
        self,
        project_id: str,
        event: EventType,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        await self._state.log_event(
            project_id=project_id,
            event_type=event.value,
            task_id=task_id,
            data=data,
        )
