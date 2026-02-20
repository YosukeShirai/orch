"""Scheduler: determines next tasks to execute based on DAG and state."""

from __future__ import annotations

from orch.core.dag import DAGEngine, TaskNode


class Scheduler:
    def __init__(self, dag: DAGEngine) -> None:
        self._dag = dag

    def get_next_tasks(self, completed: set[str]) -> list[TaskNode]:
        """Get the next tasks ready for execution."""
        return self._dag.get_ready_tasks(completed)

    def is_complete(self, completed: set[str]) -> bool:
        """Check if all tasks are completed."""
        return len(completed) == self._dag.task_count()

    def get_progress(self, completed: set[str]) -> tuple[int, int]:
        """Return (completed_count, total_count)."""
        return len(completed), self._dag.task_count()
