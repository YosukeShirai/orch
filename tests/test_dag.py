"""Tests for DAG engine."""

from __future__ import annotations

import pytest

from orch.agents.base import TaskStatus
from orch.core.dag import DAGEngine, DAGValidationError, TaskNode


class TestDAGEngine:
    def test_build_simple_dag(self, sample_tasks: list[TaskNode]) -> None:
        dag = DAGEngine()
        dag.build(sample_tasks)
        assert dag.task_count() == 3

    def test_build_detects_missing_dependency(self) -> None:
        tasks = [
            TaskNode(id="t1", title="Task 1", description="", dependencies=["nonexistent"]),
        ]
        dag = DAGEngine()
        with pytest.raises(DAGValidationError, match="unknown task"):
            dag.build(tasks)

    def test_build_detects_cycle(self) -> None:
        tasks = [
            TaskNode(id="t1", title="Task 1", description="", dependencies=["t2"]),
            TaskNode(id="t2", title="Task 2", description="", dependencies=["t1"]),
        ]
        dag = DAGEngine()
        with pytest.raises(DAGValidationError, match="cycles"):
            dag.build(tasks)

    def test_get_ready_tasks_initial(self, sample_dag: DAGEngine) -> None:
        ready = sample_dag.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "task-1"

    def test_get_ready_tasks_after_completion(self, sample_dag: DAGEngine) -> None:
        ready = sample_dag.get_ready_tasks(completed={"task-1"})
        assert len(ready) == 1
        assert ready[0].id == "task-2"

    def test_get_ready_tasks_parallel(self) -> None:
        tasks = [
            TaskNode(id="t1", title="Base", description=""),
            TaskNode(id="t2", title="A", description="", dependencies=["t1"]),
            TaskNode(id="t3", title="B", description="", dependencies=["t1"]),
            TaskNode(id="t4", title="Final", description="", dependencies=["t2", "t3"]),
        ]
        dag = DAGEngine()
        dag.build(tasks)

        ready = dag.get_ready_tasks(completed={"t1"})
        ids = {t.id for t in ready}
        assert ids == {"t2", "t3"}

    def test_get_ready_tasks_all_done(self, sample_dag: DAGEngine) -> None:
        ready = sample_dag.get_ready_tasks(completed={"task-1", "task-2", "task-3"})
        assert len(ready) == 0

    def test_execution_order(self, sample_dag: DAGEngine) -> None:
        order = sample_dag.get_execution_order()
        assert order.index("task-1") < order.index("task-2")
        assert order.index("task-2") < order.index("task-3")

    def test_snapshot_roundtrip(self, sample_dag: DAGEngine) -> None:
        snapshot = sample_dag.to_snapshot()
        restored = DAGEngine.from_snapshot(snapshot)
        assert restored.task_count() == sample_dag.task_count()
        assert set(restored.nodes.keys()) == set(sample_dag.nodes.keys())

    def test_display_string(self, sample_dag: DAGEngine) -> None:
        display = sample_dag.to_display_string()
        assert "task-1" in display
        assert "task-2" in display
        assert "task-3" in display
        assert "Setup project" in display

    def test_single_task_dag(self) -> None:
        tasks = [TaskNode(id="solo", title="Only task", description="Do it")]
        dag = DAGEngine()
        dag.build(tasks)
        assert dag.task_count() == 1
        ready = dag.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "solo"


class TestTaskNode:
    def test_to_dict_roundtrip(self) -> None:
        node = TaskNode(
            id="t1",
            title="Test",
            description="Desc",
            dependencies=["t0"],
            metadata={"key": "value"},
        )
        restored = TaskNode.from_dict(node.to_dict())
        assert restored.id == node.id
        assert restored.title == node.title
        assert restored.dependencies == node.dependencies
        assert restored.metadata == node.metadata
