"""Tests for SQLite state manager."""

from __future__ import annotations

import pytest

from orch.agents.base import AgentResult, ProjectStatus, TaskStatus
from orch.core.state import StateManager


class TestStateManager:
    async def test_create_and_get_project(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Build something")
        project = await tmp_state.get_project("p1")
        assert project is not None
        assert project["goal"] == "Build something"
        assert project["status"] == "active"

    async def test_update_project_status(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal")
        await tmp_state.update_project_status("p1", ProjectStatus.PAUSED)
        project = await tmp_state.get_project("p1")
        assert project["status"] == "paused"

    async def test_get_resumable_project(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal 1")
        await tmp_state.create_project("p2", "Goal 2")
        await tmp_state.update_project_status("p1", ProjectStatus.COMPLETED)

        project = await tmp_state.get_resumable_project()
        assert project is not None
        assert project["id"] == "p2"

    async def test_get_resumable_project_none(self, tmp_state: StateManager) -> None:
        project = await tmp_state.get_resumable_project()
        assert project is None

    async def test_create_and_get_tasks(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal")
        tasks = [
            {"id": "t1", "title": "Task 1", "description": "Do thing 1"},
            {"id": "t2", "title": "Task 2", "description": "Do thing 2"},
        ]
        await tmp_state.create_tasks("p1", tasks)

        result = await tmp_state.get_tasks("p1")
        assert len(result) == 2
        assert result[0]["id"] == "t1"
        assert result[1]["id"] == "t2"

    async def test_update_task_status_with_result(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal")
        await tmp_state.create_tasks("p1", [{"id": "t1", "title": "Task 1"}])

        agent_result = AgentResult(success=True, output="Done!", error=None)
        await tmp_state.update_task_status("p1", "t1", TaskStatus.COMPLETED, agent_result)

        tasks = await tmp_state.get_tasks("p1")
        assert tasks[0]["status"] == "completed"
        assert tasks[0]["result_output"] == "Done!"

    async def test_get_completed_task_ids(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal")
        await tmp_state.create_tasks(
            "p1",
            [
                {"id": "t1", "title": "T1"},
                {"id": "t2", "title": "T2"},
                {"id": "t3", "title": "T3"},
            ],
        )
        await tmp_state.update_task_status("p1", "t1", TaskStatus.COMPLETED)
        await tmp_state.update_task_status("p1", "t3", TaskStatus.COMPLETED)

        completed = await tmp_state.get_completed_task_ids("p1")
        assert completed == {"t1", "t3"}

    async def test_reset_running_tasks(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal")
        await tmp_state.create_tasks(
            "p1",
            [{"id": "t1", "title": "T1"}, {"id": "t2", "title": "T2"}],
        )
        await tmp_state.update_task_status("p1", "t1", TaskStatus.RUNNING)
        await tmp_state.update_task_status("p1", "t2", TaskStatus.COMPLETED)

        count = await tmp_state.reset_running_tasks("p1")
        assert count == 1

        tasks = await tmp_state.get_tasks("p1")
        statuses = {t["id"]: t["status"] for t in tasks}
        assert statuses["t1"] == "pending"
        assert statuses["t2"] == "completed"

    async def test_dag_snapshot_roundtrip(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal")
        snapshot = '{"tasks": [{"id": "t1"}]}'
        await tmp_state.save_dag_snapshot("p1", snapshot)

        result = await tmp_state.get_dag_snapshot("p1")
        assert result == snapshot

    async def test_log_event(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal")
        await tmp_state.log_event("p1", "test_event", task_id="t1", data={"key": "val"})
        # No assertion needed — just verify no error

    async def test_list_projects(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal 1")
        await tmp_state.create_project("p2", "Goal 2")
        projects = await tmp_state.list_projects()
        assert len(projects) == 2

    async def test_get_task_result(self, tmp_state: StateManager) -> None:
        await tmp_state.create_project("p1", "Goal")
        await tmp_state.create_tasks("p1", [{"id": "t1", "title": "T1"}])
        result = AgentResult(success=True, output="output text")
        await tmp_state.update_task_status("p1", "t1", TaskStatus.COMPLETED, result)

        output = await tmp_state.get_task_result("p1", "t1")
        assert output == "output text"
