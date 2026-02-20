"""Shared fixtures for orch tests."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from orch.agents.base import AgentResult, BaseAgent
from orch.core.dag import DAGEngine, TaskNode
from orch.core.state import StateManager


class MockAgent(BaseAgent):
    """Agent that returns predefined results."""

    def __init__(
        self,
        results: list[AgentResult] | None = None,
        delay: float = 0.0,
    ) -> None:
        self._results = list(results) if results else []
        self._call_count = 0
        self._delay = delay
        self.calls: list[tuple[str, str | None]] = []
        self.concurrency_log: list[int] = []
        self._active_count = 0

    def add_result(self, result: AgentResult) -> None:
        self._results.append(result)

    async def execute(self, prompt: str, context: str | None = None) -> AgentResult:
        self._active_count += 1
        self.concurrency_log.append(self._active_count)
        self.calls.append((prompt, context))
        try:
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            if self._call_count < len(self._results):
                result = self._results[self._call_count]
            else:
                result = AgentResult(success=True, output=f"Mock result {self._call_count}")
            self._call_count += 1
            return result
        finally:
            self._active_count -= 1


@pytest.fixture
def mock_agent() -> MockAgent:
    return MockAgent()


@pytest.fixture
def sample_tasks() -> list[TaskNode]:
    return [
        TaskNode(id="task-1", title="Setup project", description="Initialize the project structure"),
        TaskNode(
            id="task-2",
            title="Implement feature",
            description="Implement the main feature",
            dependencies=["task-1"],
        ),
        TaskNode(
            id="task-3",
            title="Write tests",
            description="Write unit tests",
            dependencies=["task-2"],
        ),
    ]


@pytest.fixture
def sample_dag(sample_tasks: list[TaskNode]) -> DAGEngine:
    dag = DAGEngine()
    dag.build(sample_tasks)
    return dag


@pytest.fixture
def diamond_tasks() -> list[TaskNode]:
    """Diamond DAG: t1 -> {t2, t3} -> t4."""
    return [
        TaskNode(id="t1", title="Root", description="Root task"),
        TaskNode(id="t2", title="Left", description="Left branch", dependencies=["t1"]),
        TaskNode(id="t3", title="Right", description="Right branch", dependencies=["t1"]),
        TaskNode(id="t4", title="Join", description="Join task", dependencies=["t2", "t3"]),
    ]


@pytest.fixture
def diamond_dag(diamond_tasks: list[TaskNode]) -> DAGEngine:
    dag = DAGEngine()
    dag.build(diamond_tasks)
    return dag


@pytest.fixture
async def tmp_state() -> StateManager:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        state = StateManager(db_path)
        await state.initialize()
        yield state
        await state.close()
