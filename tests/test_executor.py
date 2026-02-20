"""Integration tests for the executor using mock agent."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from orch.agents.base import AgentResult, ProjectStatus, TaskStatus
from orch.core.dag import DAGEngine, TaskNode
from orch.core.executor import Executor
from orch.core.monitor import Monitor
from orch.core.state import StateManager
from tests.conftest import MockAgent


def _dag_json(tasks: list[dict]) -> str:
    return json.dumps({"tasks": tasks})


SIMPLE_DAG_RESPONSE = _dag_json(
    [
        {"id": "task-1", "title": "Step 1", "description": "First step", "dependencies": []},
        {
            "id": "task-2",
            "title": "Step 2",
            "description": "Second step",
            "dependencies": ["task-1"],
        },
    ]
)


class TestExecutor:
    @pytest.fixture
    async def setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            state = StateManager(db_path)
            await state.initialize()

            agent = MockAgent()
            monitor = Monitor(state)
            executor = Executor(state, agent, monitor, supervised=False)

            yield state, agent, executor

            await state.close()

    async def test_run_full_flow(self, setup) -> None:
        state, agent, executor = setup

        # First call: DAG generation
        agent.add_result(AgentResult(success=True, output=SIMPLE_DAG_RESPONSE))
        # Task executions
        agent.add_result(AgentResult(success=True, output="Step 1 done"))
        agent.add_result(AgentResult(success=True, output="Step 2 done"))

        with patch("builtins.input", return_value="a"):
            project_id = await executor.run("Test goal")

        project = await state.get_project(project_id)
        assert project["status"] == "completed"

        tasks = await state.get_tasks(project_id)
        assert len(tasks) == 2
        assert all(t["status"] == "completed" for t in tasks)

    async def test_run_cancel(self, setup) -> None:
        state, agent, executor = setup

        agent.add_result(AgentResult(success=True, output=SIMPLE_DAG_RESPONSE))

        with patch("builtins.input", return_value="c"):
            project_id = await executor.run("Test goal")

        project = await state.get_project(project_id)
        assert project["status"] == "cancelled"

    async def test_run_task_failure(self, setup) -> None:
        state, agent, executor = setup

        agent.add_result(AgentResult(success=True, output=SIMPLE_DAG_RESPONSE))
        agent.add_result(AgentResult(success=False, output="", error="Something broke"))

        with patch("builtins.input", return_value="a"):
            project_id = await executor.run("Test goal")

        project = await state.get_project(project_id)
        assert project["status"] == "failed"

        tasks = await state.get_tasks(project_id)
        failed = [t for t in tasks if t["status"] == "failed"]
        assert len(failed) == 1

    async def test_resume_flow(self, setup) -> None:
        state, agent, executor = setup

        # Set up a paused project manually
        await state.create_project("test-proj", "Resume goal")
        await state.update_project_status("test-proj", ProjectStatus.PAUSED)

        dag_snapshot = _dag_json(
            [
                {"id": "task-1", "title": "Done", "description": "Already done", "dependencies": []},
                {
                    "id": "task-2",
                    "title": "Todo",
                    "description": "Needs doing",
                    "dependencies": ["task-1"],
                },
            ]
        )
        await state.save_dag_snapshot("test-proj", dag_snapshot)
        await state.create_tasks(
            "test-proj",
            [
                {"id": "task-1", "title": "Done", "description": "Already done"},
                {"id": "task-2", "title": "Todo", "description": "Needs doing"},
            ],
        )
        await state.update_task_status("test-proj", "task-1", TaskStatus.COMPLETED)

        agent.add_result(AgentResult(success=True, output="Task 2 done"))

        project_id = await executor.resume()
        assert project_id == "test-proj"

        project = await state.get_project("test-proj")
        assert project["status"] == "completed"

    async def test_supervised_pause(self, setup) -> None:
        state, agent, executor_unsupervised = setup

        executor = Executor(state, agent, Monitor(state), supervised=True)

        agent.add_result(AgentResult(success=True, output=SIMPLE_DAG_RESPONSE))
        agent.add_result(AgentResult(success=True, output="Step 1 done"))

        # Approve DAG, then pause at checkpoint
        inputs = iter(["a", "p"])
        with patch("builtins.input", side_effect=inputs):
            project_id = await executor.run("Test goal")

        project = await state.get_project(project_id)
        assert project["status"] == "paused"


# ---------- Helper for direct DAG execution tests ----------

DIAMOND_DAG_SNAPSHOT = _dag_json(
    [
        {"id": "t1", "title": "Root", "description": "Root task", "dependencies": []},
        {"id": "t2", "title": "Left", "description": "Left branch", "dependencies": ["t1"]},
        {"id": "t3", "title": "Right", "description": "Right branch", "dependencies": ["t1"]},
        {"id": "t4", "title": "Join", "description": "Join task", "dependencies": ["t2", "t3"]},
    ]
)

FOUR_INDEPENDENT_DAG_SNAPSHOT = _dag_json(
    [
        {"id": "t1", "title": "Task 1", "description": "Independent 1", "dependencies": []},
        {"id": "t2", "title": "Task 2", "description": "Independent 2", "dependencies": []},
        {"id": "t3", "title": "Task 3", "description": "Independent 3", "dependencies": []},
        {"id": "t4", "title": "Task 4", "description": "Independent 4", "dependencies": []},
    ]
)


async def _setup_project_with_dag(state, dag_snapshot, project_id="test-proj"):
    """Create a project and tasks from a DAG snapshot, ready for _execute_dag."""
    await state.create_project(project_id, "Test goal")
    await state.save_dag_snapshot(project_id, dag_snapshot)
    dag = DAGEngine.from_snapshot(dag_snapshot)
    await state.create_tasks(project_id, [node.to_dict() for node in dag.nodes.values()])
    return dag


class TestParallelExecution:
    @pytest.fixture
    async def setup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            state = StateManager(db_path)
            await state.initialize()
            yield state
            await state.close()

    async def test_parallel_basic(self, setup) -> None:
        """Diamond DAG with concurrency=2. All tasks should complete."""
        state = setup
        dag = await _setup_project_with_dag(state, DIAMOND_DAG_SNAPSHOT)

        agent = MockAgent(delay=0.01)
        # t1, t2, t3, t4 — 4 executions
        for i in range(4):
            agent.add_result(AgentResult(success=True, output=f"Result {i}"))

        executor = Executor(state, agent, Monitor(state), supervised=False, concurrency=2)
        await executor._execute_dag("test-proj", dag)

        project = await state.get_project("test-proj")
        assert project["status"] == "completed"

        tasks = await state.get_tasks("test-proj")
        assert all(t["status"] == "completed" for t in tasks)
        assert len(agent.calls) == 4

    async def test_parallel_concurrency_limit(self, setup) -> None:
        """4 independent tasks with concurrency=2 should execute in 2 batches."""
        state = setup
        dag = await _setup_project_with_dag(state, FOUR_INDEPENDENT_DAG_SNAPSHOT)

        agent = MockAgent(delay=0.05)
        for i in range(4):
            agent.add_result(AgentResult(success=True, output=f"Result {i}"))

        executor = Executor(state, agent, Monitor(state), supervised=False, concurrency=2)
        await executor._execute_dag("test-proj", dag)

        project = await state.get_project("test-proj")
        assert project["status"] == "completed"

        # With delay, concurrency_log should show max 2 concurrent at any point
        assert max(agent.concurrency_log) <= 2
        assert len(agent.calls) == 4

    async def test_parallel_single_concurrency(self, setup) -> None:
        """concurrency=1 should behave the same as Phase 1 sequential execution."""
        state = setup
        dag = await _setup_project_with_dag(state, DIAMOND_DAG_SNAPSHOT)

        agent = MockAgent()
        for i in range(4):
            agent.add_result(AgentResult(success=True, output=f"Result {i}"))

        executor = Executor(state, agent, Monitor(state), supervised=False, concurrency=1)
        await executor._execute_dag("test-proj", dag)

        project = await state.get_project("test-proj")
        assert project["status"] == "completed"

        tasks = await state.get_tasks("test-proj")
        assert all(t["status"] == "completed" for t in tasks)

        # With concurrency=1, max concurrency should always be 1
        assert max(agent.concurrency_log) == 1

    async def test_parallel_batch_partial_failure(self, setup) -> None:
        """In a batch, t2 succeeds and t3 fails. t2→COMPLETED, t3→FAILED, t4 not run."""
        state = setup
        dag = await _setup_project_with_dag(state, DIAMOND_DAG_SNAPSHOT)

        agent = MockAgent(delay=0.01)
        # t1 succeeds
        agent.add_result(AgentResult(success=True, output="Root done"))
        # t2 succeeds, t3 fails (batch)
        agent.add_result(AgentResult(success=True, output="Left done"))
        agent.add_result(AgentResult(success=False, output="", error="Right failed"))

        executor = Executor(state, agent, Monitor(state), supervised=False, concurrency=2)
        await executor._execute_dag("test-proj", dag)

        project = await state.get_project("test-proj")
        assert project["status"] == "failed"

        tasks = {t["id"]: t["status"] for t in await state.get_tasks("test-proj")}
        assert tasks["t1"] == "completed"
        assert tasks["t2"] == "completed"
        assert tasks["t3"] == "failed"
        assert tasks["t4"] == "pending"  # never reached

    async def test_parallel_supervised_checkpoint(self, setup) -> None:
        """Supervised mode with parallel batch should show one checkpoint per batch."""
        state = setup
        dag = await _setup_project_with_dag(state, DIAMOND_DAG_SNAPSHOT)

        agent = MockAgent(delay=0.01)
        for i in range(4):
            agent.add_result(AgentResult(success=True, output=f"Result {i}"))

        executor = Executor(state, agent, Monitor(state), supervised=True, concurrency=2)

        # Checkpoint after t1 (batch of 1) → continue
        # Checkpoint after t2+t3 (batch of 2) → continue
        # t4 completes, project done — no checkpoint
        inputs = iter(["c", "c"])
        with patch("builtins.input", side_effect=inputs):
            await executor._execute_dag("test-proj", dag)

        project = await state.get_project("test-proj")
        assert project["status"] == "completed"

    async def test_parallel_context_passing(self, setup) -> None:
        """Parallel tasks should receive context from their completed dependencies."""
        state = setup
        dag = await _setup_project_with_dag(state, DIAMOND_DAG_SNAPSHOT)

        agent = MockAgent(delay=0.01)
        agent.add_result(AgentResult(success=True, output="Root output"))
        agent.add_result(AgentResult(success=True, output="Left output"))
        agent.add_result(AgentResult(success=True, output="Right output"))
        agent.add_result(AgentResult(success=True, output="Join output"))

        executor = Executor(state, agent, Monitor(state), supervised=False, concurrency=2)
        await executor._execute_dag("test-proj", dag)

        # t2 and t3 should have received context from t1
        # calls: [(prompt, context), ...]
        # call 0 = t1 (no context)
        assert agent.calls[0][1] is None

        # calls 1 and 2 are t2 and t3 (parallel), both depend on t1
        for call in agent.calls[1:3]:
            assert call[1] is not None
            assert "Root output" in call[1]

        # call 3 = t4, depends on t2 and t3
        t4_context = agent.calls[3][1]
        assert t4_context is not None
        assert "Left output" in t4_context or "Right output" in t4_context

    async def test_parallel_resume(self, setup) -> None:
        """Resume after parallel failure should skip completed tasks and finish."""
        state = setup
        dag = await _setup_project_with_dag(state, DIAMOND_DAG_SNAPSHOT)

        # First run: t1 completes, t2 succeeds but t3 fails
        agent1 = MockAgent(delay=0.01)
        agent1.add_result(AgentResult(success=True, output="Root done"))
        agent1.add_result(AgentResult(success=True, output="Left done"))
        agent1.add_result(AgentResult(success=False, output="", error="Right failed"))

        executor1 = Executor(state, agent1, Monitor(state), supervised=False, concurrency=2)
        await executor1._execute_dag("test-proj", dag)

        project = await state.get_project("test-proj")
        assert project["status"] == "failed"

        # Prepare for resume: reset failed tasks to pending, set project paused
        await state.update_project_status("test-proj", ProjectStatus.PAUSED)
        await state.reset_running_tasks("test-proj")
        # Manually reset failed task t3 to pending for resume
        await state.update_task_status("test-proj", "t3", TaskStatus.PENDING)

        # Resume with a new agent
        agent2 = MockAgent()
        agent2.add_result(AgentResult(success=True, output="Right done (retry)"))
        agent2.add_result(AgentResult(success=True, output="Join done"))

        executor2 = Executor(state, agent2, Monitor(state), supervised=False, concurrency=2)

        # Rebuild DAG and re-execute
        dag2 = DAGEngine.from_snapshot(DIAMOND_DAG_SNAPSHOT)
        completed = await state.get_completed_task_ids("test-proj")
        # t1, t2 already completed
        assert "t1" in completed
        assert "t2" in completed

        await state.update_project_status("test-proj", ProjectStatus.ACTIVE)
        await executor2._execute_dag("test-proj", dag2)

        project = await state.get_project("test-proj")
        assert project["status"] == "completed"

        tasks = {t["id"]: t["status"] for t in await state.get_tasks("test-proj")}
        assert all(s == "completed" for s in tasks.values())

        # agent2 should only have been called for t3 and t4
        assert len(agent2.calls) == 2
