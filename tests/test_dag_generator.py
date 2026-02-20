"""Tests for DAG generator and JSON parsing."""

from __future__ import annotations

import json

import pytest

from orch.agents.base import AgentResult
from orch.planner.dag_generator import DAGGenerationError, DAGGenerator
from tests.conftest import MockAgent


VALID_DAG_JSON = json.dumps(
    {
        "tasks": [
            {
                "id": "task-1",
                "title": "Initialize",
                "description": "Set up the project",
                "dependencies": [],
            },
            {
                "id": "task-2",
                "title": "Implement",
                "description": "Build the feature",
                "dependencies": ["task-1"],
            },
        ]
    }
)

VALID_DAG_IN_MARKDOWN = f"""
Here is the plan:

```json
{VALID_DAG_JSON}
```

This should work well.
"""


class TestDAGGenerator:
    async def test_generate_from_clean_json(self) -> None:
        agent = MockAgent([AgentResult(success=True, output=VALID_DAG_JSON)])
        generator = DAGGenerator(agent)
        dag = await generator.generate("Build a TODO app")
        assert dag.task_count() == 2

    async def test_generate_from_markdown_wrapped_json(self) -> None:
        agent = MockAgent([AgentResult(success=True, output=VALID_DAG_IN_MARKDOWN)])
        generator = DAGGenerator(agent)
        dag = await generator.generate("Build a TODO app")
        assert dag.task_count() == 2

    async def test_generate_retries_on_bad_json(self) -> None:
        agent = MockAgent(
            [
                AgentResult(success=True, output="not json at all"),
                AgentResult(success=True, output=VALID_DAG_JSON),
            ]
        )
        generator = DAGGenerator(agent)
        dag = await generator.generate("Build something")
        assert dag.task_count() == 2
        assert len(agent.calls) == 2

    async def test_generate_fails_after_max_retries(self) -> None:
        agent = MockAgent(
            [
                AgentResult(success=True, output="bad"),
                AgentResult(success=True, output="still bad"),
                AgentResult(success=True, output="still bad"),
            ]
        )
        generator = DAGGenerator(agent)
        with pytest.raises(DAGGenerationError):
            await generator.generate("Build something")

    async def test_generate_fails_on_agent_error(self) -> None:
        agent = MockAgent(
            [
                AgentResult(success=False, output="", error="API error"),
                AgentResult(success=False, output="", error="API error"),
                AgentResult(success=False, output="", error="API error"),
            ]
        )
        generator = DAGGenerator(agent)
        with pytest.raises(DAGGenerationError, match="Agent failed"):
            await generator.generate("Build something")

    async def test_generate_with_json_array(self) -> None:
        tasks_array = json.dumps(
            [
                {"id": "task-1", "title": "Only", "description": "Single task", "dependencies": []},
            ]
        )
        agent = MockAgent([AgentResult(success=True, output=tasks_array)])
        generator = DAGGenerator(agent)
        dag = await generator.generate("Simple task")
        assert dag.task_count() == 1

    async def test_generate_validates_dag(self) -> None:
        """Cyclic dependencies should cause a retry."""
        cyclic = json.dumps(
            {
                "tasks": [
                    {"id": "t1", "title": "A", "description": "", "dependencies": ["t2"]},
                    {"id": "t2", "title": "B", "description": "", "dependencies": ["t1"]},
                ]
            }
        )
        agent = MockAgent(
            [
                AgentResult(success=True, output=cyclic),
                AgentResult(success=True, output=cyclic),
                AgentResult(success=True, output=cyclic),
            ]
        )
        generator = DAGGenerator(agent)
        with pytest.raises(DAGGenerationError):
            await generator.generate("Something")
