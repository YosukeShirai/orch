"""DAG generation from a goal using Claude Code CLI."""

from __future__ import annotations

import json
import re

from orch.agents.base import BaseAgent
from orch.core.dag import DAGEngine, DAGValidationError, TaskNode

DAG_GENERATION_PROMPT = """\
あなたはプロジェクト計画AIです。以下のゴールを達成するためのタスク分解を行ってください。

## ゴール
{goal}

## 制約
- タスク数は3〜8個
- 各タスクにはsequentialなID（"task-1", "task-2", ...）を付与
- 各タスクの依存関係を明示
- 最初のタスクは依存関係なし

## 出力形式（JSON only、説明不要）
```json
{{
  "tasks": [
    {{
      "id": "task-1",
      "title": "タスクのタイトル",
      "description": "タスクの詳細な説明。何を実装するか具体的に記述。",
      "dependencies": []
    }},
    {{
      "id": "task-2",
      "title": "次のタスク",
      "description": "詳細説明",
      "dependencies": ["task-1"]
    }}
  ]
}}
```

JSONのみを出力してください。
"""

MAX_RETRIES = 3


class DAGGenerationError(Exception):
    pass


class DAGGenerator:
    def __init__(self, agent: BaseAgent) -> None:
        self._agent = agent

    async def generate(self, goal: str) -> DAGEngine:
        """Generate a DAG from a goal string."""
        prompt = DAG_GENERATION_PROMPT.format(goal=goal)

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            result = await self._agent.execute(prompt)
            if not result.success:
                last_error = DAGGenerationError(
                    f"Agent failed: {result.error}"
                )
                continue

            try:
                tasks_data = self._extract_json(result.output)
                tasks = [TaskNode.from_dict(t) for t in tasks_data]
                engine = DAGEngine()
                engine.build(tasks)
                return engine
            except (json.JSONDecodeError, KeyError, DAGValidationError) as e:
                last_error = DAGGenerationError(
                    f"Failed to parse DAG (attempt {attempt + 1}): {e}"
                )
                continue

        raise last_error or DAGGenerationError("DAG generation failed")

    def _extract_json(self, text: str) -> list[dict]:
        """Extract tasks JSON from agent output."""
        # Try parsing the entire text as JSON first
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tasks" in data:
                return data["tasks"]
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "tasks" in data:
                return data["tasks"]
            if isinstance(data, list):
                return data

        # Try finding a JSON object in the text
        match = re.search(r"\{.*\"tasks\".*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data["tasks"]

        raise json.JSONDecodeError("No valid JSON found in output", text, 0)
