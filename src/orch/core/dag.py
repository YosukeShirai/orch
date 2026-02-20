"""DAG engine built on networkx for task dependency management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from orch.agents.base import TaskStatus


@dataclass
class TaskNode:
    id: str
    title: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskNode:
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            dependencies=data.get("dependencies", []),
            metadata=data.get("metadata", {}),
        )


class DAGValidationError(Exception):
    pass


class DAGEngine:
    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._nodes: dict[str, TaskNode] = {}

    @property
    def nodes(self) -> dict[str, TaskNode]:
        return dict(self._nodes)

    def build(self, tasks: list[TaskNode]) -> None:
        """Build the DAG from a list of task nodes."""
        self._graph.clear()
        self._nodes.clear()

        for task in tasks:
            self._nodes[task.id] = task
            self._graph.add_node(task.id)

        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in self._nodes:
                    raise DAGValidationError(
                        f"Task '{task.id}' depends on unknown task '{dep_id}'"
                    )
                self._graph.add_edge(dep_id, task.id)

        self.validate()

    def validate(self) -> None:
        """Validate the DAG has no cycles."""
        if not nx.is_directed_acyclic_graph(self._graph):
            cycles = list(nx.simple_cycles(self._graph))
            raise DAGValidationError(f"DAG contains cycles: {cycles}")

    def get_ready_tasks(self, completed: set[str] | None = None) -> list[TaskNode]:
        """Get tasks whose dependencies are all satisfied."""
        completed = completed or set()
        ready = []
        for node_id, task in self._nodes.items():
            if node_id in completed:
                continue
            if task.status in (TaskStatus.COMPLETED, TaskStatus.RUNNING):
                continue
            predecessors = set(self._graph.predecessors(node_id))
            if predecessors <= completed:
                ready.append(task)
        return ready

    def get_execution_order(self) -> list[str]:
        """Return a valid topological execution order."""
        return list(nx.topological_sort(self._graph))

    def to_snapshot(self) -> str:
        """Serialize the DAG to a JSON string for storage."""
        tasks = [node.to_dict() for node in self._nodes.values()]
        return json.dumps(tasks, ensure_ascii=False)

    @classmethod
    def from_snapshot(cls, snapshot: str) -> DAGEngine:
        """Reconstruct a DAGEngine from a stored snapshot."""
        raw = json.loads(snapshot)
        if isinstance(raw, dict) and "tasks" in raw:
            tasks_data = raw["tasks"]
        else:
            tasks_data = raw
        tasks = [TaskNode.from_dict(d) for d in tasks_data]
        engine = cls()
        engine.build(tasks)
        return engine

    def to_display_string(self) -> str:
        """Generate a human-readable representation of the DAG."""
        order = self.get_execution_order()
        lines = []
        for i, node_id in enumerate(order, 1):
            task = self._nodes[node_id]
            deps = ""
            if task.dependencies:
                deps = f" (depends on: {', '.join(task.dependencies)})"
            status_icon = {
                TaskStatus.PENDING: "[ ]",
                TaskStatus.RUNNING: "[~]",
                TaskStatus.COMPLETED: "[x]",
                TaskStatus.FAILED: "[!]",
                TaskStatus.SKIPPED: "[-]",
            }.get(task.status, "[ ]")
            lines.append(f"  {status_icon} {task.id}: {task.title}{deps}")
            if task.description:
                lines.append(f"       {task.description}")
        return "\n".join(lines)

    def task_count(self) -> int:
        return len(self._nodes)
