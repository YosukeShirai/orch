"""Executor: orchestrates the full run/resume flow."""

from __future__ import annotations

import asyncio
import signal
import uuid
from typing import Any

from rich.console import Console
from rich.panel import Panel

from orch.agents.base import BaseAgent, ProjectStatus, TaskStatus
from orch.core.dag import DAGEngine
from orch.core.monitor import EventType, Monitor
from orch.core.scheduler import Scheduler
from orch.core.state import StateManager
from orch.planner.dag_generator import DAGGenerator

console = Console()


class ExecutionInterrupted(Exception):
    pass


class Executor:
    def __init__(
        self,
        state: StateManager,
        agent: BaseAgent,
        monitor: Monitor,
        supervised: bool = True,
        concurrency: int = 1,
    ) -> None:
        self._state = state
        self._agent = agent
        self._monitor = monitor
        self._supervised = supervised
        self._concurrency = concurrency
        self._interrupted = False
        self._current_project_id: str | None = None
        self._current_task_id: str | None = None
        self._running_task_ids: set[str] = set()

    async def run(self, goal: str) -> str:
        """Run a new project from a goal. Returns project_id."""
        project_id = str(uuid.uuid4())[:8]
        self._current_project_id = project_id

        # Set up SIGINT handler
        self._setup_signal_handler()

        await self._state.create_project(project_id, goal)
        await self._monitor.log(project_id, EventType.PROJECT_CREATED, data={"goal": goal})

        console.print(f"\n[bold]Project:[/bold] {project_id}")
        console.print(f"[bold]Goal:[/bold] {goal}\n")

        # Generate DAG
        console.print("[dim]Generating task plan...[/dim]")
        generator = DAGGenerator(self._agent)
        dag = await generator.generate(goal)
        await self._monitor.log(project_id, EventType.DAG_GENERATED)

        # Approval loop
        approved = await self._approval_loop(dag, generator, goal)
        if not approved:
            await self._state.update_project_status(project_id, ProjectStatus.CANCELLED)
            console.print("[yellow]Project cancelled.[/yellow]")
            return project_id

        # Save DAG and create task records
        await self._state.save_dag_snapshot(project_id, dag.to_snapshot())
        await self._state.create_tasks(
            project_id,
            [node.to_dict() for node in dag.nodes.values()],
        )
        await self._monitor.log(project_id, EventType.DAG_APPROVED)

        # Execute
        await self._execute_dag(project_id, dag)
        return project_id

    async def resume(self) -> str:
        """Resume the most recent active/paused project. Returns project_id."""
        project = await self._state.get_resumable_project()
        if not project:
            console.print("[red]No resumable project found.[/red]")
            raise RuntimeError("No resumable project found")

        project_id = project["id"]
        self._current_project_id = project_id
        self._setup_signal_handler()

        console.print(f"\n[bold]Resuming project:[/bold] {project_id}")
        console.print(f"[bold]Goal:[/bold] {project['goal']}\n")

        # Reset crashed tasks
        reset_count = await self._state.reset_running_tasks(project_id)
        if reset_count > 0:
            console.print(f"[yellow]Reset {reset_count} interrupted task(s) to pending.[/yellow]")

        # Rebuild DAG
        snapshot = await self._state.get_dag_snapshot(project_id)
        if not snapshot:
            raise RuntimeError(f"No DAG snapshot found for project {project_id}")
        dag = DAGEngine.from_snapshot(snapshot)

        await self._state.update_project_status(project_id, ProjectStatus.ACTIVE)
        await self._monitor.log(project_id, EventType.PROJECT_RESUMED)

        await self._execute_dag(project_id, dag)
        return project_id

    async def _execute_dag(self, project_id: str, dag: DAGEngine) -> None:
        """Execute tasks in DAG order."""
        scheduler = Scheduler(dag)
        completed = await self._state.get_completed_task_ids(project_id)

        done, total = scheduler.get_progress(completed)
        if done > 0:
            console.print(f"[dim]Progress: {done}/{total} tasks completed[/dim]\n")

        while not scheduler.is_complete(completed):
            if self._interrupted:
                await self._handle_interruption(project_id)
                return

            ready = scheduler.get_next_tasks(completed)
            if not ready:
                console.print("[red]No tasks ready but not all completed. Possible dependency issue.[/red]")
                await self._state.update_project_status(project_id, ProjectStatus.FAILED)
                return

            if self._concurrency <= 1:
                success = await self._execute_task_sequential(
                    project_id, scheduler, ready[0], completed
                )
            else:
                batch = ready[: self._concurrency]
                success = await self._execute_task_batch(
                    project_id, scheduler, batch, completed
                )

            if not success:
                return

            # Checkpoint (supervised mode)
            if self._supervised and not scheduler.is_complete(completed):
                action = await self._checkpoint()
                if action == "pause":
                    await self._state.update_project_status(project_id, ProjectStatus.PAUSED)
                    await self._monitor.log(project_id, EventType.PROJECT_PAUSED)
                    console.print("[yellow]Project paused. Use 'orch resume' to continue.[/yellow]")
                    return
                elif action == "quit":
                    await self._state.update_project_status(project_id, ProjectStatus.CANCELLED)
                    console.print("[yellow]Project cancelled.[/yellow]")
                    return

        await self._state.update_project_status(project_id, ProjectStatus.COMPLETED)
        await self._monitor.log(project_id, EventType.PROJECT_COMPLETED)
        console.print("[bold green]All tasks completed! Project finished.[/bold green]")

    async def _execute_task_sequential(
        self,
        project_id: str,
        scheduler: Scheduler,
        task: TaskNode,
        completed: set[str],
    ) -> bool:
        """Execute a single task sequentially. Returns True if successful."""
        self._current_task_id = task.id

        done, total = scheduler.get_progress(completed)
        console.print(
            Panel(
                f"[bold]{task.title}[/bold]\n{task.description}",
                title=f"Task {task.id} ({done + 1}/{total})",
                border_style="blue",
            )
        )

        context = await self._build_context(project_id, task.dependencies)

        await self._state.update_task_status(project_id, task.id, TaskStatus.RUNNING)
        await self._monitor.log(project_id, EventType.TASK_STARTED, task_id=task.id)

        result = await self._agent.execute(task.description, context)

        if result.success:
            await self._state.update_task_status(
                project_id, task.id, TaskStatus.COMPLETED, result
            )
            await self._monitor.log(
                project_id, EventType.TASK_COMPLETED, task_id=task.id
            )
            completed.add(task.id)
            console.print(f"[green]  Task {task.id} completed.[/green]\n")
            return True
        else:
            await self._state.update_task_status(
                project_id, task.id, TaskStatus.FAILED, result
            )
            await self._monitor.log(
                project_id,
                EventType.TASK_FAILED,
                task_id=task.id,
                data={"error": result.error},
            )
            console.print(f"[red]  Task {task.id} failed: {result.error}[/red]\n")
            await self._state.update_project_status(project_id, ProjectStatus.FAILED)
            return False

    async def _execute_task_batch(
        self,
        project_id: str,
        scheduler: Scheduler,
        batch: list[TaskNode],
        completed: set[str],
    ) -> bool:
        """Execute a batch of tasks in parallel. Returns True if all succeeded."""
        done, total = scheduler.get_progress(completed)
        task_names = ", ".join(t.id for t in batch)
        console.print(
            Panel(
                "\n".join(f"[bold]{t.title}[/bold]: {t.description}" for t in batch),
                title=f"Batch ({done + 1}-{done + len(batch)}/{total}): {task_names}",
                border_style="blue",
            )
        )

        # Mark all tasks as RUNNING
        for task in batch:
            self._running_task_ids.add(task.id)
            await self._state.update_task_status(project_id, task.id, TaskStatus.RUNNING)
            await self._monitor.log(project_id, EventType.TASK_STARTED, task_id=task.id)

        # Build coroutines
        async def _run_one(task: TaskNode) -> tuple[TaskNode, AgentResult]:
            context = await self._build_context(project_id, task.dependencies)
            result = await self._agent.execute(task.description, context)
            return task, result

        results = await asyncio.gather(
            *(_run_one(task) for task in batch), return_exceptions=True
        )

        # Process results
        all_ok = True
        for entry in results:
            if isinstance(entry, BaseException):
                # Unexpected exception — treat as failure for the whole batch
                all_ok = False
                continue

            task, result = entry
            self._running_task_ids.discard(task.id)

            if result.success:
                await self._state.update_task_status(
                    project_id, task.id, TaskStatus.COMPLETED, result
                )
                await self._monitor.log(
                    project_id, EventType.TASK_COMPLETED, task_id=task.id
                )
                completed.add(task.id)
                console.print(f"[green]  Task {task.id} completed.[/green]")
            else:
                await self._state.update_task_status(
                    project_id, task.id, TaskStatus.FAILED, result
                )
                await self._monitor.log(
                    project_id,
                    EventType.TASK_FAILED,
                    task_id=task.id,
                    data={"error": result.error},
                )
                console.print(f"[red]  Task {task.id} failed: {result.error}[/red]")
                all_ok = False

        console.print()

        if not all_ok:
            await self._state.update_project_status(project_id, ProjectStatus.FAILED)

        return all_ok

    async def _build_context(self, project_id: str, dependency_ids: list[str]) -> str | None:
        """Build context string from completed dependency results."""
        if not dependency_ids:
            return None
        parts = []
        for dep_id in dependency_ids:
            result = await self._state.get_task_result(project_id, dep_id)
            if result:
                parts.append(f"## Result from {dep_id}:\n{result}")
        return "\n\n".join(parts) if parts else None

    async def _approval_loop(
        self, dag: DAGEngine, generator: DAGGenerator, goal: str
    ) -> bool:
        """Show DAG and get user approval. Returns True if approved."""
        while True:
            console.print(
                Panel(
                    dag.to_display_string(),
                    title="Task Plan",
                    border_style="cyan",
                )
            )
            console.print("[a]pprove / [r]egenerate / [c]ancel: ", end="")
            choice = input().strip().lower()

            if choice in ("a", "approve"):
                return True
            elif choice in ("r", "regenerate"):
                console.print("[dim]Regenerating...[/dim]")
                dag = await generator.generate(goal)
            elif choice in ("c", "cancel"):
                return False
            else:
                console.print("[yellow]Invalid choice. Use a/r/c.[/yellow]")

    async def _checkpoint(self) -> str:
        """Show checkpoint prompt. Returns 'continue', 'pause', or 'quit'."""
        console.print("[dim][c]ontinue / [p]ause / [q]uit:[/dim] ", end="")
        choice = input().strip().lower()
        if choice in ("p", "pause"):
            return "pause"
        elif choice in ("q", "quit"):
            return "quit"
        return "continue"

    def _setup_signal_handler(self) -> None:
        """Install SIGINT handler for graceful interruption."""
        original = signal.getsignal(signal.SIGINT)

        def handler(signum: int, frame: Any) -> None:
            if self._interrupted:
                # Second Ctrl+C: force exit
                signal.signal(signal.SIGINT, original)
                raise KeyboardInterrupt
            self._interrupted = True
            console.print("\n[yellow]Interrupt received. Finishing current task...[/yellow]")

        signal.signal(signal.SIGINT, handler)

    async def _handle_interruption(self, project_id: str) -> None:
        """Handle graceful interruption."""
        # Mark all currently running tasks as FAILED
        task_ids_to_fail = set(self._running_task_ids)
        if self._current_task_id:
            task_ids_to_fail.add(self._current_task_id)
        for task_id in task_ids_to_fail:
            await self._state.update_task_status(
                project_id, task_id, TaskStatus.FAILED
            )
        self._running_task_ids.clear()
        await self._state.update_project_status(project_id, ProjectStatus.PAUSED)
        await self._monitor.log(project_id, EventType.PROJECT_PAUSED)
        console.print("[yellow]Project paused. Use 'orch resume' to continue.[/yellow]")
