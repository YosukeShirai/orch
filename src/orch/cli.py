"""CLI entry point for orch."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from orch.agents.base import ProjectStatus, TaskStatus
from orch.agents.claude_code import ClaudeCodeAgent
from orch.core.executor import Executor
from orch.core.monitor import Monitor
from orch.core.state import StateManager

app = typer.Typer(
    name="orch",
    help="Goal-driven task orchestrator powered by Claude Code",
    no_args_is_help=True,
)
console = Console()

DB_DIR = ".orch"
DB_FILE = "state.db"


def _get_db_path() -> str:
    db_dir = Path.cwd() / DB_DIR
    db_dir.mkdir(exist_ok=True)
    return str(db_dir / DB_FILE)


async def _run(goal: str, supervised: bool, concurrency: int = 2) -> None:
    state = StateManager(_get_db_path())
    await state.initialize()
    try:
        agent = ClaudeCodeAgent()
        monitor = Monitor(state)
        executor = Executor(
            state, agent, monitor, supervised=supervised, concurrency=concurrency
        )
        await executor.run(goal)
    finally:
        await state.close()


async def _resume(supervised: bool, concurrency: int = 2) -> None:
    state = StateManager(_get_db_path())
    await state.initialize()
    try:
        agent = ClaudeCodeAgent()
        monitor = Monitor(state)
        executor = Executor(
            state, agent, monitor, supervised=supervised, concurrency=concurrency
        )
        await executor.resume()
    finally:
        await state.close()


async def _status(project_id: str | None) -> None:
    state = StateManager(_get_db_path())
    await state.initialize()
    try:
        if project_id:
            project = await state.get_project(project_id)
            if not project:
                console.print(f"[red]Project '{project_id}' not found.[/red]")
                return
            _print_project_detail(project, await state.get_tasks(project_id))
        else:
            projects = await state.list_projects()
            if not projects:
                console.print("[dim]No projects found.[/dim]")
                return
            _print_projects_table(projects)

            # Show latest project details
            if projects:
                latest = projects[0]
                tasks = await state.get_tasks(latest["id"])
                if tasks:
                    console.print()
                    _print_project_detail(latest, tasks)
    finally:
        await state.close()


def _print_projects_table(projects: list[dict]) -> None:
    table = Table(title="Projects")
    table.add_column("ID", style="bold")
    table.add_column("Goal")
    table.add_column("Status")
    table.add_column("Updated")

    status_styles = {
        "active": "blue",
        "paused": "yellow",
        "completed": "green",
        "failed": "red",
        "cancelled": "dim",
    }

    for p in projects:
        status = p["status"]
        style = status_styles.get(status, "")
        table.add_row(
            p["id"],
            p["goal"][:60] + ("..." if len(p["goal"]) > 60 else ""),
            f"[{style}]{status}[/{style}]",
            p["updated_at"][:19],
        )
    console.print(table)


def _print_project_detail(project: dict, tasks: list[dict]) -> None:
    console.print(f"[bold]Project {project['id']}[/bold]: {project['goal']}")
    console.print(f"Status: {project['status']}\n")

    if not tasks:
        console.print("[dim]No tasks.[/dim]")
        return

    table = Table(title="Tasks")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Status")

    status_icons = {
        "pending": "[ ] pending",
        "running": "[~] running",
        "completed": "[x] completed",
        "failed": "[!] failed",
        "skipped": "[-] skipped",
    }

    status_styles = {
        "pending": "dim",
        "running": "blue",
        "completed": "green",
        "failed": "red",
        "skipped": "dim",
    }

    for t in tasks:
        status = t["status"]
        style = status_styles.get(status, "")
        table.add_row(
            t["id"],
            t["title"],
            f"[{style}]{status_icons.get(status, status)}[/{style}]",
        )
    console.print(table)


@app.command()
def run(
    goal: str = typer.Argument(..., help="The goal to achieve"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Run without checkpoints"),
    concurrency: int = typer.Option(
        2, "--concurrency", "-c", help="Max parallel tasks"
    ),
) -> None:
    """Start a new project from a goal."""
    asyncio.run(_run(goal, supervised=not auto, concurrency=concurrency))


@app.command()
def resume(
    auto: bool = typer.Option(False, "--auto", "-a", help="Run without checkpoints"),
    concurrency: int = typer.Option(
        2, "--concurrency", "-c", help="Max parallel tasks"
    ),
) -> None:
    """Resume the most recent paused/active project."""
    asyncio.run(_resume(supervised=not auto, concurrency=concurrency))


@app.command()
def status(
    project_id: str | None = typer.Argument(None, help="Project ID (shows latest if omitted)"),
) -> None:
    """Show project status and tasks."""
    asyncio.run(_status(project_id))


if __name__ == "__main__":
    app()
