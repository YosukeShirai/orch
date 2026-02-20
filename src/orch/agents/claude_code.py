"""Claude Code CLI agent implementation."""

from __future__ import annotations

import asyncio
import json

from orch.agents.base import AgentResult, BaseAgent


class ClaudeCodeAgent(BaseAgent):
    async def execute(self, prompt: str, context: str | None = None) -> AgentResult:
        """Execute a task via `claude -p` subprocess."""
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n---\n\n{prompt}"

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude",
                "-p",
                full_prompt,
                "--output-format",
                "json",
                "--allowedTools",
                "Edit",
                "Write",
                "Bash",
                "Read",
                "Glob",
                "Grep",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                return AgentResult(
                    success=False,
                    output=stdout_text,
                    error=f"Exit code {proc.returncode}: {stderr_text}",
                )

            result_text = stdout_text
            try:
                parsed = json.loads(stdout_text)
                if isinstance(parsed, dict) and "result" in parsed:
                    result_text = parsed["result"]
                elif isinstance(parsed, dict) and "content" in parsed:
                    result_text = parsed["content"]
            except json.JSONDecodeError:
                pass

            return AgentResult(success=True, output=result_text)

        except FileNotFoundError:
            return AgentResult(
                success=False,
                output="",
                error="'claude' command not found. Please install Claude Code CLI.",
            )
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                error=str(e),
            )
