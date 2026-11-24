"""Local tmux transport for the project-owned KIRA agent and interval runtime."""

import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from harbor.models.agent.context import AgentContext
from harbor.models.trial.result import ExceptionInfo

from .tmux_tasks import grade, task_env, worker_command, writable_workspace
from .verification import CandidateError, export_candidate


class LocalTmuxSession:
    def __init__(self, *, environment, **kwargs):
        self.environment = environment
        self.socket = str(environment.home / "terminal.sock")
        self.previous = []

    async def _tmux(self, *args, check=True):
        process = await asyncio.create_subprocess_exec(
            *worker_command(["tmux", "-S", self.socket, *args]),
            cwd=self.environment.repo,
            env=self.environment.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if check and process.returncode:
            raise RuntimeError(stderr.decode(errors="replace"))
        return stdout.decode(errors="replace").rstrip("\n")

    async def start(self):
        await self._tmux(
            "-f",
            "/dev/null",
            "new-session",
            "-d",
            "-s",
            "agent",
            "-x",
            "180",
            "-y",
            "40",
            "/bin/bash --noprofile --norc",
        )

    async def stop(self):
        await self._tmux("kill-server", check=False)

    async def send_keys(self, keys, **kwargs):
        if keys in {"C-c", "C-d", "C-z", "Escape", "Enter", "Tab", "Up", "Down"}:
            await self._tmux("send-keys", "-t", "agent", keys)
        else:
            await self._tmux("send-keys", "-t", "agent", "-l", "--", keys)

    async def capture_pane(self):
        return (
            await self._tmux("capture-pane", "-p", "-J", "-S", "-", "-t", "agent")
        ).rstrip()

    async def get_incremental_output(self):
        lines = (await self.capture_pane()).splitlines()
        # A shell prompt can be replaced in place as the next command is typed.
        overlap = min(len(self.previous), len(lines))
        while overlap and self.previous[-overlap:] != lines[:overlap]:
            overlap -= 1
        self.previous = lines[:-1]
        return "\n".join(lines[overlap:])


class LocalTmuxTrial:
    def __init__(self, config, *, swe_task):
        self.config = config
        self.task = swe_task
        self.trial_dir = config.trials_dir / config.trial_name
        self.swe_evidence = {}

    async def run(self):
        from ..terminal_bench.harbor_agent import (
            CollectionGenerationError,
            TerminusKiraTito,
        )

        bundle = Path(self.task["path"])
        spec = json.loads((bundle / "spec.json").read_text())
        rewards = exception = None
        try:
            with tempfile.TemporaryDirectory(prefix="swe-attempt-") as dirname:
                home = Path(dirname)
                repo = home / "repo"
                await asyncio.to_thread(shutil.copytree, bundle / "base", repo)
                writable_workspace(home)
                env = task_env(home, repo, bundle / "venv/bin/python")
                for args in (
                    ["init", "-q"],
                    ["add", "-f", "."],
                    [
                        "-c",
                        "user.name=Baseline",
                        "-c",
                        "user.email=baseline@local",
                        "commit",
                        "-qm",
                        "baseline",
                    ],
                ):
                    await asyncio.to_thread(
                        subprocess.run,
                        worker_command(["git", *args]),
                        cwd=repo,
                        env=env,
                        check=True,
                    )
                environment = SimpleNamespace(repo=repo, home=home, env=env)
                agent = TerminusKiraTito(
                    logs_dir=self.trial_dir / "agent", **self.config.agent.kwargs
                )
                try:
                    await agent.setup(environment)
                    async with asyncio.timeout(
                        self.config.agent.override_timeout_sec or 1800
                    ):
                        await agent.run(
                            spec["problem_statement"], environment, AgentContext()
                        )
                finally:
                    if hasattr(agent, "session"):
                        await agent.session.stop()
                self.swe_evidence = await asyncio.to_thread(
                    export_candidate,
                    repo,
                    bundle / "base",
                    self.trial_dir / "candidate",
                )
            outcome = await asyncio.to_thread(
                grade,
                bundle,
                self.swe_evidence["snapshot"]["path"],
                self.trial_dir / "verifier",
            )
            self.swe_evidence["verification"] = outcome
            if outcome["reward"] is None:
                raise ValueError(outcome["reason"])
            rewards = {"reward": outcome["reward"]}
        except CollectionGenerationError:
            raise
        except CandidateError as exc:
            self.swe_evidence["export_failure"] = str(exc)
            rewards = {"reward": 0}
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            exception = ExceptionInfo.from_exception(exc)
        return SimpleNamespace(
            task_checksum=self.task["checksum"],
            exception_info=exception,
            verifier_result=SimpleNamespace(rewards=rewards)
            if rewards is not None
            else None,
        )
