"""Deterministic host-only pilot. Never use for untrusted policy execution.

Commands are restricted to the fixture's two fixed edits and an inspection.
This exercises EpisodeRuntime and the shared collector without Docker/tmux.
"""
import asyncio
from pathlib import Path
import shutil
import re
import subprocess
from types import SimpleNamespace

from .verification import export_candidate, verify_local_fixture

INSPECT_COMMAND = "cat names.py\n"
FIX_COMMAND = "printf 'def normalize_name(value):\n    return value.strip()\n' > names.py\n"
WRONG_COMMAND = "printf 'def normalize_name(value):\n    return value.upper()\n' > names.py\n"


class FixtureSession:
    def __init__(self, *, environment, **kwargs):
        self.root = environment.root
        self.output = ""
    async def start(self):
        pass
    async def stop(self):
        pass
    async def send_keys(self, keys, **kwargs):
        marker = re.fullmatch(r"printf '\\n%s\\n' '(__KIRA_DONE_[a-f0-9]+__)'\n", keys)
        if marker:
            self.output += "\n" + marker[1] + "\n"
            return
        if keys not in {INSPECT_COMMAND, FIX_COMMAND, WRONG_COMMAND, "", "C-c"}:
            raise ValueError("Local fixture rejects non-scripted terminal commands")
        if keys in {"", "C-c"}:
            return
        result = subprocess.run(["bash", "--noprofile", "--norc", "-c", keys], cwd=self.root,
                                capture_output=True, text=True, timeout=5)
        self.output += result.stdout + result.stderr
    async def capture_pane(self):
        return self.output
    async def get_incremental_output(self):
        result, self.output = self.output, ""
        return result


class FixtureTrial:
    def __init__(self, config, *, swe_task):
        from ..terminal_bench.harbor_agent import TerminusKiraTito
        self.config, self.task = config, swe_task
        self.trial_dir = config.trials_dir / config.trial_name
        root = self.trial_dir / "sandbox"
        shutil.copytree(Path(swe_task["path"]) / "environment/repo", root)
        self.environment = SimpleNamespace(root=root)
        self.agent = TerminusKiraTito(logs_dir=self.trial_dir / "agent", **config.agent.kwargs)
        self.swe_evidence = {}

    async def run(self):
        from harbor.models.agent.context import AgentContext
        await self.agent.setup(self.environment)
        await self.agent.run(Path(self.task["path"], "instruction.md").read_text(), self.environment, AgentContext())
        self.swe_evidence = await asyncio.to_thread(export_candidate, self.environment.root,
            Path(self.task["path"]) / "environment/repo", self.trial_dir / "candidate")
        result = await asyncio.to_thread(verify_local_fixture, self.task,
            self.swe_evidence["snapshot"]["path"], self.trial_dir / "trusted-verifier")
        self.swe_evidence["verification"] = result
        return SimpleNamespace(task_checksum=self.task["checksum"], exception_info=None,
                               verifier_result=SimpleNamespace(rewards=result["rewards"]))
