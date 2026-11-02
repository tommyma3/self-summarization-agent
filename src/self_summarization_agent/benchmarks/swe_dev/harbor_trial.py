"""Harbor-compatible fresh-verification hook for the pinned Harbor release."""
import asyncio
from pathlib import Path
import tempfile

from harbor.trial.trial import Trial
from harbor.models.verifier.result import VerifierResult

from .verification import CandidateError, export_candidate, verify_docker


class SWETrial(Trial):
    def __init__(self, config, *, swe_task):
        super().__init__(config)
        self.swe_task = swe_task
        self.swe_evidence = {}
        if self._task.config.environment.docker_image != swe_task["image"]:
            raise ValueError("Harbor rollout image differs from manifest's pinned base image")
        if self._task.config.environment.allow_internet:
            raise ValueError("SWE text-only profile requires network disabled")

    async def _verify_with_retry(self):
        # Stop all workload processes before host-side copying, never invoke
        # candidate git/config/hooks to export. Harbor cleans up even on error.
        await self._environment._run_docker_compose_command(["stop", "main"], check=True)
        with tempfile.TemporaryDirectory(prefix="swe-candidate-") as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            await self._environment.download_dir("/app/repo", root)
            try:
                self.swe_evidence = await asyncio.to_thread(export_candidate, root,
                    Path(self.swe_task["path"]) / "environment/repo", self.trial_dir / "candidate")
            except CandidateError as exc:
                self.swe_evidence = dict(classification="candidate_export_failure", error=str(exc))
                self.result.verifier_result = VerifierResult(rewards={"reward": 0})
                return
        snapshot = self.swe_evidence["snapshot"]["path"]
        outcome = await asyncio.to_thread(verify_docker, self.swe_task, snapshot,
                                          self.trial_dir / "trusted-verifier")
        self.swe_evidence["verification"] = outcome
        if outcome.get("exception"):
            raise RuntimeError("Trusted verifier failed: " + str(outcome["exception"]))
        self.swe_evidence["base_commit"] = self.swe_task["source"]["base_commit"]
        self.result.verifier_result = VerifierResult(rewards=outcome["rewards"])
