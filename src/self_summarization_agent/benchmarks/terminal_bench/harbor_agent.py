"""Harbor external-agent entry point; all policy turns use EpisodeRuntime."""
import asyncio
from concurrent.futures import CancelledError
import json
from pathlib import Path
import threading

from harbor.agents.base import BaseAgent
from harbor.agents.terminus_2.tmux_session import TmuxSession

from self_summarization_agent.config import BenchmarkConfig, RuntimeConfig, load_train_config
from self_summarization_agent.launcher_utils import serialize_runtime_result
from self_summarization_agent.runtime import EpisodeRuntime
from . import SCAFFOLD_VERSION
from .scaffold import TerminusKiraScaffold
from .tools import FrozenVisionService, TerminalBackend


# Owned by one collection invocation, released in its finally block. Harbor's
# serializable agent kwargs carry a key, never an opaque model object.
POLICIES = {}


class CollectionGenerationError(RuntimeError):
    """Inference protocol failures abort collection rather than receiving a reward."""


class LockedPolicy:
    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate_token_batch(self, requests):
        with self._lock:
            try:
                results = self.model.generate_token_batch(requests)
                if len(results) != len(requests) or any(
                    r.prompt_token_ids is None or r.completion_token_ids is None for r in results
                ):
                    raise ValueError("Missing authoritative generation token IDs")
                return results
            except Exception as exc:
                raise CollectionGenerationError(str(exc)) from exc


class TerminusKiraTito(BaseAgent):
    def __init__(self, *args, policy_key=None, config_path=None, query_id=None,
                 runtime=None, benchmark=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_id = query_id or self.logs_dir.parent.name
        if policy_key is not None:
            self.policy, self.vision = POLICIES[policy_key]
            self.runtime_config = RuntimeConfig(**(runtime or {}))
            self.benchmark_config = BenchmarkConfig(**(benchmark or {}))
        elif config_path:
            from self_summarization_agent.rollout_collection import _build_rollout_generator
            config = load_train_config(config_path)
            self.policy = LockedPolicy(_build_rollout_generator(config, Path(config.model.model_path), split="eval"))
            self.runtime_config = config.runtime
            self.benchmark_config = config.benchmark
            self.vision = make_vision(config.benchmark, Path(config.model.model_path))
        else:
            raise ValueError("Provide config_path or an active collection policy_key")
        self.result = None
        self.instruction = None

    @staticmethod
    def name():
        return "terminus-kira-tito"

    def version(self):
        return SCAFFOLD_VERSION

    async def setup(self, environment):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session = TmuxSession(
            session_name="kira", environment=environment,
            logging_path=Path("/logs/agent/terminal.log"),
            local_asciinema_recording_path=None, remote_asciinema_recording_path=None)
        await self.session.start()

    async def run(self, instruction, environment, context):
        self.instruction = instruction
        backend = TerminalBackend(self.session, loop=asyncio.get_running_loop(),
            logs_dir=self.logs_dir, timeout=self.runtime_config.tool_execution_timeout_seconds or 600,
            max_output_chars=self.benchmark_config.max_output_chars, vision=self.vision)
        scaffold = TerminusKiraScaffold(backend, image_enabled=self.vision is not None)
        rt = EpisodeRuntime(model=self.policy, backend=backend, scaffold=scaffold,
            context_threshold_tokens=self.runtime_config.context_threshold_tokens,
            max_context_tokens=self.runtime_config.max_context_tokens,
            max_summary_tokens=self.runtime_config.max_summary_tokens,
            max_tool_calls=self.runtime_config.tool_budget,
            generated_token_budget=self.runtime_config.generated_token_budget,
            token_counter=self.policy.count_tokens)
        active = rt._new_active_episode(self.query_id, instruction)
        try:
            while active.result is None:
                worker = asyncio.create_task(asyncio.to_thread(rt._advance_active_episodes, [active]))
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    backend.cancel()
                    # No concurrent finalization while inference owns the ledger.
                    try:
                        await worker
                    except (Exception, CancelledError):
                        pass
                    raise
                self._save_intervals(active.trajectory_records)
            self.result = active.result
        except BaseException:
            if active.token_ledger is not None and not active.token_ledger.finalized:
                rt._finalize_trajectory(active, active.state.messages,
                                        termination_kind="infrastructure_error")
            self.result = rt._penalized_result(active, status="infrastructure_error")
            raise
        finally:
            backend.cancel()
            if self.result is not None:
                self._save_intervals(self.result.trajectory_records)
                payload = serialize_runtime_result(self.result, query_text=instruction)
                (self.logs_dir / "rollout.json").write_text(json.dumps(payload, ensure_ascii=False))
                context.n_output_tokens = sum(len(g["completion_token_ids"])
                    for r in self.result.trajectory_records for g in r["collection_tokens"]["generations"])
                context.n_input_tokens = sum(len(g["prompt_token_ids"])
                    for r in self.result.trajectory_records for g in r["collection_tokens"]["generations"])
                context.metadata = {"scaffold": SCAFFOLD_VERSION, "status": self.result.status,
                    "image_profile": self.benchmark_config.image_profile,
                    "exact_token_records": "trajectory.records.jsonl"}
            try:
                await self.session.stop()
            except Exception:
                self.logger.exception("Terminal session cleanup failed")

    def _save_intervals(self, records):
        path = self.logs_dir / "trajectory.records.jsonl"
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        temporary.replace(path)


def make_vision(config, policy_checkpoint):
    if config.image_profile == "text-only":
        return None
    if Path(config.vision_model_path).resolve() == policy_checkpoint.resolve():
        raise ValueError("Vision service must use a separate frozen checkpoint, not the trainable policy")
    return FrozenVisionService(model_path=config.vision_model_path,
                               device=config.vision_device, max_tokens=config.vision_max_tokens)
