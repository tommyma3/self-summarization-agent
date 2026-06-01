from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any

from self_summarization_agent.bcplus_backend import build_backend
from self_summarization_agent.config import load_train_config
from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.generation import build_generator
from self_summarization_agent.judge import JudgeDecision, RewardJudge
from self_summarization_agent.launcher_utils import build_runtime
from self_summarization_agent.rewards import apply_terminal_reward
from self_summarization_agent.runtime import EpisodeRuntime, parse_model_tool_call

try:  # OpenRLHF is intentionally optional for local unit tests.
    from openrlhf.utils.agent import AgentExecutorBase
except ImportError:  # pragma: no cover - exercised only when OpenRLHF is absent
    class AgentExecutorBase:  # type: ignore[no-redef]
        pass


@dataclass(slots=True)
class _JudgePayload:
    query_id: str
    query: str
    answer: str | None
    status: str
    response: str


@dataclass(slots=True)
class _AgentResources:
    config: Any
    backend: Any
    runtime: EpisodeRuntime
    local_judge: RewardJudge | None
    judge_url: str | None


class _UnusedModel:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("OpenRLHF agent executor calls vLLM directly")


_RESOURCES: _AgentResources | None = None


def _load_resources() -> _AgentResources:
    global _RESOURCES
    if _RESOURCES is not None:
        return _RESOURCES

    config_path = os.environ.get("SELF_SUMMARIZATION_OPENRLHF_CONFIG", "configs/train/default.yaml")
    config = load_train_config(config_path)
    backend = build_backend(config.experiment.bc_plus_root, config.retrieval)
    judge_url = os.environ.get("SELF_SUMMARIZATION_JUDGE_URL")
    local_judge = None if judge_url else RewardJudge(build_generator(config.model, judge_config=config.judge))
    runtime = build_runtime(_UnusedModel(), backend, config.runtime)
    _RESOURCES = _AgentResources(
        config=config,
        backend=backend,
        runtime=runtime,
        local_judge=local_judge,
        judge_url=judge_url,
    )
    return _RESOURCES


def _parse_label(label: Any) -> dict[str, Any]:
    if isinstance(label, dict):
        return dict(label)
    if isinstance(label, str):
        try:
            parsed = json.loads(label)
        except json.JSONDecodeError:
            return {"answer": label}
        if isinstance(parsed, dict):
            return parsed
        return {"answer": label}
    return {"answer": label}


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text=text, add_special_tokens=False, return_tensors="pt")
    return encoded["input_ids"][0].tolist()


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _logprobs_for_output(output: Any, token_ids: list[int]) -> list[float] | None:
    raw_logprobs = getattr(output, "logprobs", None)
    if raw_logprobs is None:
        return None
    values: list[float] = []
    for token_id, logprob_dict in zip(token_ids, raw_logprobs):
        token_logprob = logprob_dict.get(token_id) if hasattr(logprob_dict, "get") else None
        values.append(float(getattr(token_logprob, "logprob", 0.0)) if token_logprob is not None else 0.0)
    return values


def _completion_from_finish_tool(raw_output: str) -> tuple[str, str] | None:
    parsed = parse_model_tool_call(raw_output)
    if parsed is None:
        return None
    payload, normalized = parsed
    if payload.get("tool_name") != "finish":
        return None
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return None
    answer = arguments.get("answer")
    if not isinstance(answer, str):
        return None
    return answer, normalized


async def _evaluate_with_http(url: str, payload: _JudgePayload) -> JudgeDecision:
    def post() -> dict[str, Any]:
        data = json.dumps(
            {
                "query_id": payload.query_id,
                "query": payload.query,
                "answer": payload.answer,
                "status": payload.status,
                "response": payload.response,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        result = await asyncio.to_thread(post)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"OpenRLHF judge request failed: {exc}") from exc
    return JudgeDecision(
        outcome=str(result.get("outcome")),
        judge_prompt=result.get("judge_prompt"),
        judge_response=result.get("judge_response"),
        parse_error=bool(result.get("parse_error")),
    )


async def _evaluate_reward(resources: _AgentResources, payload: _JudgePayload) -> JudgeDecision:
    if resources.judge_url:
        return await _evaluate_with_http(resources.judge_url, payload)
    if resources.local_judge is None:
        raise RuntimeError("No OpenRLHF judge is configured")
    example = QueryExample(query_id=payload.query_id, query=payload.query, answer=payload.answer)
    return await asyncio.to_thread(resources.local_judge.evaluate, example, payload.status, payload.response)


class AgentExecutor(AgentExecutorBase):
    """OpenRLHF executor for sanitized BrowseComp+ self-summarization rollouts.

    Tool-call turns are generated by the policy but are treated as environment
    interaction. Only normalized tool calls and tool results enter later
    prompts. Summary and final-answer completions are returned as trainable
    action ranges and receive the same terminal reward.
    """

    async def execute(
        self,
        prompt: str,
        label: Any,
        sampling_params: Any,
        max_length: int,
        hf_tokenizer: Any,
        llm_engine: Any,
        images: Any = None,
    ) -> dict[str, Any]:
        del images
        resources = _load_resources()
        label_payload = _parse_label(label)
        query_id = str(label_payload.get("query_id") or label_payload.get("id") or "")
        answer = label_payload.get("answer")
        answer = str(answer) if answer is not None else None
        if not query_id:
            query_id = str(abs(hash(prompt)))

        runtime = resources.runtime
        active = runtime._new_active_episode(query_id, prompt)
        train_tokens: list[int] = []
        action_ranges: list[tuple[int, int]] = []
        rollout_log_probs: list[float] | None = [] if sampling_params.logprobs is not None else None
        summary_segments: list[tuple[str, str, list[int], list[float] | None]] = []
        terminal_completion: tuple[str, list[int], list[float] | None] | None = None
        extra_logs: dict[str, Any] = {}

        async def generate_for_prompt(model_prompt: str) -> tuple[str, list[int], list[float] | None]:
            prompt_tokens = _tokenize(hf_tokenizer, model_prompt)
            if len(prompt_tokens) >= max_length:
                prompt_tokens = prompt_tokens[-max(1, max_length - 1) :]
            effective_params = deepcopy(sampling_params)
            effective_params.max_tokens = max(1, max_length - len(prompt_tokens))
            request_output = await llm_engine.generate(prompt_tokens, effective_params)
            output = request_output.outputs[0]
            token_ids = list(output.token_ids)
            text = output.text if output.text is not None else _decode(hf_tokenizer, token_ids)
            return text, token_ids, _logprobs_for_output(output, token_ids)

        while active.result is None:
            if runtime.max_tool_calls is not None and sum(active.tool_call_counts.values()) >= runtime.max_tool_calls:
                active.result = runtime._budget_exhausted_result(
                    active.state.query_id,
                    active.summary_turns,
                    active.retrieved_docids,
                    active.tool_call_counts,
                    active.turn_records,
                )
                break

            acting_prompt = runtime._build_runtime_prompt(active.state)
            active.context_manager.assert_fits(acting_prompt)
            raw_action, action_token_ids, action_logprobs = await generate_for_prompt(acting_prompt)
            finish = _completion_from_finish_tool(raw_action)
            runtime._apply_action_output(active, raw_action)

            if active.result is not None:
                if active.result.status == "completed" and finish is not None:
                    _, normalized_finish = finish
                    terminal_tokens = _tokenize(hf_tokenizer, normalized_finish)
                    terminal_completion = (acting_prompt, terminal_tokens, None)
                break

            summary_request = runtime._build_summary_prompt_for_active(active)
            if summary_request is not None:
                summary_prompt, retired_count = summary_request
                summary_count_before = len(active.summary_turns)
                raw_summary, summary_token_ids, summary_logprobs = await generate_for_prompt(summary_prompt)
                runtime._apply_summary_output(active, summary_prompt, retired_count, raw_summary)
                if len(active.summary_turns) > summary_count_before:
                    summary_segments.append((summary_prompt, raw_summary, summary_token_ids, summary_logprobs))

        result = active.result
        if result is None:
            result = runtime._budget_exhausted_result(
                active.state.query_id,
                active.summary_turns,
                active.retrieved_docids,
                active.tool_call_counts,
                active.turn_records,
            )

        if result.status == "malformed_tool_call":
            summary_segments = []
            terminal_completion = None
            reward = -1.0
            score = 0.0
            outcome = "malformed_tool_call"
        else:
            judge_decision = await _evaluate_reward(
                resources,
                _JudgePayload(
                    query_id=query_id,
                    query=prompt,
                    answer=answer,
                    status=result.status,
                    response=result.final_answer or "",
                ),
            )
            outcome = judge_decision.outcome
            final_answer_turn_id = "final-answer" if result.final_answer is not None else None
            result.turn_rewards = apply_terminal_reward(
                outcome=outcome,
                summary_turn_ids=result.summary_turns,
                final_answer_turn_id=final_answer_turn_id,
            )
            reward = 1.0 if outcome == "correct_answer" else -1.0
            score = 1.0 if outcome == "correct_answer" else 0.0
            extra_logs.update(
                {
                    "judge_parse_error": judge_decision.parse_error,
                    "judge_outcome": outcome,
                }
            )

        for summary_prompt, _raw_summary, summary_token_ids, summary_logprobs in summary_segments:
            prompt_tokens = _tokenize(hf_tokenizer, summary_prompt)
            start = len(train_tokens) + len(prompt_tokens)
            train_tokens.extend(prompt_tokens)
            train_tokens.extend(summary_token_ids)
            action_ranges.append((start, start + len(summary_token_ids)))
            if rollout_log_probs is not None:
                rollout_log_probs.extend([0.0] * len(prompt_tokens))
                rollout_log_probs.extend(summary_logprobs or [0.0] * len(summary_token_ids))

        if terminal_completion is not None:
            final_prompt, terminal_tokens, terminal_logprobs = terminal_completion
            prompt_tokens = _tokenize(hf_tokenizer, final_prompt)
            start = len(train_tokens) + len(prompt_tokens)
            train_tokens.extend(prompt_tokens)
            train_tokens.extend(terminal_tokens)
            action_ranges.append((start, start + len(terminal_tokens)))
            if rollout_log_probs is not None:
                rollout_log_probs.extend([0.0] * len(prompt_tokens))
                rollout_log_probs.extend(terminal_logprobs or [0.0] * len(terminal_tokens))

        if not train_tokens:
            fallback_prompt = runtime._build_runtime_prompt(active.state)
            train_tokens = _tokenize(hf_tokenizer, fallback_prompt)
            if rollout_log_probs is not None:
                rollout_log_probs = [0.0] * len(train_tokens)

        extra_logs.update(
            {
                "query_id": query_id,
                "status": result.status,
                "outcome": outcome,
                "summary_turn_count": len(result.summary_turns),
                "tool_calls": dict(result.tool_call_counts),
                "trainable_action_ranges": len(action_ranges),
            }
        )
        return {
            "prompt": prompt,
            "label": label,
            "images": None,
            "mm_train_inputs": None,
            "reward": reward,
            "scores": score,
            "observation_tokens": train_tokens,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            "extra_logs": extra_logs,
        }
