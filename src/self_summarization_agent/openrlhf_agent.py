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
from self_summarization_agent.rewards import apply_terminal_reward, trainable_turn_ids_from_records
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

    Tool-call turns are generated by the policy, normalized before entering
    later prompts, and returned as trainable action ranges alongside summary
    and final-answer completions.
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
        train_segments: list[tuple[str, list[int], list[float] | None]] = []
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
            remaining_tool_calls = runtime._remaining_tool_calls(active)
            if remaining_tool_calls == 0:
                acting_prompt = runtime._build_forced_answer_prompt(active)
                active.context_manager.assert_fits(acting_prompt)
                raw_action, action_token_ids, action_logprobs = await generate_for_prompt(acting_prompt)
                parsed_action = parse_model_tool_call(raw_action)
                if parsed_action is None:
                    train_segments.append((acting_prompt, action_token_ids, action_logprobs))
                else:
                    _, normalized_action = parsed_action
                    train_segments.append((acting_prompt, _tokenize(hf_tokenizer, normalized_action), None))
                runtime._apply_forced_answer_output(active, raw_action, acting_prompt)
                break

            acting_prompt = runtime._build_runtime_prompt(active.state)
            active.context_manager.assert_fits(acting_prompt)
            raw_action, action_token_ids, action_logprobs = await generate_for_prompt(acting_prompt)
            parsed_action = parse_model_tool_call(raw_action)
            if parsed_action is None:
                train_segments.append((acting_prompt, action_token_ids, action_logprobs))
            else:
                _, normalized_action = parsed_action
                train_segments.append((acting_prompt, _tokenize(hf_tokenizer, normalized_action), None))
            runtime._apply_action_output(active, raw_action, acting_prompt)

            if active.result is not None:
                break

            summary_request = runtime._build_summary_prompt_for_active(active)
            if summary_request is not None:
                summary_prompt, retired_count = summary_request
                turn_record_count_before = len(active.turn_records)
                raw_summary, summary_token_ids, summary_logprobs = await generate_for_prompt(summary_prompt)
                runtime._apply_summary_output(active, summary_prompt, retired_count, raw_summary)
                if len(active.turn_records) > turn_record_count_before:
                    train_segments.append((summary_prompt, summary_token_ids, summary_logprobs))

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
            result.turn_rewards = apply_terminal_reward(
                outcome=outcome,
                trainable_turn_ids=trainable_turn_ids_from_records(result.turn_records),
            )
            reward = 1.0 if outcome == "correct_answer" else -1.0
            score = 1.0 if outcome == "correct_answer" else 0.0
            extra_logs.update(
                {
                    "judge_parse_error": judge_decision.parse_error,
                    "judge_outcome": outcome,
                }
            )

        for segment_prompt, segment_tokens, segment_logprobs in train_segments:
            prompt_tokens = _tokenize(hf_tokenizer, segment_prompt)
            start = len(train_tokens) + len(prompt_tokens)
            train_tokens.extend(prompt_tokens)
            train_tokens.extend(segment_tokens)
            action_ranges.append((start, start + len(segment_tokens)))
            if rollout_log_probs is not None:
                rollout_log_probs.extend([0.0] * len(prompt_tokens))
                rollout_log_probs.extend(segment_logprobs or [0.0] * len(segment_tokens))

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
