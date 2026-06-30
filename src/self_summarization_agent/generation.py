from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import os
import sys
from typing import Any, Protocol

import torch
from transformers import AutoModel, AutoTokenizer

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = AutoModel  # type: ignore[misc,assignment]

from self_summarization_agent.config import JudgeConfig, ModelConfig


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    cumulative_logprob: float | None = None
    token_logprobs: list[float] | None = None


def _resolve_torch_dtype(dtype_name: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


@dataclass(slots=True)
class TransformersGenerator:
    model_path: str
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    dtype: str = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = False
    enable_thinking: bool = False
    tokenizer: Any = field(init=False)
    model: Any = field(init=False)

    def __post_init__(self) -> None:
        torch_dtype = _resolve_torch_dtype(self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = _load_transformers_model(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _format_prompt(self, prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, prompt: str) -> str:
        encoded = self.tokenizer(self._format_prompt(prompt), return_tensors="pt")
        encoded = {name: tensor.to(self.model.device) for name, tensor in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                **generation_kwargs,
            )
        generated_ids = output_ids[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]


def _apply_vllm_subprocess_fix() -> None:
    """Work around a vLLM 0.19.1 import-order bug that causes a SIGSEGV
    during model architecture inspection in subprocesses.

    The model-inspection subprocess imports
    ``vllm.model_executor.models.registry`` which triggers native extension
    loading in an order that crashes unless ``vllm.config.vllm`` has been
    loaded first.  We replace the subprocess command so that it pre-imports
    the missing module before entering the registry's ``_run()`` entry point.
    """
    try:
        import vllm.model_executor.models.registry as _vllm_reg
    except ImportError:
        return

    _cmd = getattr(_vllm_reg, "_SUBPROCESS_COMMAND", None)
    if _cmd is None or len(_cmd) < 2:
        return

    _fix_code = (
        "import vllm.config.vllm;"
        "import runpy;"
        "runpy.run_module('vllm.model_executor.models.registry', "
        "run_name='__main__', alter_sys=True)"
    )
    _fixed_cmd = [sys.executable, "-c", _fix_code]
    if _cmd != _fixed_cmd:
        _vllm_reg._SUBPROCESS_COMMAND = _fixed_cmd  # type: ignore[attr-defined]


@dataclass(slots=True)
class VLLMGenerator:
    model_path: str
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    tensor_parallel_size: int = 1
    attention_backend: str | None = None
    max_model_len: int | None = None
    trust_remote_code: bool = False
    enable_thinking: bool = False
    language_model_only: bool = False
    tokenizer: Any = field(init=False)
    llm: Any = field(init=False)
    _sampling_params_cls: Any = field(init=False)

    def __post_init__(self) -> None:
        if self.attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = self.attention_backend
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed. Install it in the remote environment to use backend='vllm' or 'vllm_offline'."
            ) from exc
        _apply_vllm_subprocess_fix()
        self._sampling_params_cls = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        llm_kwargs = {
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "language_model_only": self.language_model_only,
        }
        sig = inspect.signature(LLM)
        supported_kwargs = set(sig.parameters)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        self.llm = LLM(
            **{
                key: value
                for key, value in llm_kwargs.items()
                if (key in supported_kwargs or has_var_keyword) and value is not None
            }
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _format_prompt(self, prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, prompt: str) -> str:
        outputs = self.generate_batch([prompt])
        return outputs[0] if outputs else ""

    def _sampling_kwargs(self, *, include_logprobs: bool = False) -> dict[str, Any]:
        sampling_kwargs = {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 0.0,
        }
        if self.do_sample:
            sampling_kwargs["top_p"] = self.top_p
        if include_logprobs:
            sampling_kwargs["logprobs"] = 1
        return sampling_kwargs

    def generate_batch(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(
            [self._format_prompt(prompt) for prompt in prompts],
            self._sampling_params_cls(**self._sampling_kwargs()),
        )
        completions: list[str] = []
        for output in outputs:
            if not output.outputs:
                completions.append("")
                continue
            completions.append(output.outputs[0].text or "")
        return completions

    def generate_batch_with_metadata(self, prompts: list[str]) -> list[GenerationResult]:
        outputs = self.llm.generate(
            [self._format_prompt(prompt) for prompt in prompts],
            self._sampling_params_cls(**self._sampling_kwargs(include_logprobs=True)),
        )
        completions: list[GenerationResult] = []
        for output in outputs:
            if not output.outputs:
                completions.append(GenerationResult(text=""))
                continue
            completion = output.outputs[0]
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            completion_token_ids = getattr(completion, "token_ids", None)
            cumulative_logprob = getattr(completion, "cumulative_logprob", None)
            token_logprobs = _extract_completion_token_logprobs(completion)
            completions.append(
                GenerationResult(
                    text=completion.text or "",
                    prompt_token_ids=list(prompt_token_ids) if prompt_token_ids is not None else None,
                    completion_token_ids=list(completion_token_ids) if completion_token_ids is not None else None,
                    cumulative_logprob=float(cumulative_logprob)
                    if cumulative_logprob is not None
                    else None,
                    token_logprobs=token_logprobs,
                )
            )
        return completions


def _extract_completion_token_logprobs(completion: Any) -> list[float] | None:
    token_ids = getattr(completion, "token_ids", None)
    raw_logprobs = getattr(completion, "logprobs", None)
    if token_ids is None or raw_logprobs is None:
        return None
    token_ids = list(token_ids)
    raw_logprobs = list(raw_logprobs)
    if len(token_ids) != len(raw_logprobs):
        return None
    values: list[float] = []
    for token_id, candidates in zip(token_ids, raw_logprobs):
        candidate = None
        if isinstance(candidates, dict):
            candidate = candidates.get(token_id)
            if candidate is None:
                candidate = candidates.get(str(token_id))
            if candidate is None and len(candidates) == 1:
                candidate = next(iter(candidates.values()))
        else:
            candidate = candidates
        logprob = getattr(candidate, "logprob", candidate)
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            return None
        values.append(float(logprob))
    return values


def build_generator(model_config: ModelConfig, *, judge_config: JudgeConfig | None = None) -> TextGenerator:
    max_new_tokens = judge_config.max_new_tokens if judge_config else model_config.max_new_tokens
    temperature = judge_config.temperature if judge_config else model_config.temperature
    top_p = judge_config.top_p if judge_config else model_config.top_p
    do_sample = judge_config.do_sample if judge_config else model_config.do_sample
    model_path = (
        judge_config.model_path
        if judge_config and judge_config.model_path
        else model_config.judge_model_path
        if judge_config and model_config.judge_model_path
        else model_config.model_path
    )
    backend_name = (judge_config.backend if judge_config and judge_config.backend else model_config.backend).lower()
    tensor_parallel_size = (
        judge_config.tensor_parallel_size
        if judge_config and judge_config.tensor_parallel_size is not None
        else model_config.tensor_parallel_size
    )
    attention_backend = (
        judge_config.attention_backend
        if judge_config and judge_config.attention_backend is not None
        else model_config.attention_backend
    )
    max_model_len = (
        judge_config.max_model_len
        if judge_config and judge_config.max_model_len is not None
        else model_config.max_model_len
    )
    if backend_name == "transformers":
        return TransformersGenerator(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            dtype=model_config.dtype,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            enable_thinking=model_config.enable_thinking,
        )
    if backend_name in {"vllm", "vllm_offline"}:
        return VLLMGenerator(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            tensor_parallel_size=tensor_parallel_size,
            attention_backend=attention_backend,
            max_model_len=max_model_len,
            trust_remote_code=model_config.trust_remote_code,
            enable_thinking=model_config.enable_thinking,
            language_model_only=model_config.language_model_only,
        )
    raise ValueError(f"Unsupported model backend: {model_config.backend}")


def _load_transformers_model(
    model_path: str,
    *,
    torch_dtype: Any,
    device_map: str,
    trust_remote_code: bool,
) -> Any:
    return AutoModelForMultimodalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
