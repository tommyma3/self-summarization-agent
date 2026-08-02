from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import inspect
import json
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
from self_summarization_agent.chat_template import configure_tokenizer_chat_template
from self_summarization_agent.models import Message, ToolCall
from self_summarization_agent.prompts import ConversationPrompt, serialize_messages


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...

    def count_prompt_tokens(self, prompt: str) -> int:
        ...


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    cumulative_logprob: float | None = None
    token_logprobs: list[float] | None = None
    message: Message | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


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
    sampling_extra: dict[str, Any] = field(default_factory=dict)
    dtype: str = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = False
    enable_thinking: bool = False
    chat_template_path: str | None = None
    tokenizer: Any = field(init=False)
    model: Any = field(init=False)

    def __post_init__(self) -> None:
        torch_dtype = _resolve_torch_dtype(self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        configure_tokenizer_chat_template(self.tokenizer, self.chat_template_path)
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

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(self._format_prompt(prompt), add_special_tokens=False))

    def _format_prompt(self, prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        if isinstance(prompt, ConversationPrompt):
            messages = serialize_messages(prompt.messages)
        else:
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
            generation_kwargs.update(self.sampling_extra)
        with torch.no_grad():
            output_ids = self.model.generate(
                **encoded,
                **generation_kwargs,
            )
        generated_ids = output_ids[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]


def _object_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if model_dump is not None:
        dumped = model_dump()
        if isinstance(dumped, dict):
            model_extra = getattr(value, "model_extra", None)
            if isinstance(model_extra, dict):
                dumped.update(model_extra)
            return dumped
    raise ValueError(f"Expected an object-like API response, got {type(value).__name__}")


def _required_int_list(value: object, *, field_name: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise RuntimeError(
            f"OpenAI-compatible collection response is missing exact {field_name}. "
            "For vLLM v0.19.1, enable the request extension return_token_ids=true."
        )
    return [int(item) for item in value]


def _parse_api_message(payload: dict[str, Any]) -> Message:
    content = payload.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ValueError("OpenAI-compatible assistant content must be text or null")
    reasoning = payload.get("reasoning_content", payload.get("reasoning"))
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("OpenAI-compatible assistant reasoning must be text or null")
    tool_calls: list[ToolCall] = []
    raw_tool_calls = payload.get("tool_calls") or []
    if not isinstance(raw_tool_calls, list):
        raise ValueError("OpenAI-compatible assistant tool_calls must be a list")
    for index, raw_tool_call in enumerate(raw_tool_calls):
        tool_call_payload = _object_payload(raw_tool_call)
        function_payload = _object_payload(tool_call_payload.get("function"))
        call_id = tool_call_payload.get("id")
        name = function_payload.get("name")
        raw_arguments = function_payload.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(raw_arguments, str):
            raise ValueError(f"OpenAI-compatible tool_calls[{index}] is malformed")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenAI-compatible tool_calls[{index}] has invalid JSON arguments") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"OpenAI-compatible tool_calls[{index}] arguments must decode to an object")
        tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return Message(
        role="assistant",
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )


@dataclass(slots=True)
class OpenAICompatibleGenerator:
    """Chat Completions client for a Qwen-compatible vLLM server.

    Exact server-side prompt and completion token IDs are mandatory by default.
    The runtime persists those IDs as the authoritative GRPO training sequence.
    """

    model_path: str
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    api_base_url: str
    api_model: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_timeout_seconds: float = 600.0
    api_max_retries: int = 2
    api_max_concurrency: int = 32
    api_extra_body: dict[str, Any] = field(default_factory=dict)
    require_exact_token_ids: bool = True
    trust_remote_code: bool = False
    enable_thinking: bool = True
    chat_template_path: str | None = None
    tokenizer: Any = field(init=False)
    client: Any = field(init=False)
    chat_template: str | None = field(init=False, default=None)
    supports_native_tools: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        if not self.api_base_url:
            raise ValueError("model.api_base_url is required for backend='openai_compatible'")
        if self.api_max_concurrency < 1:
            raise ValueError("model.api_max_concurrency must be at least 1")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The openai package is required for backend='openai_compatible'."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.chat_template = configure_tokenizer_chat_template(self.tokenizer, self.chat_template_path)
        self.client = OpenAI(
            api_key=os.environ.get(self.api_key_env) or "EMPTY",
            base_url=self.api_base_url,
            timeout=self.api_timeout_seconds,
            max_retries=self.api_max_retries,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _format_prompt(self, prompt: str) -> str:
        if not isinstance(prompt, ConversationPrompt):
            messages = [{"role": "user", "content": prompt}]
            tools: list[dict[str, Any]] = []
        else:
            messages = serialize_messages(prompt.messages)
            tools = list(prompt.tools)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": self.enable_thinking,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(self._format_prompt(prompt), add_special_tokens=False))

    def _request_payload(self, prompt: str) -> dict[str, Any]:
        if not isinstance(prompt, ConversationPrompt):
            messages = [{"role": "user", "content": prompt}]
            tools: list[dict[str, Any]] = []
            tool_choice: str | dict[str, Any] | None = None
            parallel_tool_calls = False
        else:
            messages = serialize_messages(prompt.messages, for_api=True)
            tools = list(prompt.tools)
            tool_choice = prompt.tool_choice
            parallel_tool_calls = prompt.parallel_tool_calls
        payload: dict[str, Any] = {
            "model": self.api_model or self.model_path,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 0.0,
        }
        if self.do_sample:
            payload["top_p"] = self.top_p
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
            payload["parallel_tool_calls"] = parallel_tool_calls
        elif tool_choice == "none":
            payload["tool_choice"] = "none"
        extra_body = dict(self.api_extra_body)
        template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        template_kwargs["enable_thinking"] = self.enable_thinking
        extra_body["chat_template_kwargs"] = template_kwargs
        chat_template = getattr(self, "chat_template", None)
        if chat_template is not None:
            extra_body["chat_template"] = chat_template
        if self.require_exact_token_ids:
            extra_body["return_token_ids"] = True
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    def _generate_one(self, prompt: str) -> GenerationResult:
        response = self.client.chat.completions.create(**self._request_payload(prompt))
        response_payload = _object_payload(response)
        raw_choices = response_payload.get("choices")
        if not isinstance(raw_choices, list) or len(raw_choices) != 1:
            raise RuntimeError("OpenAI-compatible collection requires exactly one response choice")
        choice_payload = _object_payload(raw_choices[0])
        message_payload = _object_payload(choice_payload.get("message"))
        message = _parse_api_message(message_payload)

        prompt_token_ids_value = response_payload.get("prompt_token_ids")
        completion_token_ids_value = choice_payload.get("token_ids")
        if self.require_exact_token_ids:
            prompt_token_ids = _required_int_list(prompt_token_ids_value, field_name="prompt_token_ids")
            completion_token_ids = _required_int_list(
                completion_token_ids_value,
                field_name="completion token_ids",
            )
        else:
            prompt_token_ids = (
                [int(item) for item in prompt_token_ids_value]
                if isinstance(prompt_token_ids_value, list)
                else None
            )
            completion_token_ids = (
                [int(item) for item in completion_token_ids_value]
                if isinstance(completion_token_ids_value, list)
                else None
            )
        if completion_token_ids is not None:
            # Keep the full sampled representation, including Qwen control tokens.
            # Structured fields drive runtime behavior; this text is the immutable
            # collection artifact and diagnostic view of the exact returned IDs.
            raw_text = self.tokenizer.decode(completion_token_ids, skip_special_tokens=False)
        else:
            parts = []
            if message.reasoning_content:
                parts.append(f"<think>{message.reasoning_content}</think>")
            if message.content:
                parts.append(message.content)
            parts.extend(
                json.dumps(
                    {"tool_name": call.name, "arguments": call.arguments},
                    ensure_ascii=False,
                )
                for call in message.tool_calls
            )
            raw_text = "\n".join(parts)
        usage = response_payload.get("usage")
        return GenerationResult(
            text=raw_text,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            message=message,
            finish_reason=choice_payload.get("finish_reason"),
            usage=dict(usage) if isinstance(usage, dict) else None,
        )

    def generate(self, prompt: str) -> str:
        return self._generate_one(prompt).text

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [result.text for result in self.generate_batch_with_metadata(prompts)]

    def generate_batch_with_metadata(self, prompts: list[str]) -> list[GenerationResult]:
        if not prompts:
            return []
        worker_count = min(len(prompts), self.api_max_concurrency)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(self._generate_one, prompts))


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
    sampling_extra: dict[str, Any] = field(default_factory=dict)
    tensor_parallel_size: int = 1
    attention_backend: str | None = None
    max_model_len: int | None = None
    trust_remote_code: bool = False
    enable_thinking: bool = False
    chat_template_path: str | None = None
    language_model_only: bool = False
    enable_prefix_caching: bool = False
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
        configure_tokenizer_chat_template(self.tokenizer, self.chat_template_path)
        llm_kwargs = {
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "language_model_only": self.language_model_only,
            "enable_prefix_caching": self.enable_prefix_caching,
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

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(self._format_prompt(prompt), add_special_tokens=False))

    def _format_prompt(self, prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        if isinstance(prompt, ConversationPrompt):
            messages = []
            for message in prompt.messages:
                item: dict[str, Any] = {"role": message.role, "content": message.content}
                if message.reasoning_content is not None:
                    item["reasoning_content"] = message.reasoning_content
                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                        for tool_call in message.tool_calls
                    ]
                if message.tool_call_id is not None:
                    item["tool_call_id"] = message.tool_call_id
                messages.append(item)
        else:
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
            sampling_kwargs.update(self.sampling_extra)
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


@dataclass(slots=True)
class SGLangGenerator:
    model_path: str
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    sampling_extra: dict[str, Any] = field(default_factory=dict)
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    attention_backend: str | None = None
    max_model_len: int | None = None
    trust_remote_code: bool = False
    enable_thinking: bool = False
    chat_template_path: str | None = None
    tokenizer: Any = field(init=False)
    engine: Any = field(init=False)

    def __post_init__(self) -> None:
        try:
            import sglang as sgl
        except ImportError as exc:
            raise ImportError(
                "SGLang is not installed. Install it in the remote environment to use backend='sglang'."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        configure_tokenizer_chat_template(self.tokenizer, self.chat_template_path)
        engine_kwargs = {
            "model_path": self.model_path,
            "dtype": self.dtype,
            "tp_size": self.tensor_parallel_size,
            "attention_backend": self.attention_backend,
            "context_length": self.max_model_len,
            "trust_remote_code": self.trust_remote_code,
        }
        self.engine = sgl.Engine(**{key: value for key, value in engine_kwargs.items() if value is not None})

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(self._format_prompt(prompt), add_special_tokens=False))

    def _format_prompt(self, prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        if isinstance(prompt, ConversationPrompt):
            messages = []
            for message in prompt.messages:
                item: dict[str, Any] = {"role": message.role, "content": message.content}
                if message.reasoning_content is not None:
                    item["reasoning_content"] = message.reasoning_content
                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                        for tool_call in message.tool_calls
                    ]
                if message.tool_call_id is not None:
                    item["tool_call_id"] = message.tool_call_id
                messages.append(item)
        else:
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

    def _sampling_params(self) -> dict[str, Any]:
        sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 0.0,
        }
        if self.do_sample:
            sampling_params["top_p"] = self.top_p
            sampling_params.update(self.sampling_extra)
        return sampling_params

    def _generate_outputs(self, prompts: list[str], *, return_logprob: bool = False) -> list[dict[str, Any]]:
        if not prompts:
            return []
        outputs = self.engine.generate(
            prompts,
            self._sampling_params(),
            return_logprob=return_logprob,
        )
        return outputs if isinstance(outputs, list) else [outputs]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
        return [str(output.get("text") or "") for output in self._generate_outputs(formatted_prompts)]

    def generate_batch_with_metadata(self, prompts: list[str]) -> list[GenerationResult]:
        formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
        outputs = self._generate_outputs(formatted_prompts, return_logprob=True)
        completions: list[GenerationResult] = []
        for prompt, output in zip(formatted_prompts, outputs):
            completion_token_ids, token_logprobs = _extract_sglang_completion_logprobs(output)
            completions.append(
                GenerationResult(
                    text=str(output.get("text") or ""),
                    prompt_token_ids=list(self.tokenizer.encode(prompt, add_special_tokens=False)),
                    completion_token_ids=completion_token_ids,
                    cumulative_logprob=sum(token_logprobs) if token_logprobs is not None else None,
                    token_logprobs=token_logprobs,
                )
            )
        return completions


def _extract_sglang_completion_logprobs(output: Any) -> tuple[list[int] | None, list[float] | None]:
    if not isinstance(output, dict):
        return None, None
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, dict):
        return None, None
    raw_logprobs = meta_info.get("output_token_logprobs")
    if not isinstance(raw_logprobs, list):
        return None, None
    token_ids: list[int] = []
    logprobs: list[float] = []
    for item in raw_logprobs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None, None
        logprob, token_id = item[0], item[1]
        if (
            not isinstance(logprob, (int, float))
            or isinstance(logprob, bool)
            or not isinstance(token_id, int)
            or isinstance(token_id, bool)
        ):
            return None, None
        logprobs.append(float(logprob))
        token_ids.append(token_id)
    return token_ids, logprobs


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


def build_generator(
    model_config: ModelConfig,
    *,
    judge_config: JudgeConfig | None = None,
    sampling_extra: dict[str, Any] | None = None,
) -> TextGenerator:
    sampling_extra = dict(sampling_extra or {})
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
            sampling_extra=sampling_extra,
            dtype=model_config.dtype,
            device_map=model_config.device_map,
            trust_remote_code=model_config.trust_remote_code,
            enable_thinking=model_config.enable_thinking,
            chat_template_path=model_config.chat_template_path,
        )
    if backend_name in {"vllm", "vllm_offline"}:
        return VLLMGenerator(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            sampling_extra=sampling_extra,
            tensor_parallel_size=tensor_parallel_size,
            attention_backend=attention_backend,
            max_model_len=max_model_len,
            trust_remote_code=model_config.trust_remote_code,
            enable_thinking=model_config.enable_thinking,
            chat_template_path=model_config.chat_template_path,
            language_model_only=model_config.language_model_only,
            enable_prefix_caching=model_config.enable_prefix_caching,
        )
    if backend_name in {"sglang", "sglang_offline"}:
        return SGLangGenerator(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            sampling_extra=sampling_extra,
            dtype=model_config.dtype,
            tensor_parallel_size=tensor_parallel_size,
            attention_backend=attention_backend,
            max_model_len=max_model_len,
            trust_remote_code=model_config.trust_remote_code,
            enable_thinking=model_config.enable_thinking,
            chat_template_path=model_config.chat_template_path,
        )
    if backend_name in {"openai", "openai_compatible", "openai-compatible"}:
        if not model_config.api_base_url:
            raise ValueError("model.api_base_url is required for backend='openai_compatible'")
        return OpenAICompatibleGenerator(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            api_base_url=model_config.api_base_url,
            api_model=model_config.api_model,
            api_key_env=model_config.api_key_env,
            api_timeout_seconds=model_config.api_timeout_seconds,
            api_max_retries=model_config.api_max_retries,
            api_max_concurrency=model_config.api_max_concurrency,
            api_extra_body={**model_config.api_extra_body, **sampling_extra},
            require_exact_token_ids=model_config.require_exact_token_ids,
            trust_remote_code=model_config.trust_remote_code,
            enable_thinking=model_config.enable_thinking,
            chat_template_path=model_config.chat_template_path,
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
