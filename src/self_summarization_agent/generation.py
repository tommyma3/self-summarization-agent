from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import os
from typing import Any, Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from self_summarization_agent.config import JudgeConfig, ModelConfig


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...


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
        self.model = AutoModelForCausalLM.from_pretrained(
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


@dataclass(slots=True)
class VLLMGenerator:
    model_path: str
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    tensor_parallel_size: int = 1
    attention_backend: str | None = None
    trust_remote_code: bool = False
    enable_thinking: bool = False
    tokenizer: Any = field(init=False)
    llm: Any = field(init=False)
    _sampling_params_cls: Any = field(init=False)

    def __post_init__(self) -> None:
        if self.attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = self.attention_backend
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("vLLM is not installed. Install it in the remote environment to use backend='vllm'.") from exc
        self._sampling_params_cls = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        llm_kwargs = {
            "model": self.model_path,
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        supported_kwargs = set(inspect.signature(LLM).parameters)
        self.llm = LLM(**{key: value for key, value in llm_kwargs.items() if key in supported_kwargs})

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
        sampling_kwargs = {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature if self.do_sample else 0.0,
        }
        if self.do_sample:
            sampling_kwargs["top_p"] = self.top_p
        outputs = self.llm.generate(
            [self._format_prompt(prompt)],
            self._sampling_params_cls(**sampling_kwargs),
        )
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text or ""


def build_generator(model_config: ModelConfig, *, judge_config: JudgeConfig | None = None) -> TextGenerator:
    max_new_tokens = judge_config.max_new_tokens if judge_config else model_config.max_new_tokens
    temperature = judge_config.temperature if judge_config else model_config.temperature
    top_p = judge_config.top_p if judge_config else model_config.top_p
    do_sample = judge_config.do_sample if judge_config else model_config.do_sample
    model_path = model_config.judge_model_path if judge_config and model_config.judge_model_path else model_config.model_path
    backend_name = model_config.backend.lower()
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
    if backend_name == "vllm":
        return VLLMGenerator(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            tensor_parallel_size=model_config.tensor_parallel_size,
            attention_backend=model_config.attention_backend,
            trust_remote_code=model_config.trust_remote_code,
            enable_thinking=model_config.enable_thinking,
        )
    raise ValueError(f"Unsupported model backend: {model_config.backend}")
