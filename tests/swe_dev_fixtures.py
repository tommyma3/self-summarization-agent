from pathlib import Path
import json
import re

import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer as NativeTokenizer
from tokenizers.models import WordLevel

from test_tito import Tokenizer, TokenModel, SUMMARY
from test_terminal_bench import call, COMPLETE
from self_summarization_agent.benchmarks.swe_dev.local_fixture import FIX_COMMAND, WRONG_COMMAND
from self_summarization_agent.benchmarks.swe_dev.task_builder import build_fixture
from self_summarization_agent.config import load_train_config
from self_summarization_agent.generation import GenerationResult


class ChunkTokenizer(Tokenizer):
    """Small deterministic test vocabulary. Chunks decode without normalization."""
    def __init__(self):
        super().__init__()
        self.chunks = {}
    def encode(self, text, **kwargs):
        assert "sampled-only-reason" not in text, "Historical completion re-encoded"
        result = []
        for part in re.split(r"(<\|im_start\|>|<\|im_end\|>)", text):
            if part in self.special:
                result.append(self.special[part])
                continue
            for start in range(0, len(part), 16):
                chunk = part[start:start+16]
                result.append(self.chunks.setdefault(chunk, len(self.chunks) + 100))
        return result
    def decode(self, ids, **kwargs):
        inverse = {v: k for k, v in self.chunks.items()}
        return "".join({1: "<|im_start|>", 2: "<|im_end|>", 3: "sampled-only-reason"}.get(i, inverse.get(i, "")) for i in ids)


class FixturePolicy(TokenModel):
    fixture_only = True
    def __init__(self, outputs, *, raw_logprobs=False):
        super().__init__(outputs)
        self.tokenizer = ChunkTokenizer()
        self.raw_logprobs = raw_logprobs
    def generate_token_batch(self, requests):
        results = []
        for request in requests:
            self.requests.append(request)
            ids = [3] + self.tokenizer.encode(next(self.outputs))
            result = GenerationResult(self.tokenizer.decode(ids), list(request.prompt_token_ids), ids,
                token_logprobs=[-0.25] * len(ids) if self.raw_logprobs else None,
                token_logprobs_mode="raw_logprobs" if self.raw_logprobs else None, finish_reason="stop")
            results.append(result)
            self.results.append(result)
        return results


def action(command):
    return call("execute_commands", dict(analysis="inspect", plan="fix", commands=[dict(keystrokes=command, duration=0)]))


def outputs(compactions=2):
    for attempt in range(4):
        if attempt == 1:
            yield COMPLETE
            if compactions:
                yield SUMMARY
            yield COMPLETE
            continue
        yield action(WRONG_COMMAND if attempt == 2 else FIX_COMMAND)
        if compactions:
            yield SUMMARY
        yield COMPLETE
        if compactions == 2:
            yield SUMMARY
        if compactions != 1:
            yield COMPLETE


def pilot_config(tmp_path, mode="grpo", compactions=2):
    manifest, evaluation = build_fixture("tests/fixtures/swe_dev/trim-name", tmp_path / "pilot")
    cfg = load_train_config(f"configs/train/swe_dev_{mode}.yaml", overrides={
        "benchmark.train_manifest_path": str(manifest), "benchmark.eval_manifest_path": str(evaluation),
        "runtime.context_threshold_tokens": 1 if compactions else 10000,
        "runtime.max_context_tokens": 12000, "rollout.max_model_len": 12000,
        "runtime.max_summary_tokens": 100, "runtime.tool_budget": 1 if compactions == 1 else 100,
        "model.max_new_tokens": 1024, "training.max_sequence_length": 12000,
        "model.dtype": "float32", "model.device_map": "cpu", "training.learning_rate": 0.001,
        "training.value.fitting_epochs": 1, "training.value.prefix_microbatch_size": 4,
        "training.value.activation_checkpointing": False,
        "training.value.evalset_interval": 0, "training.value.diagnostics_interval": 1,
        "training.value.learning_rate": 0.002, "training.activation_checkpointing": False,
    })
    path = tmp_path / "base"
    torch.manual_seed(13)
    GPT2LMHeadModel(GPT2Config(vocab_size=2048, n_positions=12000, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.)).save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=NativeTokenizer(WordLevel(
        {str(i): i for i in range(2048)}, unk_token="0")), unk_token="0", pad_token="0", eos_token="2")
    tokenizer.save_pretrained(path)
    cfg.model.model_path = str(path)
    cfg.training.value.model_path = str(path) if cfg.training.value.separate else None
    return cfg, path
