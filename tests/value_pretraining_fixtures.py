"""Small exact-token rollouts and a real CPU critic for pretraining tests."""
from dataclasses import replace
import json
import re
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer as FastTokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast

from self_summarization_agent.collection_contract import collection_profile_id
from self_summarization_agent.config import ValuePretrainingConfig, load_train_config, resolved_rollout_sampling_profile, sampling_profile_id
from self_summarization_agent.generation import GenerationResult
from self_summarization_agent.launcher_utils import serialize_runtime_result
from self_summarization_agent.token_stream import TITO_CONTRACT
from self_summarization_agent.trajectory import build_rollout_native_training_cache
from test_tito import Tokenizer, TokenModel, runtime, SEARCH, SUMMARY, FINISH


class CompactTokenizer(Tokenizer):
    def __init__(self, vocab):
        self.vocab, self.template_calls = vocab, 0

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(piece, len(self.vocab)) for piece in
                re.findall(r"<\|im_start\|>|<\|im_end\|>|\w+|\s+|[^\w\s]", text)]

    def decode(self, ids, skip_special_tokens=False):
        inverse = {index: piece for piece, index in self.vocab.items()}
        return "".join(inverse[index] for index in ids)

    def get_vocab(self):
        return dict(self.vocab)


class CompactModel(TokenModel):
    def __init__(self, outputs, vocab):
        super().__init__(outputs)
        self.tokenizer = CompactTokenizer(vocab)

    def generate_token_batch(self, requests):
        results = []
        for request in requests:
            text = next(self.outputs)
            ids = self.tokenizer.encode(text)
            result = GenerationResult(text, list(request.prompt_token_ids), ids,
                                      token_logprobs=[-.25] * len(ids), token_logprobs_mode="raw_logprobs",
                                      finish_reason="stop")
            self.requests.append(request)
            self.results.append(result)
            results.append(result)
        return results


def make_batch(tmp_path, *, task_count=4, samples_per_task=4):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = load_train_config("configs/train/compact_separate_gae_ppo.yaml")
    source = tmp_path / "base"
    allowed = {f"q{i}": f"question {i}" for i in range(task_count)}
    vocab = {"<unk>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<pad>": 3}
    rows, longest = [], 1
    for i, (query_id, query) in enumerate(allowed.items()):
        for index in range(samples_per_task):
            compactions = i % 3
            outputs = [SEARCH, SUMMARY] * compactions + [FINISH]
            if i == 3 and index == 1:
                outputs, compactions = [SEARCH, "</think>bad summary<|im_end|>"], 1
            result = runtime(CompactModel(outputs, vocab), context_threshold_tokens=1 if compactions else 15000).run(query_id, query)
            reward = 1. if index == 0 else -1.
            for record in result.trajectory_records:
                record["training_cache"] = build_rollout_native_training_cache(record["collection_tokens"])
                longest = max(longest, len(record["collection_tokens"]["full_token_ids"]))
            row = serialize_runtime_result(result, query_text=query, judge={
                "outcome": "correct_answer" if reward > 0 else "wrong_answer", "parse_error": False})
            row.update(rollout_index=index, rollout_split="train", rollout_samples_per_task=samples_per_task,
                       policy_checkpoint_id=source.name, policy_checkpoint_path=str(source.resolve()),
                       collection_contract=TITO_CONTRACT, trainable_sample_count=len(result.trajectory_records),
                       turn_rewards={record["turn_id"]: reward for record in result.trajectory_records})
            rows.append(row)
    GPT2Model(GPT2Config(vocab_size=len(vocab), n_positions=longest + 16, n_embd=8, n_layer=1, n_head=2,
                        resid_pdrop=.1, embd_pdrop=.1, attn_pdrop=.1)).save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=FastTokenizer(WordLevel(vocab, unk_token="<unk>")),
                                       unk_token="<unk>", pad_token="<pad>", eos_token="<|im_end|>")
    tokenizer.save_pretrained(source)
    config.model = replace(config.model, model_path=str(source), dtype="float32")
    config.training.value = replace(config.training.value, model_path=str(source), rollout_minibatch_size=4,
                                    prefix_microbatch_size=2, eval_prefix_microbatch_size=2,
                                    activation_checkpointing=False)
    config.training.value_pretraining = ValuePretrainingConfig(
        enabled=True, task_count=task_count, samples_per_task=samples_per_task, validation_task_count=1,
        max_epochs=3, patience=5, learning_rate=.01, head_learning_rate=.02)
    profile = resolved_rollout_sampling_profile(config, split="train")
    for row in rows:
        row.update(collection_profile_id=collection_profile_id(config, source.resolve()),
                   sampling_profile=profile, sampling_profile_id=sampling_profile_id(profile))
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return SimpleNamespace(config=config, actor=source, path=path, allowed=allowed, rows=rows)
