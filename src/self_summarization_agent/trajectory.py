from dataclasses import dataclass
import math
from collections.abc import Mapping


@dataclass(slots=True)
class RLSample:
    query_id: str
    turn_id: str
    prompt: str
    completion: str
    reward: float
    trainable_kind: str


def extract_trainable_samples(turns: list[Mapping[str, object]], rewards: dict[str, float]) -> list[RLSample]:
    samples: list[RLSample] = []
    seen_turn_ids: set[str] = set()
    allowed_trainable_kinds = {"summary", "final_answer"}
    allowed_non_trainable_kinds = {"tool"}
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise ValueError(f"Turn record must be a mapping, got {type(turn).__name__}")
        required_keys = {"kind", "turn_id"}
        missing_keys = sorted(required_keys - set(turn))
        if missing_keys:
            raise ValueError(f"Turn record is missing required keys: {', '.join(missing_keys)}")

        turn_kind = turn["kind"]
        turn_id = turn["turn_id"]
        if not isinstance(turn_id, str):
            raise ValueError(f"Turn record has non-string turn_id: {turn_id!r}")
        if turn_id in seen_turn_ids:
            raise ValueError(f"Duplicate turn_id found: {turn_id}")
        seen_turn_ids.add(turn_id)
        if turn_kind not in allowed_trainable_kinds and turn_kind not in allowed_non_trainable_kinds:
            raise ValueError(f"Unknown turn kind: {turn_kind}")
        if turn_kind not in allowed_trainable_kinds:
            continue
        for field_name in ("query_id", "prompt", "completion"):
            if field_name not in turn:
                raise ValueError(f"Trainable turn record is missing required key: {field_name}")
        query_id = turn["query_id"]
        prompt = turn["prompt"]
        completion = turn["completion"]
        if not isinstance(query_id, str):
            raise ValueError(f"Trainable turn_id {turn_id} has non-string query_id")
        if not isinstance(prompt, str):
            raise ValueError(f"Trainable turn_id {turn_id} has non-string prompt")
        if not isinstance(completion, str):
            raise ValueError(f"Trainable turn_id {turn_id} has non-string completion")
        if turn_id not in rewards:
            raise ValueError(f"Missing reward for trainable turn_id: {turn_id}")
        reward = rewards[turn_id]
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise ValueError(f"Trainable turn_id {turn_id} has non-numeric reward")
        if not math.isfinite(float(reward)):
            raise ValueError(f"Trainable turn_id {turn_id} has non-finite reward")
        samples.append(
            RLSample(
                query_id=query_id,
                turn_id=turn_id,
                prompt=prompt,
                completion=completion,
                reward=float(reward),
                trainable_kind=turn_kind,
            )
        )
    unknown_reward_ids = sorted(set(rewards) - seen_turn_ids)
    if unknown_reward_ids:
        raise ValueError(f"Reward ids do not match any turn: {', '.join(unknown_reward_ids)}")
    return samples
