from pathlib import Path

import pytest
import torch

from self_summarization_agent.value_model import (
    VALUE_HEAD_FILENAME,
    VALUE_HEAD_MANIFEST_FILENAME,
    CompactionValueHead,
    expected_binary_value,
    load_value_head,
    migrate_compaction_value_head_sidecar,
    reward_class_indices,
    rollout_normalized_value_loss,
    rollout_value_weights,
    save_value_head,
    state_hidden_states,
)


def test_zero_initialized_head_predicts_zero_expected_reward() -> None:
    head = CompactionValueHead(3, zero_initialize=True)

    logits = head(torch.randn(4, 3))

    assert torch.equal(logits, torch.zeros_like(logits))
    assert torch.equal(expected_binary_value(logits), torch.zeros(4))


def test_rollout_value_weights_give_each_rollout_equal_total_weight() -> None:
    weights = rollout_value_weights(["short", "long", "long", "long"])

    assert weights == pytest.approx([0.5, 1 / 6, 1 / 6, 1 / 6])
    assert weights[0] == pytest.approx(sum(weights[1:]))


def test_rollout_normalized_value_loss_uses_binary_targets() -> None:
    logits = torch.zeros((4, 2))
    rewards = torch.tensor([-1.0, 1.0, 1.0, 1.0])
    weights = torch.tensor(rollout_value_weights(["short", "long", "long", "long"]))

    loss = rollout_normalized_value_loss(logits, rewards, weights)

    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert reward_class_indices(rewards).tolist() == [0, 1, 1, 1]


def test_state_hidden_states_selects_exact_prompt_end() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    selected = state_hidden_states(hidden, torch.tensor([2, 4]))

    assert torch.equal(selected[0], hidden[0, 1])
    assert torch.equal(selected[1], hidden[1, 3])


def test_value_head_checkpoint_round_trip(tmp_path: Path) -> None:
    head = CompactionValueHead(3, zero_initialize=False)
    with torch.no_grad():
        head.projection.weight.fill_(0.25)
        head.projection.bias.copy_(torch.tensor([-0.5, 0.5]))

    save_value_head(head, tmp_path)
    loaded, found = load_value_head(tmp_path, hidden_size=3, zero_initialize=True)

    assert found is True
    for expected, actual in zip(head.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)


def test_value_head_sidecar_lives_in_vllm_ignored_subdirectory(tmp_path: Path) -> None:
    head = CompactionValueHead(3, zero_initialize=False)
    save_value_head(head, tmp_path)

    assert (tmp_path / VALUE_HEAD_FILENAME).exists()
    assert (tmp_path / VALUE_HEAD_MANIFEST_FILENAME).exists()
    # Legacy top-level files must not be created.
    assert not (tmp_path / "compaction_value_head.safetensors").exists()
    assert not (tmp_path / "compaction_value_config.json").exists()


def test_legacy_top_level_sidecar_is_migrated_and_loadable(tmp_path: Path) -> None:
    head = CompactionValueHead(3, zero_initialize=False)
    with torch.no_grad():
        head.projection.weight.fill_(0.75)
        head.projection.bias.copy_(torch.tensor([0.25, -0.25]))
    save_value_head(head, tmp_path)

    # Simulate an old checkpoint by moving the sidecar back to the top level.
    old_weights = tmp_path / "compaction_value_head.safetensors"
    old_manifest = tmp_path / "compaction_value_config.json"
    (tmp_path / VALUE_HEAD_FILENAME).rename(old_weights)
    (tmp_path / VALUE_HEAD_MANIFEST_FILENAME).rename(old_manifest)

    assert migrate_compaction_value_head_sidecar(tmp_path) is True

    loaded, found = load_value_head(tmp_path, hidden_size=3, zero_initialize=True)
    assert found is True
    for expected, actual in zip(head.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
    assert not old_weights.exists()
    assert not old_manifest.exists()
