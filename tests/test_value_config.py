from pathlib import Path

import pytest

from self_summarization_agent.config import (
    CompactionValueConfig, ModelConfig, TrainingConfig, config_to_dict, load_train_config,
)
from self_summarization_agent.verl_ray_trainer import build_verl_fsdp_worker_config


@pytest.mark.parametrize("invalid", [True, False, "1e-4", 0, -1, float("nan"), float("inf"), float("-inf")])
def test_head_lr_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="head_learning_rate"):
        CompactionValueConfig(head_learning_rate=invalid)


@pytest.mark.parametrize("lr", [None, 1e-6, 1e-4])
def test_head_lr_reaches_worker_without_changing_actor_lr(lr):
    config = TrainingConfig(
        learning_rate=1e-6, advantage_estimator="compaction_mc_value",
        value=CompactionValueConfig(enabled=True, head_learning_rate=lr),
    )
    worker = build_verl_fsdp_worker_config(ModelConfig(model_path="unused"), config)
    assert worker["compaction_value"]["head_learning_rate"] == lr
    assert worker["actor"]["optim"]["lr"] == 1e-6
    assert worker["compaction_value"]["loss_coefficient"] == config.value.loss_coefficient


def test_preset_changes_only_identity_and_head_lr():
    root = Path(__file__).resolve().parents[1] / "configs" / "train"
    baseline = config_to_dict(load_train_config(root / "compact_value_mc.yaml"))
    experiment = config_to_dict(load_train_config(root / "compact_value_mc_head_lr.yaml"))
    assert experiment["training"]["value"]["head_learning_rate"] == 1e-4
    assert baseline["training"]["value"]["head_learning_rate"] is None
    assert experiment["experiment"]["name"] != baseline["experiment"]["name"]
    assert experiment["experiment"]["output_root"] != baseline["experiment"]["output_root"]
    for config in (baseline, experiment):
        config["experiment"].pop("name")
        config["experiment"].pop("output_root")
        config["training"]["value"].pop("head_learning_rate")
    assert experiment == baseline
    overridden = load_train_config(root / "compact_value_mc_head_lr.yaml", overrides={
        "training.value.head_learning_rate": 1e-5,
    })
    assert overridden.training.value.head_learning_rate == 1e-5
