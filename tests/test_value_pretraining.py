from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from self_summarization_agent import value_pretraining as module
from self_summarization_agent.config import ValuePretrainingConfig, config_to_dict, load_train_config, resolved_rollout_sampling_profile
from self_summarization_agent.separate_value import CRITIC_STATE, digest, source_critic_identity
from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
from self_summarization_agent.value_diagnostics import value_diagnostics
from value_pretraining_fixtures import make_batch


@pytest.fixture
def batch(tmp_path):
    return make_batch(tmp_path)


@pytest.mark.parametrize('changes', [
    {'enabled': 1}, {'task_count': True}, {'task_count': 10}, {'validation_task_count': 0},
    {'samples_per_task': 0}, {'seed': -1}, {'max_epochs': 0}, {'patience': False},
    {'min_delta': float('nan')}, {'learning_rate': 0}, {'head_learning_rate': True},
    {'reuse_rollouts_path': ''},
])
def test_pretraining_config_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        ValuePretrainingConfig(**changes)


def test_config_round_trip_and_legacy_value_fingerprint(tmp_path):
    import yaml
    config = load_train_config('configs/train/compact_separate_gae_ppo.yaml')
    old_value = digest(asdict(config.training.value))
    assert not config.training.value_pretraining.enabled
    config.training.value_pretraining = ValuePretrainingConfig(enabled=True)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config_to_dict(config)))
    loaded = load_train_config(path)
    assert loaded.training.value_pretraining == config.training.value_pretraining
    assert digest(asdict(loaded.training.value)) == old_value
    before = resolved_rollout_sampling_profile(loaded, split='eval')
    loaded.rollout = replace(loaded.rollout, sampling_seed=7)
    assert resolved_rollout_sampling_profile(loaded, split='eval') == before
    assert resolved_rollout_sampling_profile(loaded, split='train')['engine_seed'] == 7
    assert resolved_rollout_sampling_profile(loaded, split='train')['extra_sampling_params'] == {}
    with pytest.raises(ValueError):
        loaded.training = replace(loaded.training, value_pretraining=ValuePretrainingConfig(enabled=True),
                                  value=replace(loaded.training.value, architecture='shared', enabled=False))


def test_split_exact_prefixes_and_source_immutability(batch):
    original = batch.path.read_bytes()
    train, validation, manifest = module.prepare_dataset(batch.config, batch.actor, batch.path, batch.allowed)
    assert len({s.rollout_id for s in train}) == 12
    assert len({s.rollout_id for s in validation}) == 4
    assert {s.query_id for s in train}.isdisjoint(s.query_id for s in validation)
    assert {s.state_kind for s in [*train, *validation]} == {'initial', 'post_compaction'}
    for sample in [*train, *validation]:
        assert sample.input_ids == sample.collection_full_token_ids[:-1]
        assert sample.completion_mask == sample.collection_assistant_token_mask[1:]
    assert manifest == module.prepare_dataset(batch.config, batch.actor, batch.path, batch.allowed)[2]
    assert batch.path.read_bytes() == original


def test_default_budget_is_200_attempts_with_160_40_split(tmp_path):
    data = make_batch(tmp_path, task_count=50)
    data.config.training.value_pretraining.validation_task_count = 10
    _, _, manifest = module.prepare_dataset(data.config, data.actor, data.path, data.allowed)
    assert manifest['requested_rollout_count'] == 200
    assert manifest['train_rollout_count'] == 160
    assert manifest['validation_rollout_count'] == 40


@pytest.mark.parametrize('corruption', ['count', 'duplicate', 'actor', 'sampling', 'reward', 'cache', 'query'])
def test_invalid_reused_data_fails_closed(batch, corruption):
    rows = deepcopy(batch.rows)
    if corruption == 'count':
        rows.pop()
    elif corruption == 'duplicate':
        rows[1] = deepcopy(rows[0])
    elif corruption == 'actor':
        rows[0]['policy_checkpoint_path'] = '/another-actor'
    elif corruption == 'sampling':
        rows[0]['sampling_profile_id'] = 'wrong'
    elif corruption == 'reward':
        rows[0]['turn_rewards'] = {key: True for key in rows[0]['turn_rewards']}
    elif corruption == 'cache':
        rows[0]['trajectory_records'][0]['training_cache']['input_ids'][0] += 1
    else:
        rows[0]['query_id'] = 'evaluation-question'
    batch.path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError):
        module.prepare_dataset(batch.config, batch.actor, batch.path, batch.allowed)


def test_real_mc_fit_best_reload_and_fresh_online_optimizer(batch, tmp_path):
    actor_digest, data_digest = source_critic_identity(batch.actor, resume=False), module.file_digest(batch.path)
    captured = []
    def factory(*args, **kwargs):
        critic = SeparateValueTrainer(*args, **kwargs)
        original = critic.fit
        def fit(samples, **options):
            assert all(p.requires_grad for p in [*critic.model.parameters(), *critic.head.parameters()])
            assert options['scalar_targets'] == {module.sample_key(s): s.reward for s in samples}
            captured.append(critic.update_count)
            return original(samples, **options)
        critic.fit = fit
        return critic
    output = tmp_path / 'warmup'
    best = module.fit_pretraining(batch.config, batch.actor, batch.path, output, batch.allowed,
                                 device='cpu', trainer_factory=factory)
    manifest = json.loads((output / module.MANIFEST).read_text())
    assert captured == [0, 3, 6]
    assert manifest['optimizer_steps'] == 9
    assert manifest['target_kind'] == 'terminal_mc'
    assert module.completed_pretraining(batch.config, batch.actor, output) == best
    before = load_file(str(batch.actor / 'model.safetensors'))
    after = load_file(str(output / 'last' / 'model.safetensors'))
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert source_critic_identity(batch.actor, resume=False) == actor_digest
    assert module.file_digest(batch.path) == data_digest
    online = SeparateValueTrainer(best, batch.actor, batch.config.training.value, dtype='float32', device='cpu',
                                  resume=False, activation_checkpointing=False, advantage_estimator='compaction_gae_value')
    assert online.update_count == 0 and not online.optimizer.state
    assert online.optimizer.param_groups[1]['lr'] == batch.config.training.value.head_learning_rate
    _, validation, _ = module.prepare_dataset(batch.config, batch.actor, batch.path, batch.allowed)
    values, losses = online.evaluate(validation)
    score = value_diagnostics(validation, values, losses, prefix='validation')['validation/brier_score']
    assert score == pytest.approx(manifest['best_validation_brier'], abs=1e-7)
    assert len(list((output / 'epochs').iterdir())) <= 2
    (output / 'best' / 'compaction_value' / 'compaction_value_head.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed'):
        module.completed_pretraining(batch.config, batch.actor, output)


def test_interrupted_epoch_resumes_exact_weights_and_optimizer(batch, tmp_path, monkeypatch):
    uninterrupted, interrupted = tmp_path / 'uninterrupted', tmp_path / 'interrupted'
    module.fit_pretraining(batch.config, batch.actor, batch.path, uninterrupted, batch.allowed, device='cpu')
    original = module.atomic_json
    failed = []
    def atomic_json(path, body):
        if Path(path).name == module.STATE and body['epoch'] == 2 and not failed:
            failed.append(True)
            raise RuntimeError('injected state commit failure')
        return original(path, body)
    monkeypatch.setattr(module, 'atomic_json', atomic_json)
    with pytest.raises(RuntimeError, match='injected'):
        module.fit_pretraining(batch.config, batch.actor, batch.path, interrupted, batch.allowed, device='cpu')
    assert not (interrupted / '.complete').exists()
    assert json.loads((interrupted / module.STATE).read_text())['epoch'] == 1
    with pytest.raises(ValueError, match='--resume'):
        module.fit_pretraining(batch.config, batch.actor, batch.path, interrupted, batch.allowed, device='cpu')
    module.fit_pretraining(batch.config, batch.actor, batch.path, interrupted, batch.allowed, resume=True, device='cpu')
    for filename in ['model.safetensors', 'compaction_value/compaction_value_head.safetensors']:
        a, b = load_file(str(uninterrupted / 'last' / filename)), load_file(str(interrupted / 'last' / filename))
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    a = torch.load(uninterrupted / 'last' / CRITIC_STATE, weights_only=True)
    b = torch.load(interrupted / 'last' / CRITIC_STATE, weights_only=True)
    assert a['update_count'] == b['update_count'] == 9
    for key, values in a['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, b['optimizer']['state'][key][name], rtol=0, atol=0)


@pytest.mark.parametrize('backend,seed', [('vllm_offline', True), ('vllm_offline', -1),
    ('vllm_offline', 2**32), ('vllm_offline', 1.5), ('transformers', 1), ('api', 1)])
def test_rollout_engine_seed_validation(backend, seed):
    from self_summarization_agent.config import RolloutConfig
    with pytest.raises(ValueError, match='sampling_seed'):
        RolloutConfig(backend=backend, sampling_seed=seed)
