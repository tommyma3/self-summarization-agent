from dataclasses import replace
import fcntl
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from self_summarization_agent import iteration_launcher as launcher
from self_summarization_agent import value_pretraining as core
from self_summarization_agent import value_pretraining_step as step
from self_summarization_agent.config import load_train_config, resolved_rollout_sampling_profile
from self_summarization_agent.separate_value import ADVANTAGES, CRITIC_STATE, FROZEN_TARGETS
from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
from value_pretraining_fixtures import make_batch


def test_reuse_skips_collection_and_completed_retry_skips_fitter(tmp_path):
    batch = make_batch(tmp_path)
    batch.config.training.value_pretraining.reuse_rollouts_path = str(batch.path)
    output = tmp_path / 'warmup'
    def fit(*args, **kwargs):
        return core.fit_pretraining(*args, **kwargs, device='cpu')
    best = step.run_value_pretraining(batch.config, checkpoint_path=batch.actor, output_path=output,
        allowed_queries=batch.allowed, command_runner=lambda cmd: pytest.fail('must reuse'), fitter=fit)
    assert step.run_value_pretraining(batch.config, checkpoint_path=batch.actor, output_path=output,
        allowed_queries=batch.allowed, command_runner=lambda cmd: pytest.fail('must skip'),
        fitter=lambda *a, **k: pytest.fail('already complete')) == best


def test_collect_exact_budget_no_eval_and_resume_forwarding(tmp_path):
    batch = make_batch(tmp_path)
    # Default production budget, but inject collection/fitting so no model work occurs.
    batch.config.training.value_pretraining = replace(batch.config.training.value_pretraining,
        task_count=50, validation_task_count=10)
    allowed = {f'q{i}': f'question {i}' for i in range(50)}
    calls = []
    def collect(command):
        local = load_train_config(command[command.index('--config') + 1])
        assert local.collection.train_task_count == 50 and local.training.group_size == 4
        assert (local.dataset.train_limit, local.dataset.eval_limit) == (730, 100)
        assert not any(arg.startswith('--eval-') for arg in command)
        assert command[command.index('--sample-seed') + 1] == '0'
        calls.append(command)
        return 0
    def fit(config, actor, rollouts, output, allowed_queries, **kwargs):
        assert calls and rollouts == output / 'rollouts.cached.jsonl'
        raise RuntimeError('interrupted fitter')
    output = tmp_path / 'warmup'
    for resume in (False, True):
        with pytest.raises(RuntimeError, match='interrupted fitter'):
            step.run_value_pretraining(batch.config, checkpoint_path=batch.actor, output_path=output,
                allowed_queries=allowed, command_runner=collect, fitter=fit, resume=resume)
    assert '--resume' not in calls[0] and '--resume' in calls[1]
    assert batch.config.collection.train_task_count == 50
    assert batch.config.training.group_size == 4
    assert len((output / 'phase_timings.jsonl').read_text().splitlines()) == 2


def test_collection_failure_and_request_mismatch_fail_before_fit(tmp_path):
    batch = make_batch(tmp_path)
    output = tmp_path / 'warmup'
    with pytest.raises(subprocess.CalledProcessError):
        step.run_value_pretraining(batch.config, checkpoint_path=batch.actor, output_path=output,
            allowed_queries=batch.allowed, command_runner=lambda cmd: 9,
            fitter=lambda *a, **k: pytest.fail('collection failed'))
    assert not (output / '.complete').exists()
    batch.config.training.value_pretraining.seed += 1
    with pytest.raises(ValueError, match='inputs changed'):
        step.run_value_pretraining(batch.config, checkpoint_path=batch.actor, output_path=output,
            allowed_queries=batch.allowed, resume=True)


def test_lock_and_output_layout_preserve_initializers(tmp_path):
    batch = make_batch(tmp_path)
    for output in (batch.actor, batch.actor / 'warmup', tmp_path):
        with pytest.raises(ValueError, match='disjoint'):
            step.run_value_pretraining(batch.config, checkpoint_path=batch.actor, output_path=output)
    assert not (batch.actor / '.lock').exists()
    output = tmp_path / 'locked'
    output.mkdir()
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already running'):
            step.run_value_pretraining(batch.config, checkpoint_path=batch.actor, output_path=output)


@pytest.mark.parametrize('iteration', [0, 1])
def test_initialization_phase_once_and_advanced_pair_skips(tmp_path, monkeypatch, iteration):
    batch = make_batch(tmp_path)
    root = tmp_path / 'run'
    output = root / 'value_pretraining'
    core.fit_pretraining(batch.config, batch.actor, batch.path, output, batch.allowed, device='cpu')
    phases = []
    monkeypatch.setattr(launcher, '_run_or_skip_phase', lambda **kw: phases.append(kw))
    options = dict(actor=batch.actor, train_dir=root, config_path='config.yaml', iteration=iteration,
        command_runner=lambda cmd: pytest.fail('already complete'), python_executable='python',
        resume=True, overrides=(), timeout=90)
    assert launcher._initial_pretrained_critic(batch.config, **options) == output / 'best'
    assert phases[0]['phase'] == 'value_pretraining' and phases[0]['completed']
    assert phases[0]['iteration'] == iteration
    assert '--resume' in phases[0]['command']
    assert json.loads((root / 'value_pretraining_initialization.json').read_text())['critic_optimizer'] == 'fresh'
    monkeypatch.setattr('self_summarization_agent.separate_value.validate_resume_checkpoint', lambda path: True)
    assert launcher._initial_pretrained_critic(batch.config, **{**options, 'iteration': 2}) is None
    assert len(phases) == 1


def test_first_actor_commands_use_fresh_sampling_and_selected_critic(tmp_path, monkeypatch):
    batch = make_batch(tmp_path)
    best = tmp_path / 'warmup' / 'best'
    monkeypatch.setattr(launcher, 'resolve_latest_checkpoint', lambda root: SimpleNamespace(path=batch.actor, checkpoint_id=batch.actor.name))
    monkeypatch.setattr(launcher, '_initial_pretrained_critic', lambda *a, **k: best)
    phases = []
    def capture(**kw):
        phases.append(kw)
        if kw['phase'] == 'train_update':
            raise RuntimeError('captured commands')
    monkeypatch.setattr(launcher, '_run_or_skip_phase', capture)
    with pytest.raises(RuntimeError, match='captured commands'):
        launcher.run_training_iteration(batch.config, config_path='original.yaml', iteration=1,
            latest_root=tmp_path / 'run')
    collect, train = [phase['command'] for phase in phases]
    assert 'rollout.sampling_seed=1' in collect and 'rollout.sampling_seed=1' in train
    override = f'training.value.model_path={json.dumps(str(best))}'
    assert override in train and override not in collect
    assert '--eval-raw-output' in collect  # normal eval stays part of the online iteration
    assert batch.config.rollout.sampling_seed is None
    assert batch.config.training.value.model_path == str(batch.actor)


def test_fresh_batch_gae_freezes_pretrained_predictions_before_online_fit(tmp_path, monkeypatch):
    from self_summarization_agent import critic_step
    from self_summarization_agent.train_step import samples_from_rollout_rows
    import torch
    batch = make_batch(tmp_path / 'warmup_source')
    best = core.fit_pretraining(batch.config, batch.actor, batch.path, tmp_path / 'warmup', batch.allowed, device='cpu')
    fresh = make_batch(tmp_path / 'fresh_collection')  # independent runtime generations
    for row in fresh.rows:
        row['policy_checkpoint_id'] = batch.actor.name
        row['policy_checkpoint_path'] = str(batch.actor)
    fresh.path.write_text(''.join(json.dumps(row) + '\n' for row in fresh.rows))
    samples = samples_from_rollout_rows(fresh.rows, expected_checkpoint_id=batch.actor.name, retain_critic_only_states=True)
    batch.config.training.value = replace(batch.config.training.value, model_path=str(best), fitting_epochs=1)
    expected = SeparateValueTrainer(best, batch.actor, batch.config.training.value, dtype='float32',
        device='cpu', resume=False, advantage_estimator='compaction_gae_value')
    values, _ = expected.evaluate(samples)
    output = tmp_path / 'online'
    frozen_before_fit = []
    def factory(*args, **kwargs):
        assert Path(args[0]) == best and kwargs['resume'] is False
        critic = SeparateValueTrainer(*args, **{**kwargs, 'device': 'cpu'})
        assert critic.update_count == 0 and not critic.optimizer.state
        assert critic.optimizer.param_groups[1]['lr'] == batch.config.training.value.head_learning_rate
        original = critic.fit
        def fit(*a, **k):
            frozen_before_fit.append((output / FROZEN_TARGETS).read_bytes())
            return original(*a, **k)
        critic.fit = fit
        return critic
    monkeypatch.setattr('self_summarization_agent.separate_value_trainer.SeparateValueTrainer', factory)
    critic_step.run(batch.config, batch.actor, fresh.path, output, 'iteration-00001')
    frozen = json.loads((output / FROZEN_TARGETS).read_text())
    assert [row['value'] for row in frozen['rows']] == pytest.approx(values, abs=1e-7)
    assert (output / FROZEN_TARGETS).read_bytes() == frozen_before_fit[0]
    assert json.loads((output / ADVANTAGES).read_text())['rows'] == frozen['rows']
    state = torch.load(output / 'critic' / CRITIC_STATE, weights_only=True)
    assert state['update_count'] == 4  # online steps exclude all warmup steps


def test_vllm_seed_reaches_engine_without_pinning_each_request(tmp_path, monkeypatch):
    import sys
    from self_summarization_agent import generation, rollout_collection
    from self_summarization_agent.generation import build_generator
    from self_summarization_agent.config import ModelConfig
    captured = {}
    class Engine:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=object))
    monkeypatch.setattr(generation, '_apply_vllm_subprocess_fix', lambda: None)
    monkeypatch.setattr(generation.AutoTokenizer, 'from_pretrained', lambda *a, **k: SimpleNamespace(chat_template=None))
    generator = build_generator(ModelConfig(backend='vllm_offline', model_path='base', do_sample=True), engine_seed=7)
    assert captured['seed'] == 7
    assert 'seed' not in generator._sampling_kwargs()
    config = load_train_config('configs/train/compact_separate_gae_ppo_pretrained.yaml')
    config.rollout = replace(config.rollout, sampling_seed=7, gpu_ids=[])
    options = {}
    monkeypatch.setattr(rollout_collection, 'build_generator', lambda *a, **kw: options.update(kw))
    rollout_collection._build_rollout_generator(config, tmp_path / 'actor', split='train')
    assert options['engine_seed'] == 7 and options['sampling_extra'] == {}
    assert resolved_rollout_sampling_profile(config, split='train')['engine_seed'] == 7
