"""Wrapper startup ordering and value-config dispatch; no rollout workers launched."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from self_summarization_agent import iteration_launcher as launcher
from self_summarization_agent.config import load_train_config, parse_cli_overrides

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('current,target,fail_startup', [(0, 2, False), (3, 4, False), (0, 2, True)])
def test_wrapper_runs_startup_before_iterations_and_stops_on_failure(tmp_path, current, target, fail_startup):
    binary = tmp_path / 'bin'
    binary.mkdir()
    log = tmp_path / 'commands.jsonl'
    uv = binary / 'uv'
    uv.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
with open(os.environ['TEST_UV_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if '--pretrain-only' in sys.argv and os.environ.get('TEST_FAIL_STARTUP') == '1':
    sys.exit(7)
''')
    uv.chmod(0o755)
    config = tmp_path / 'config with spaces.yaml'
    config.write_text('training: {}\n')  # fake uv records commands without loading models
    root = tmp_path / 'run with spaces'
    override = 'training.value_pretraining.reuse_rollouts_path=/data/batch with spaces.jsonl'
    result = subprocess.run(['bash', str(REPO / 'scripts/train_iterations.sh'),
        '--config', str(config), '--latest-root', str(root), '--iterations', str(target),
        '--current-iteration', str(current), '--set', override],
        env={**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH'],
             'TEST_UV_LOG': str(log), 'TEST_FAIL_STARTUP': str(int(fail_startup))},
        text=True, capture_output=True)
    assert result.returncode == (7 if fail_startup else 0), result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][calls[0].index('--iteration') + 1] == '0'
    assert '--pretrain-only' in calls[0]
    assert len(calls) == (1 if fail_startup else 1 + target - current)
    for command in calls:
        assert command[:3] == ['run', '-m', 'self_summarization_agent.iteration_launcher']
        assert '--auto-value-pretraining' in command and '--resume' in command
        assert command[command.index('--config') + 1] == str(config)
        assert command[command.index('--latest-root') + 1] == str(root)
        assert override in command
    if not fail_startup:
        assert [int(cmd[cmd.index('--iteration') + 1]) for cmd in calls[1:]] == list(range(current + 1, target + 1))
        assert all('--pretrain-only' not in command for command in calls[1:])


@pytest.mark.parametrize('preset,extra,expected', [
    ('compact_separate_gae_ppo', [], True),
    ('compact_separate_mc_ppo', [], True),
    ('compact_separate_gae_ppo_pretrained', [], True),
    ('default', [], False),
    ('compact_value_mc', [], False),  # historical shared-head method
    ('compact_separate_gae_ppo', ['training.value_pretraining.enabled=false'], False),
    ('compact_separate_gae_ppo', ['training.value.enabled=false', 'training.advantage_estimator=group_relative',
        'training.value.architecture=shared', 'training.value.num_bins=2', 'training.value.baseline_timing=post_fit'], False),
])
def test_startup_cli_detects_resolved_config_and_forwards_effective_override(monkeypatch, tmp_path, preset, extra, expected):
    captured = []
    def startup(config, **options):
        captured.append((config, options))
        return tmp_path / 'best' if expected else None
    monkeypatch.setattr(launcher, 'run_value_pretraining_startup', startup)
    monkeypatch.setattr(launcher, 'run_training_iteration', lambda *a, **k: pytest.fail('startup must not train actor'))
    monkeypatch.setattr(launcher, 'run_checkpoint_evaluation', lambda *a, **k: pytest.fail('startup must precede evaluation'))
    args = ['launcher', '--config', f'configs/train/{preset}.yaml', '--iteration', '0',
            '--pretrain-only', '--auto-value-pretraining', '--resume', '--latest-root', str(tmp_path)]
    for override in extra:
        args.extend(['--set', override])
    monkeypatch.setattr(sys, 'argv', args)
    launcher.main()
    config, options = captured[0]
    assert config.training.value_pretraining.enabled is expected
    assert options['resume'] is True and options['latest_root'] == str(tmp_path)
    effective = parse_cli_overrides(options['overrides'])
    if config.training.value.separate:
        assert effective['training.value_pretraining.enabled'] is expected
    else:
        assert 'training.value_pretraining.enabled' not in effective


def test_online_cli_keeps_same_auto_enabled_config_as_startup(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['launcher', '--config', 'configs/train/compact_separate_gae_ppo.yaml',
        '--iteration', '1', '--auto-value-pretraining', '--resume'])
    monkeypatch.setattr(launcher, 'run_training_iteration', lambda config, **kw: calls.append((config, kw)))
    launcher.main()
    config, options = calls[0]
    assert config.training.value_pretraining.enabled
    assert parse_cli_overrides(options['overrides'])['training.value_pretraining.enabled'] is True
    assert options['iteration'] == 1


def test_direct_launcher_keeps_configured_cold_critic(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, 'argv', ['launcher', '--config', 'configs/train/compact_separate_gae_ppo.yaml', '--iteration', '1'])
    monkeypatch.setattr(launcher, 'run_training_iteration', lambda config, **kw: calls.append(config))
    launcher.main()
    assert not calls[0].training.value_pretraining.enabled


def test_startup_disabled_does_not_resolve_or_touch_checkpoint(monkeypatch):
    config = load_train_config('configs/train/default.yaml')
    monkeypatch.setattr(launcher, 'resolve_latest_checkpoint', lambda *a: pytest.fail('non-value startup is a noop'))
    assert launcher.run_value_pretraining_startup(config, config_path='config.yaml') is None


def test_startup_delegates_at_iteration_zero_with_normal_timeout(tmp_path, monkeypatch):
    config = load_train_config('configs/train/compact_separate_gae_ppo_pretrained.yaml')
    actor, best = tmp_path / 'actor', tmp_path / 'value_pretraining' / 'best'
    calls = []
    monkeypatch.setattr(launcher, 'resolve_latest_checkpoint', lambda *a: SimpleNamespace(path=actor))
    def initial(config, **kw):
        calls.append(kw)
        return best
    monkeypatch.setattr(launcher, '_initial_pretrained_critic', initial)
    assert launcher.run_value_pretraining_startup(config, config_path='config.yaml', latest_root=tmp_path,
        resume=True, overrides=['training.value_pretraining.enabled=true']) == best
    assert calls[0]['iteration'] == 0 and calls[0]['actor'] == actor
    assert calls[0]['timeout'] == config.runtime.train_update_timeout_seconds
    assert calls[0]['resume'] is True


def test_startup_advanced_pair_skips_pretraining(tmp_path, monkeypatch):
    config = load_train_config('configs/train/compact_separate_gae_ppo_pretrained.yaml')
    monkeypatch.setattr(launcher, 'resolve_latest_checkpoint', lambda *a: SimpleNamespace(path=tmp_path / 'iteration-00003'))
    monkeypatch.setattr('self_summarization_agent.separate_value.validate_resume_checkpoint', lambda path: True)
    monkeypatch.setattr(launcher, '_run_or_skip_phase', lambda **kw: pytest.fail('paired resume must not pretrain'))
    assert launcher.run_value_pretraining_startup(config, config_path='config.yaml', latest_root=tmp_path, resume=True) is None


def test_startup_rejects_reusing_first_online_batch(tmp_path):
    config = load_train_config('configs/train/compact_separate_gae_ppo_pretrained.yaml')
    config.training.value_pretraining.reuse_rollouts_path = str(tmp_path / 'rollouts' / 'iteration-00001.jsonl')
    with pytest.raises(ValueError, match='separate from the first actor-update batch'):
        launcher._initial_pretrained_critic(config, actor=tmp_path / 'base', train_dir=tmp_path,
            config_path='config.yaml', iteration=0, command_runner=lambda cmd: 0,
            python_executable='python', resume=True, overrides=(), timeout=1)


@pytest.mark.parametrize('extra', [['--pretrain-only', '--eval-only'], ['--pretrain-only']])
def test_pretraining_cli_requires_exclusive_iteration_zero_mode(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['launcher', '--config', 'config.yaml', '--iteration', '1', *extra])
    with pytest.raises(SystemExit) as error:
        launcher.parse_args()
    assert error.value.code == 2
