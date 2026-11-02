from dataclasses import replace
import json

import pytest

from self_summarization_agent import value_pretraining as module
from self_summarization_agent import value_diagnostics as diagnostics
from value_pretraining_fixtures import make_batch


@pytest.mark.parametrize('scores,patience,delta,expected,best,reason', [
    ([.3, .2, .2, .2, .2, .2, .2], 5, 0., 6, 1, 'patience'),
    ([.2, .3, .4, .5, .6, .7], 5, 0., 5, 0, 'patience'),
    ([.3, .29, .28], 2, .03, 2, 0, 'patience'),
    ([.9 - i * .01 for i in range(21)], 5, 0., 20, 20, 'max_epochs'),
])
def test_selection_patience_cap_ties_and_epoch_zero(tmp_path, monkeypatch, scores, patience, delta, expected, best, reason):
    batch = make_batch(tmp_path)
    batch.config.training.value_pretraining = replace(batch.config.training.value_pretraining,
        max_epochs=20, patience=patience, min_delta=delta)
    sequence = iter(scores)
    original = diagnostics.value_diagnostics
    def controlled(*args, **kwargs):
        metrics = original(*args, **kwargs)
        if kwargs.get('prefix') == 'validation':
            metrics['validation/brier_score'] = next(sequence)
        return metrics
    monkeypatch.setattr(diagnostics, 'value_diagnostics', controlled)
    output = tmp_path / 'warmup'
    selected = module.fit_pretraining(batch.config, batch.actor, batch.path, output, batch.allowed, device='cpu')
    manifest = json.loads((output / module.MANIFEST).read_text())
    assert (manifest['completed_epochs'], manifest['best_epoch'], manifest['stop_reason']) == (expected, best, reason)
    assert selected.resolve().name == f'epoch-{best:05d}'
    assert (output / 'last').resolve().name == f'epoch-{expected:05d}'
    assert len(list((output / 'epochs').iterdir())) <= 2
    assert json.loads((output / module.STATE).read_text())['non_improving_epochs'] == (patience if reason == 'patience' else 0)


def test_nonfinite_validation_never_publishes_completion(tmp_path, monkeypatch):
    batch = make_batch(tmp_path)
    original = diagnostics.value_diagnostics
    def invalid(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get('prefix') == 'validation':
            result['validation/brier_score'] = float('nan')
        return result
    monkeypatch.setattr(diagnostics, 'value_diagnostics', invalid)
    output = tmp_path / 'warmup'
    with pytest.raises(ValueError, match='Nonfinite'):
        module.fit_pretraining(batch.config, batch.actor, batch.path, output, batch.allowed, device='cpu')
    assert not (output / '.complete').exists()


def test_completion_retry_source_binding_and_all_failure_validation(tmp_path, monkeypatch):
    batch = make_batch(tmp_path)
    _, _, split = module.prepare_dataset(batch.config, batch.actor, batch.path, batch.allowed)
    for row in batch.rows:
        if row['query_id'] in split['validation_query_ids']:
            row['judge']['outcome'] = 'wrong_answer'
            row['turn_rewards'] = {key: -1. for key in row['turn_rewards']}
    batch.path.write_text(''.join(json.dumps(row) + '\n' for row in batch.rows))
    output = tmp_path / 'warmup'
    original = module.atomic_json
    def fail_marker(path, body):
        if path.name == '.complete':
            raise RuntimeError('publication interrupted')
        return original(path, body)
    monkeypatch.setattr(module, 'atomic_json', fail_marker)
    with pytest.raises(RuntimeError, match='publication'):
        module.fit_pretraining(batch.config, batch.actor, batch.path, output, batch.allowed, device='cpu')
    committed = json.loads((output / module.STATE).read_text())
    assert not (output / '.complete').exists()
    monkeypatch.setattr(module, 'atomic_json', original)
    module.fit_pretraining(batch.config, batch.actor, batch.path, output, batch.allowed, resume=True, device='cpu')
    assert json.loads((output / module.STATE).read_text()) == committed
    history = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert all(row['validation/explained_variance'] is None for row in history)
    altered = replace(batch.config, training=replace(batch.config.training,
        value_pretraining=replace(batch.config.training.value_pretraining, seed=1)))
    with pytest.raises(ValueError, match='different inputs'):
        module.completed_pretraining(altered, batch.actor, output)
    batch.path.write_text(batch.path.read_text() + '\n')
    with pytest.raises(ValueError, match='dataset changed'):
        module.completed_pretraining(batch.config, batch.actor, output)


@pytest.mark.parametrize('failure', ['checkpoint', 'best_link'])
def test_checkpoint_and_best_link_failures_resume_committed_epoch(tmp_path, monkeypatch, failure):
    from self_summarization_agent.separate_value_trainer import SeparateValueTrainer
    batch = make_batch(tmp_path)
    output = tmp_path / 'warmup'
    failed = []
    if failure == 'checkpoint':
        original = SeparateValueTrainer.save
        def save(self, path, checkpoint_id):
            if checkpoint_id.endswith('00001') and not failed:
                failed.append(True)
                raise RuntimeError('checkpoint interrupted')
            return original(self, path, checkpoint_id)
        monkeypatch.setattr(SeparateValueTrainer, 'save', save)
    else:
        original = module._links_and_prune
        def links(path, state):
            if state['epoch'] == 1 and not failed:
                failed.append(True)
                raise RuntimeError('best link interrupted')
            return original(path, state)
        monkeypatch.setattr(module, '_links_and_prune', links)
    with pytest.raises(RuntimeError, match='interrupted'):
        module.fit_pretraining(batch.config, batch.actor, batch.path, output, batch.allowed, device='cpu')
    assert not (output / '.complete').exists()
    assert json.loads((output / module.STATE).read_text())['epoch'] == (0 if failure == 'checkpoint' else 1)
    best = module.fit_pretraining(batch.config, batch.actor, batch.path, output, batch.allowed, resume=True, device='cpu')
    assert module.completed_pretraining(batch.config, batch.actor, output) == best
    assert json.loads((output / module.MANIFEST).read_text())['optimizer_steps'] == 9
    assert len(list((output / 'epochs').iterdir())) <= 2
