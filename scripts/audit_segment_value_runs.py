"""Read-only audit of frozen critic targets and saved judge decisions.

No models are loaded and no rollout/checkpoint artifacts are modified. Outputs
are written only to --output. Run with the repository's Python environment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def value_metrics(items):
    counts = Counter(item['rollout'] for item in items)
    weights = [1 / (len(counts) * counts[x['rollout']]) for x in items]
    def mean(fn):
        return math.fsum(w * fn(x) for w, x in zip(weights, items))
    reward = mean(lambda x: x['reward'])
    value = mean(lambda x: x['value'])
    advantage = mean(lambda x: x['advantage'])
    reward_var = mean(lambda x: (x['reward'] - reward) ** 2)
    value_var = mean(lambda x: (x['value'] - value) ** 2)
    mse = mean(lambda x: (x['reward'] - x['value']) ** 2)
    residual_var = mse - (reward - value) ** 2
    covariance = mean(lambda x: (x['reward'] - reward) * (x['value'] - value))
    return {
        'states': len(items), 'rollouts': len(counts), 'mean_reward': reward,
        'mean_value': value, 'value_std': math.sqrt(value_var), 'value_mse': mse,
        'constant_oracle_mse': reward_var,
        'explained_variance': 1 - residual_var / reward_var if reward_var else None,
        'correlation': covariance / math.sqrt(reward_var * value_var) if reward_var * value_var else None,
        'advantage_variance_vs_reward': mean(lambda x: (x['advantage'] - advantage) ** 2) / reward_var if reward_var else None,
        'target_mc_mae': mean(lambda x: abs(x['target'] - x['reward'])),
        'mean_advantage': advantage,
    }


def audit_frozen(run, max_iteration=None):
    results = []
    for checkpoint in sorted((run / 'checkpoints').glob('iteration-*')):
        if max_iteration is not None and int(checkpoint.name.split('-')[-1]) > max_iteration:
            continue
        path = checkpoint / 'frozen_value_targets.json'
        if not path.is_file() or not (checkpoint / '.complete').is_file():
            continue
        body = json.loads(path.read_text())
        rows = {tuple(x['key']): x for x in body['rows']}
        assert len(rows) == len(body['rows'])
        items = []
        errors = []
        for chain in body['chains']:
            next_value = next_advantage = 0.0
            for state in reversed(chain):
                row = rows[tuple(state['key'])]
                transition_reward = state['reward'] if state['terminal'] else 0.0
                expected = transition_reward + next_value - row['value'] + body['method']['lambda'] * next_advantage
                if abs(expected - row['advantage']) > 1e-7 or abs(row['target'] - row['value'] - row['advantage']) > 1e-7:
                    errors.append(state['key'])
                items.append(dict(row, rollout=state['key'][0], reward=state['reward'], depth=state['segment_index']))
                next_value, next_advantage = row['value'], row['advantage']
        assert len(items) == len(rows)
        result = {'checkpoint': checkpoint.name, 'recurrence_errors': errors, 'all': value_metrics(items)}
        for name, selected in [('initial', [x for x in items if x['depth'] == 1]),
                               ('post_compaction', [x for x in items if x['depth'] > 1])]:
            if selected:
                result[name] = value_metrics(selected)
        results.append(result)
    return results


def audit_judges(run, max_iteration):
    from self_summarization_agent.judge import parse_judge_response
    stats = Counter()
    mismatches = []
    examples = []
    per_file = []
    for path in sorted((run / 'rollouts').glob('*.jsonl')):
        if not (path.name.endswith('.judged.jsonl') or path.name.endswith('.eval.jsonl')):
            continue
        iteration = int(path.name.split('.')[0].split('-')[-1])
        if iteration > max_iteration:
            continue
        local = Counter()
        for row in read_jsonl(path):
            stats['rollouts'] += 1
            local['rollouts'] += 1
            judge = row['judge']
            stats['stored_parse_errors'] += bool(judge.get('parse_error'))
            raw = judge.get('judge_response')
            outcome = judge['outcome']
            expected_reward = 1 if outcome == 'correct_answer' else -1
            if any(r != expected_reward for r in row['turn_rewards'].values()):
                mismatches.append({'file': path.name, 'query': row['query_id'], 'kind': 'reward'})
            stats['intervals'] += len(row['turn_rewards'])
            if not raw:
                stats['without_judge_generation'] += 1
                continue
            stats['judge_generations'] += 1
            local['judge_generations'] += 1
            parsed = parse_judge_response(raw)
            if parsed['parse_error'] != judge['parse_error'] or (not parsed['parse_error'] and (parsed['correct'] != (outcome == 'correct_answer'))):
                mismatches.append({'file': path.name, 'query': row['query_id'], 'kind': 'reparse'})
            final = raw.rsplit('</think>', 1)[-1].strip()
            # Independent line-oriented verdict cross-check for the actual outputs.
            matches = re.findall(r'^\s*correct:\s*(yes|no)\s*[.!]?\s*$', final, re.I | re.M)
            if len(matches) == 1:
                stats['plain_verdict_crosschecks'] += 1
                if (matches[0].lower() == 'yes') != (outcome == 'correct_answer'):
                    mismatches.append({'file': path.name, 'query': row['query_id'], 'kind': 'independent_verdict'})
            else:
                stats['non_plain_verdicts'] += 1
            if row.get('status') != 'completed':
                mismatches.append({'file': path.name, 'query': row['query_id'], 'kind': 'judged_noncompleted'})
            prompt = judge.get('judge_prompt') or ''
            answer_match = re.search(r'\[correct_answer\]: (.*?)\nfocus only', prompt, re.S)
            if len(examples) < 12 and iteration in (0, 1, max_iteration):
                examples.append({'file': path.name, 'query_id': row['query_id'], 'outcome': outcome,
                                 'gold_in_prompt': answer_match.group(1) if answer_match else None,
                                 'response': row.get('final_answer'), 'judge_final': final})
        per_file.append({'file': path.name, **local})
    return {'counts': dict(stats), 'mismatches': mismatches, 'files': per_file, 'examples': examples}


def audit_evidence(run, dataset_path, iterations):
    """Check matched tasks, judge gold, and cached exact-token training views."""
    from self_summarization_agent.train_step import samples_from_rollout_rows
    from self_summarization_agent.separate_value import validate_samples
    from self_summarization_agent.segment_returns import ordered_rollout_chains
    gold = {str(row['query_id']): row for row in read_jsonl(dataset_path)}
    result = {}
    for iteration in iterations:
        for split in ('train', 'eval'):
            if split == 'train' and iteration == 0:
                continue
            name = f'iteration-{iteration:05d}' + ('.judged.jsonl' if split == 'train' else '.eval.jsonl')
            path = run / 'rollouts' / name
            if not path.is_file():
                continue
            stats = Counter()
            queries = {}
            mismatches = []
            examples = []
            for row in read_jsonl(path):
                query_id = row['query_id']
                queries[query_id] = row['query']
                if row['query'] != gold[query_id]['query']:
                    mismatches.append({'query': query_id, 'kind': 'question_text'})
                judge = row['judge']
                if judge.get('judge_prompt'):
                    match = re.search(r'\[correct_answer\]: (.*?)\nfocus only', judge['judge_prompt'], re.S)
                    if not match or match.group(1) != str(gold[query_id]['answer']):
                        mismatches.append({'query': query_id, 'kind': 'gold_answer'})
                    stats['gold_prompt_checks'] += 1
                stats['status/' + row['status']] += 1
                for turn in row['turn_records']:
                    if turn.get('format_anomaly'):
                        stats['anomaly/' + str(turn['format_anomaly'])] += 1
                if split == 'train':
                    samples = samples_from_rollout_rows([row], expected_checkpoint_id=row['policy_checkpoint_id'])
                    validate_samples(samples, max_sequence_length=27000)
                    if samples:
                        ordered_rollout_chains([row], samples)
                    stats['validated_rollouts'] += 1
                    stats['validated_exact_intervals'] += len(samples)
                    for s in samples:
                        stats['state/' + s.state_kind] += 1
                        stats['trainable_tokens'] += sum(s.completion_mask)
                if judge.get('judge_response') and judge['outcome'] == 'wrong_answer' and len(examples) < 3:
                    examples.append({'query_id': query_id, 'gold': gold[query_id]['answer'],
                                     'response': row['final_answer'],
                                     'judge_final': judge['judge_response'].rsplit('</think>', 1)[-1].strip()})
            result[name] = {'counts': dict(stats), 'queries': queries, 'mismatches': mismatches,
                            'wrong_answer_examples': examples}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/train'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--judges', action='store_true')
    parser.add_argument('--evidence-only', action='store_true')
    parser.add_argument('--dataset', type=Path, default=Path('bc-plus/data/browsecomp_plus_decrypted.jsonl'))
    parser.add_argument('--max-iteration', type=int)
    args = parser.parse_args()
    names = ['qwen-bcplus-compact-separate-gae-ppo', 'qwen-bcplus-train']
    result = {'runs': {}}
    if args.evidence_only:
        for name in names:
            print('Auditing evidence:', name, flush=True)
            result['runs'][name] = audit_evidence(args.artifacts / name, args.dataset, (0, 1, 8, 16))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print('Wrote', args.output, flush=True)
        return
    for name in names:
        run = args.artifacts / name
        evaluations = list(read_jsonl(run / 'eval_metrics.jsonl'))
        steps = list(read_jsonl(run / 'step_metrics.jsonl'))
        if args.max_iteration is not None:
            evaluations = [r for r in evaluations if r['iteration'] <= args.max_iteration]
            steps = [r for r in steps if int(r['next_checkpoint_id'].split('-')[-1]) <= args.max_iteration]
        result['runs'][name] = {'eval': evaluations, 'steps': steps}
    gae = result['runs'][names[0]]
    gae['frozen'] = audit_frozen(args.artifacts / names[0], args.max_iteration)
    common = min(len(result['runs'][n]['eval']) for n in names)
    result['comparison'] = {
        'common_eval_iterations': common,
        'mean_eval_accuracy': {n: sum(r['eval_accuracy'] for r in result['runs'][n]['eval'][1:common]) / (common - 1) for n in names},
    }
    if args.judges:
        for name in names:
            print('Auditing judges:', name, flush=True)
            result['runs'][name]['judges'] = audit_judges(args.artifacts / name, common - 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    main()
