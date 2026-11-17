"""Snapshot local training metrics and render the compaction blog figures.

Run: .venv/bin/python scripts/build_compaction_blog.py
Use --plots-only to reproduce from the frozen assets without reading live runs.
Only derived blog assets are written; trajectories and training code are untouched.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_gae_ppo_vs_grpo import compact_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/blog/learning-to-compact-assets'
RUNS = {
    'grpo': 'qwen-bcplus-train',
    'none': 'qwen-bcplus-no-compact-32k-train',
    '6k': 'qwen-bcplus-compact-6k-train',
    '24k': 'qwen-bcplus-compact-24k-train',
    'no_loss': 'qwen-bcplus-no-compaction-loss',
    'segment': 'qwen-bcplus-compact-separate-gae-ppo',
    'token': 'qwen-bcplus-token-mc-ppo',
}
LABELS = {'grpo': '12k + summary loss', 'none': 'No compaction', '6k': '6k threshold',
          '24k': '24k threshold', 'no_loss': '12k, summary loss masked',
          'segment': 'Segment GAE PPO', 'token': 'Token MC PPO'}
COLORS = {'grpo': '#2278ad', 'none': '#59636e', '6k': '#bd536b', '24k': '#008574',
          'no_loss': '#8057ae', 'segment': '#c76c2b', 'token': '#8057ae'}


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def snapshot():
    sources = []
    def save(path, name):
        raw = path.read_bytes()
        (OUT / name).write_bytes(raw)
        sources.append(dict(path=str(path.relative_to(ROOT)), snapshot=name,
                            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    for method, run in RUNS.items():
        for filename in ['eval_metrics.jsonl', 'phase_timings.jsonl', 'step_metrics.jsonl']:
            save(ROOT / 'artifacts/train' / run / filename, f'{method}-{filename}')
    for name in ['default', 'no_compact_32k', 'compact_6k', 'compact_24k',
                 'no_compaction_loss', 'compact_separate_gae_ppo', 'compact_separate_token_mc_ppo']:
        save(ROOT / f'configs/train/{name}.yaml', f'{name}.yaml')
    old = ROOT / 'docs/analysis/token-segment-grpo-assets'
    # Ensure the audited per-question evidence still describes these run metrics.
    for source in json.loads((old / 'sources.json').read_text()):
        path = Path(source['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], path
    save(old / 'per-question.csv', 'critic-per-question.csv')
    save(old / 'summary.json', 'critic-audit-summary.json')
    sets = []
    for method in ['no_loss', '24k']:
        ev = read_jsonl(OUT / f'{method}-eval_metrics.jsonl')
        for i in [0, ev[-1]['iteration']]:
            path = ROOT / 'artifacts/train' / RUNS[method] / 'rollouts' / f'iteration-{i:05d}.eval.jsonl'
            rows = compact_rows(path)
            assert len(rows) == ev[i]['eval_total'] == 50
            assert sum(r['outcome'] == 'correct_answer' for r in rows) == ev[i]['eval_correct']
            sets.append({r['query_id'] for r in rows})
            name = f'{method}-checkpoint-{i}-outcomes.json'
            (OUT / name).write_text(json.dumps(rows, indent=2) + '\n')
            # Full file hash records the exact evidence read without copying token arrays.
            h = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            sources.append(dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest(),
                                extracted_snapshot=name))
    assert all(s == sets[0] for s in sets)
    qids = sets[0]
    outcomes = list(csv.DictReader((OUT / 'critic-per-question.csv').open()))
    baseline = []
    for i in range(30):
        all_rows = [r for r in outcomes if r['method'] == 'grpo' and int(r['checkpoint']) == i]
        metric = read_jsonl(OUT / 'grpo-eval_metrics.jsonl')[i]
        assert len(all_rows) == 100 and sum(int(r['correct']) for r in all_rows) == metric['eval_correct']
        selected = [r for r in all_rows if r['query_id'] in qids]
        assert len(selected) == 50
        baseline.append(dict(iteration=i, eval_total=50,
                             eval_accuracy=sum(int(r['correct']) for r in selected)/50,
                             eval_malformed=sum(int(r['malformed']) for r in selected)))
    (OUT / 'grpo-common50.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in baseline))
    (OUT / 'sources.json').write_text(json.dumps(dict(
        snapshot_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        common_question_ids=sorted(qids),
        limitations=['No raw per-question files available for 6k and no-compaction; membership inferred from matching current partition configs.',
                     'Configs are current files, not immutable launch manifests.',
                     'Baseline subset uses previously audited per-question outcomes; no new full trajectory integrity audit.']), indent=2)+'\n')


def render():
    ev = {m: read_jsonl(OUT / f'{m}-eval_metrics.jsonl') for m in RUNS}
    common = read_jsonl(OUT / 'grpo-common50.jsonl')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    def line(ax, rows, key, method, label=None):
        y = [100*r['eval_accuracy'] if key == 'accuracy' else
             100*r['eval_malformed']/r['eval_total'] for r in rows]
        ax.plot([r['iteration'] for r in rows], y, label=label or LABELS[method],
                color=COLORS[method], marker='o', markersize=3, linewidth=1.8)
    def finish(fig, axes, title, footer, stem):
        fig.suptitle(title, fontsize=17, weight='bold', x=.08, ha='left')
        for ax in np.asarray(axes).flat:
            ax.grid(axis='y', color='#e2e6eb'); ax.set_axisbelow(True)
        fig.text(.08, .025, footer, color='#586273', fontsize=9)
        fig.tight_layout(rect=(0,.10,1,.93))
        for suffix in ['png','svg','pdf']:
            fig.savefig(OUT/f'{stem}.{suffix}', dpi=200, facecolor='white')
        plt.close(fig)
    for methods, title, stem in [
        (['grpo','none'], 'Can learned memory replace retained history?', 'compaction-vs-none'),
        (['6k','grpo','24k'], 'How often should the agent compact?', 'thresholds'),
        (['grpo','no_loss'], 'Does training summary tokens help?', 'summary-loss')]:
        fig, axes = plt.subplots(1,2,figsize=(12,4.8))
        for m in methods:
            rows = common if m == 'grpo' else ev[m]
            if stem == 'summary-loss':
                rows = [r for r in rows if r['iteration'] <= ev['no_loss'][-1]['iteration']]
            line(axes[0],rows,'accuracy',m)
            line(axes[1],rows,'malformed',m)
        for ax, label in zip(axes,['Accuracy (%)','Malformed trajectories (%)']):
            ax.set(xlabel='Evaluated checkpoint',ylabel=label,ylim=(0,100 if 'Malformed' in label else 60))
        axes[0].legend(frameon=False,fontsize=9)
        footer = '50-question comparisons; baseline restricted to the ablation subset. Unsmoothed; one training seed.\n'
        footer += '6k / no-compaction question membership inferred from configs; historical training pools and context caps differ.'
        if stem == 'summary-loss':
            footer = 'Same 50 evaluation questions verified at endpoints; checkpoints 0–24. Unsmoothed; one seed.\nTraining pools differ (730 vs 780); this is not an isolated loss-mask experiment.'
        finish(fig,axes,title,footer,stem)
    fig,axes = plt.subplots(1,3,figsize=(15,5))
    for m in ['grpo','segment','token']:
        label = 'GRPO' if m == 'grpo' else LABELS[m]
        line(axes[0],ev[m],'accuracy',m,label)
        line(axes[1],ev[m],'malformed',m,label)
        phases = read_jsonl(OUT/f'{m}-phase_timings.jsonl')
        hours = [sum(p['elapsed_seconds'] for p in phases if p.get('exit_code',0)==0
                     and (p['phase']=='value_pretraining' or p.get('iteration',999)<=r['iteration']))/3600 for r in ev[m]]
        axes[2].plot(hours,[100*r['eval_accuracy'] for r in ev[m]],color=COLORS[m],linewidth=1.8)
    axes[0].set(xlabel='Evaluated checkpoint',ylabel='Accuracy (%)',ylim=(0,55))
    axes[1].set(xlabel='Evaluated checkpoint',ylabel='Malformed trajectories (%)',ylim=(0,80))
    axes[2].set(xlabel='Successful phase hours to produce checkpoint',ylabel='Accuracy (%)',ylim=(0,55))
    axes[0].legend(frameon=False,fontsize=9)
    finish(fig,axes,'Where should the learning signal live?',
           '100 shared questions; greedy decoding; unsmoothed. Cost includes critic pretraining; excludes failures / idle gaps.\nPhase hours are not GPU-hours. One training seed per method; last evaluated checkpoint is 29.', 'critics')
    summary = {}
    for m,rows in {**ev,'grpo_common50':common}.items():
        summary[m] = dict(initial_accuracy=rows[0]['eval_accuracy'],last_checkpoint=rows[-1]['iteration'],
                          last_accuracy=rows[-1]['eval_accuracy'],
                          last_malformed=rows[-1]['eval_malformed']/rows[-1]['eval_total'],
                          mean_1_24=float(np.mean([r['eval_accuracy'] for r in rows if 1<=r['iteration']<=24])),
                          mean_20_29=float(np.mean([r['eval_accuracy'] for r in rows if 20<=r['iteration']<=29])))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.plots_only:
        snapshot()
    render()
