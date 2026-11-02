"""Read saved runs; write comparison figures and compact, reproducible evidence only.

Run from the repository root with .venv/bin/python. Does not load a model or
modify collection, checkpoints, training data, or existing analysis reports.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RUNS = {'gae': 'qwen-bcplus-compact-separate-gae-ppo', 'grpo': 'qwen-bcplus-train'}
LABEL = {'gae': 'Segment GAE PPO (pretrained critic)', 'grpo': 'GRPO'}
COLOR = {'gae': '#c75630', 'grpo': '#2679ad'}
# Only one judge outcome is present in ordinary rows. Ambiguous rows use the
# full JSON parser; large historical token arrays otherwise need not be parsed.
OUTCOME = re.compile(rb'"outcome"\s*:\s*"([^"\\]*)"')
STRING = {k: re.compile(rb'"' + k.encode() + rb'"\s*:\s*("(?:[^"\\]|\\.)*")')
          for k in ['query_id', 'query', 'status', 'sampling_profile_id', 'collection_contract']}
PARSE = re.compile(rb'"parse_error"\s*:\s*(true|false)')


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def compact_rows(path):
    rows = []
    with path.open('rb') as f:
        for line in f:
            if not line.strip():
                continue
            outcomes, parses = OUTCOME.findall(line), PARSE.findall(line)
            if len(outcomes) == 1 and len(parses) == 1:
                r = {k: json.loads(pattern.search(line).group(1)) for k, pattern in STRING.items()}
                r.update(outcome=outcomes[0].decode(), parse_error=parses[0] == b'true')
            else:
                full = json.loads(line)
                r = {k: full[k] for k in STRING}
                r.update(outcome=full['judge']['outcome'], parse_error=full['judge']['parse_error'])
            rows.append(r)
    return rows


def bootstrap_ci(values, rng):
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), (20000, len(v)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def prettify(ax, title, ylabel, percent=False):
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1))


def savefig(fig, out, stem, footer):
    fig.text(.02, .015, footer, fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .045, 1, .94), h_pad=3, w_pad=3)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(out / f'{stem}.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/analysis/gae-ppo-vs-grpo-assets')
    parser.add_argument('--figures-only', action='store_true', help='Render saved compact evidence without rereading raw rollouts')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        summary = json.loads((out/'summary.json').read_text())
        ev = {m: {r['iteration']:r for r in jsonl(out/f'{m}-eval_metrics.jsonl')} for m in RUNS}
        for row in csv.DictReader((out/'evaluation.csv').open()):
            r = ev[row['method']][int(row['iteration'])]
            for key in ['conditional_accuracy', 'checkpoint_production_hours']:
                r[key] = float(row[key])
        train = {m:{} for m in RUNS}
        for row in csv.DictReader((out/'training.csv').open()):
            train[row['method']][int(row['update'])] = dict(behavior_checkpoint=int(row['behavior_checkpoint']),accuracy=float(row['accuracy']))
        critic_csv = [{k:float(v) for k,v in row.items() if v} for row in csv.DictReader((out/'critic.csv').open())]
        render_figures(out, summary, ev, train, critic_csv)
        return
    sources, ev, steps, phases, per_question, train, checks = [], {}, {}, {}, {}, {}, {}
    for method, name in RUNS.items():
        run = ROOT / 'artifacts/train' / name
        for filename in ['eval_metrics.jsonl', 'step_metrics.jsonl', 'phase_timings.jsonl']:
            path = run / filename
            raw = path.read_bytes()
            (out / f'{method}-{filename}').write_bytes(raw)
            sources.append(dict(path=str(path), size=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                mtime_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()))
        erows = jsonl(out / f'{method}-eval_metrics.jsonl')
        assert len({r['iteration'] for r in erows}) == len(erows), 'Duplicate evaluation rows'
        ev[method] = {r['iteration']: r for r in erows}
        srows = jsonl(out / f'{method}-step_metrics.jsonl')
        steps[method] = {int(r['next_checkpoint_id'].split('-')[-1]): r for r in srows}
        assert len(steps[method]) == len(srows), 'Duplicate update rows'
        phases[method] = jsonl(out / f'{method}-phase_timings.jsonl')
        per_question[method], train[method] = {}, {}
        for iteration, metric in sorted(ev[method].items()):
            path = run / 'rollouts' / f'iteration-{iteration:05d}.eval.jsonl'
            rows = compact_rows(path)
            byid = {r['query_id']: r for r in rows}
            assert len(byid) == len(rows) == metric['eval_total']
            assert sum(r['outcome'] == 'correct_answer' for r in rows) == metric['eval_correct']
            assert sum(r['outcome'] == 'malformed_tool_call' for r in rows) == metric['eval_malformed']
            assert sum(r['parse_error'] for r in rows) == metric['eval_parse_errors']
            assert all(r['sampling_profile_id'] == metric['eval_sampling_profile_id'] for r in rows)
            per_question[method][iteration] = byid
            sources.append(dict(path=str(path), size=path.stat().st_size, rows=len(rows)))
            if iteration % 5 == 0:
                print(f'{method}: verified evaluation checkpoint {iteration}', flush=True)
        for update in sorted(steps[method]):
            path = run / 'rollouts' / f'iteration-{update:05d}.judged.jsonl'
            if not path.is_file():
                continue
            rows = compact_rows(path)
            count = Counter(r['outcome'] for r in rows)
            train[method][update] = dict(update=update, behavior_checkpoint=update-1,
                total=len(rows), correct=count['correct_answer'], malformed=count['malformed_tool_call'],
                accuracy=count['correct_answer']/len(rows), malformed_rate=count['malformed_tool_call']/len(rows),
                query_ids=sorted({r['query_id'] for r in rows}),
                sampling_profile_ids=sorted({r['sampling_profile_id'] for r in rows}))
            assert len(rows) == 200
            sources.append(dict(path=str(path), size=path.stat().st_size, rows=len(rows)))
            if update % 5 == 0:
                print(f'{method}: read training update {update}', flush=True)
    common = sorted(ev['gae'].keys() & ev['grpo'].keys())
    last = max(common)
    late = [i for i in common if i >= last - 9]
    reference = per_question['gae'][common[0]]
    qids = sorted(reference)
    for method in RUNS:
        for i in common:
            assert set(per_question[method][i]) == set(qids), 'Evaluation task mismatch'
            assert all(per_question[method][i][q]['query'] == reference[q]['query'] for q in qids)
            assert ev[method][i]['eval_sampling_profile_id'] == ev['gae'][0]['eval_sampling_profile_id']
    pre = json.loads((ROOT / 'artifacts/train' / RUNS['gae'] / 'value_pretraining/dataset_manifest.json').read_text())
    preids = set(pre['train_query_ids']) | set(pre['validation_query_ids'])
    assert not preids & set(qids), 'Pretraining/evaluation overlap'
    checks['matched_evaluation_questions'] = len(qids)
    checks['matched_evaluation_checkpoints'] = common
    checks['pretraining_evaluation_overlap'] = sorted(preids & set(qids))
    checks['training_evaluation_overlap'] = {m: sorted(set(qids) & set(q for r in train[m].values() for q in r['query_ids'])) for m in RUNS}
    assert all(not ids for ids in checks['training_evaluation_overlap'].values())
    checks['training_task_sets_match_by_update'] = {u: train['gae'][u]['query_ids'] == train['grpo'][u]['query_ids']
        for u in sorted(train['gae'].keys() & train['grpo'].keys())}
    checks['collection_contracts'] = sorted({r['collection_contract'] for m in RUNS for e in per_question[m].values() for r in e.values()})
    rng = np.random.default_rng(0)
    paired = {}
    differences = {}
    for window_name, window in [('initial', [0]), ('final', [last]), ('late_10', late),
                                 ('early_10', [i for i in common if 1 <= i <= 10]),
                                 ('all_post_initial', [i for i in common if i > 0])]:
        a = np.array([[per_question['gae'][i][q]['outcome'] == 'correct_answer' for q in qids] for i in window])
        b = np.array([[per_question['grpo'][i][q]['outcome'] == 'correct_answer' for q in qids] for i in window])
        delta = (a.astype(float)-b).mean(axis=0)
        differences[window_name] = delta
        paired[window_name] = dict(checkpoints=window, gae_accuracy=float(a.mean()), grpo_accuracy=float(b.mean()),
            difference=float(delta.mean()), paired_question_bootstrap_95_ci=bootstrap_ci(delta, rng))
        if len(window) == 1:
            wins, losses = int((a & ~b).sum()), int((~a & b).sum())
            paired[window_name].update(both_correct=int((a & b).sum()), gae_only=wins, grpo_only=losses,
                neither_correct=int((~a & ~b).sum()), exact_mcnemar_p=binomtest(wins,wins+losses).pvalue if wins+losses else 1.)
    summary = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), latest_evaluated_checkpoint=last,
                   latest_completed_update={m:max(steps[m]) for m in RUNS}, paired=paired, validation=checks)
    eval_csv, critic_csv = [], []
    for method in RUNS:
        pretime = sum(r['elapsed_seconds'] for r in phases[method] if r['phase']=='value_pretraining' and r['exit_code']==0)
        def elapsed(i, include_failed=False):
            return (pretime + sum(r['elapsed_seconds'] for r in phases[method]
                if r['phase'] in ['merged_collect','train_update'] and r['iteration'] <= i
                and (include_failed or r['exit_code']==0))) / 3600
        for i,r in sorted(ev[method].items()):
            complete = sum(x['status']=='completed' for x in per_question[method][i].values())
            r['completed'] = complete
            r['conditional_accuracy'] = r['eval_correct']/complete if complete else np.nan
            r['checkpoint_production_hours'] = elapsed(i)
            r['actor_steps_to_checkpoint'] = sum(s['optimizer_step_count'] for u,s in steps[method].items() if u<=i)
            eval_csv.append(dict(method=method, **r))
        selected = [ev[method][i] for i in late]
        sr = steps[method]
        summary[method] = dict(initial_accuracy=ev[method][0]['eval_accuracy'], final=ev[method][last],
            best_accuracy=max(ev[method][i]['eval_accuracy'] for i in common),
            best_checkpoints=[i for i in common if ev[method][i]['eval_accuracy']==max(ev[method][j]['eval_accuracy'] for j in common)],
            late_malformed_rate=float(np.mean([r['eval_malformed']/r['eval_total'] for r in selected])),
            late_conditional_accuracy=sum(r['eval_correct'] for r in selected)/sum(r['completed'] for r in selected),
            actor_steps_through_29=sum(r['optimizer_step_count'] for u,r in sr.items() if u<=last),
            actor_steps_through_30=sum(r['optimizer_step_count'] for r in sr.values()),
            pretraining_hours=pretime/3600, total_success_phase_hours=elapsed(max(sr)),
            total_phase_hours_including_failures=elapsed(max(sr), True),
            checkpoint_29_hours=elapsed(last),
            mean_actor_update_minutes=np.mean([r.get('actor/update_seconds',r.get('verl_fsdp/update_seconds')) for r in sr.values()])/60,
            mean_critic_subprocess_seconds=np.mean([r['value/critic_subprocess_seconds'] for r in sr.values() if 'value/critic_subprocess_seconds' in r]) if method=='gae' else 0,
            train_accuracy_last10=float(np.mean([r['accuracy'] for u,r in train[method].items() if u>=21])),
            phase_hours={ph:sum(r['elapsed_seconds'] for r in phases[method] if r['phase']==ph and r['exit_code']==0)/3600 for ph in {r['phase'] for r in phases[method]}},
            failed_phases=[r for r in phases[method] if r['exit_code']!=0])
    for u,r in steps['gae'].items():
        c = dict(update=u, behavior_checkpoint=u-1)
        for scope in ['prefit','eval_after','evalset']:
            for stratum in ['', 'initial/', 'post_compaction/']:
                for metric in ['explained_variance','brier_score','bce','mean_success_probability','mean_reward']:
                    key = f'value/{scope}/{stratum}{metric}'
                    if key in r:c[key]=r[key]
        if 'value/evalset/brier_score' in r:
            p=(r['value/prefit/mean_reward']+1)/2
            observed=(r['value/evalset/mean_reward']+1)/2
            c['heldout_train_prior_brier']=observed*(1-p)**2+(1-observed)*p**2
            c['heldout_oracle_brier']=observed*(1-observed)
        critic_csv.append(c)
    # Retained complete checkpoints can verify exact GAE recurrence without
    # loading models or rewriting the interval evidence.
    from audit_segment_value_runs import audit_frozen
    frozen = audit_frozen(ROOT / 'artifacts/train' / RUNS['gae'])
    assert all(not r['recurrence_errors'] for r in frozen)
    summary['validation']['retained_frozen_target_checkpoints']=[r['checkpoint'] for r in frozen]
    write_csv(out/'evaluation.csv', eval_csv)
    write_csv(out/'critic.csv', critic_csv)
    write_csv(out/'training.csv', [dict(method=m,**r) for m in RUNS for r in train[m].values()])
    write_csv(out/'per-question-evaluation.csv', [dict(method=m,checkpoint=i,query_id=q,outcome=r['outcome'],status=r['status'],parse_error=r['parse_error']) for m in RUNS for i,rs in per_question[m].items() for q,r in rs.items()])
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'sources.json').write_text(json.dumps(sources, indent=2))
    (out/'frozen-target-audit.json').write_text(json.dumps(frozen, indent=2))
    render_figures(out, summary, ev, train, critic_csv)
    print(json.dumps({'paired':paired,'validation':checks,'output':str(out)},indent=2),flush=True)


def render_figures(out, summary, ev, train, critic_csv):
    paired = summary['paired']
    common = sorted(ev['gae'].keys() & ev['grpo'].keys())
    last = summary['latest_evaluated_checkpoint']
    late = paired['late_10']['checkpoints']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':'#333333','text.color':'#222222'})
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    fig.suptitle('Latest runs: GAE PPO improves late accuracy and format reliability',fontsize=17,fontweight='bold')
    for m in RUNS:
        xs=common; ys=np.array([ev[m][i]['eval_accuracy'] for i in xs])
        axs[0,0].plot(xs,ys,color=COLOR[m],alpha=.35,lw=1,marker='o',ms=3)
        axs[0,0].plot(xs[4:],np.convolve(ys,np.ones(5)/5,'valid'),color=COLOR[m],lw=2.4,label=LABEL[m])
        axs[0,1].plot(xs,[ev[m][i]['eval_malformed']/ev[m][i]['eval_total'] for i in xs],color=COLOR[m],lw=2,label=LABEL[m])
        axs[1,0].plot(xs,[ev[m][i]['conditional_accuracy'] for i in xs],color=COLOR[m],lw=2,label=LABEL[m])
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:
        ax.axvspan(late[0]-.4,last+.4,color='#e9e2d3',alpha=.4,zorder=0)
        ax.set_xlabel('Evaluated checkpoint'); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    prettify(axs[0,0],'A  Evaluation accuracy','Correct / all questions',True)
    axs[0,0].set_ylim(0,.55);axs[0,0].legend(fontsize=9,loc='lower right')
    axs[0,0].text(.02,.96,'Faint: raw; bold: trailing 5 checkpoints',transform=axs[0,0].transAxes,va='top',fontsize=9)
    prettify(axs[0,1],'B  Malformed trajectories','Malformed / all questions',True);axs[0,1].set_ylim(0,.8)
    prettify(axs[1,0],'C  Correctness among completed trajectories','Correct / completed',True);axs[1,0].set_ylim(0,1)
    windows=['early_10','late_10','final']; y=np.arange(3)
    for k,window in enumerate(windows):
        row=paired[window];d=row['difference']*100;lo,hi=np.array(row['paired_question_bootstrap_95_ci'])*100
        axs[1,1].errorbar(d,k,xerr=[[d-lo],[hi-d]],fmt='o',capsize=5,color=COLOR['gae'],lw=2)
        axs[1,1].annotate(f'{d:+.1f} pp', (d,k),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
    axs[1,1].axvline(0,color='#777777',ls='--',lw=1)
    axs[1,1].set_yticks(y,['Early: checkpoints 1–10','Late: checkpoints 20–29',f'Final: checkpoint {last}'])
    axs[1,1].set_ylim(-.6,2.6)
    prettify(axs[1,1],'D  Paired accuracy difference (GAE − GRPO)','')
    axs[1,1].set_xlabel('Percentage points; 95% question-bootstrap interval')
    savefig(fig,out,'performance','100 matched questions per checkpoint; one run per method. Shading: checkpoints 20–29. Repeated checkpoints are correlated.')
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    fig.suptitle('Efficiency: better generated-token yield, higher recorded training cost',fontsize=17,fontweight='bold')
    for m in RUNS:
        rows=[ev[m][i] for i in common]
        axs[0,0].plot([r['checkpoint_production_hours'] for r in rows],[r['eval_accuracy'] for r in rows],color=COLOR[m],label=LABEL[m],lw=2,marker='o',ms=3)
        axs[0,1].plot(common,[r['eval_correct_per_1k_total_generated_tokens'] for r in rows],color=COLOR[m],lw=2)
        axs[1,0].plot(common,[r['eval_correct_per_1k_budget_consumed_tokens'] for r in rows],color=COLOR[m],lw=2)
    prettify(axs[0,0],'A  Accuracy versus cumulative phase time','Evaluation accuracy',True)
    axs[0,0].set_xlabel('Hours to produce checkpoint (includes GAE pretraining)');axs[0,0].legend(fontsize=9);axs[0,0].set_ylim(0,.55)
    prettify(axs[0,1],'B  Evaluation generated-token efficiency','Correct / 1,000 generated tokens');axs[0,1].set_xlabel('Evaluated checkpoint');axs[0,1].set_ylim(bottom=0)
    prettify(axs[1,0],'C  Efficiency including tool-result tokens','Correct / 1,000 total budget tokens');axs[1,0].set_xlabel('Evaluated checkpoint');axs[1,0].set_ylim(bottom=0)
    bottom=np.zeros(2)
    for ph,label,color in [('value_pretraining','Critic pretraining','#c9b57d'),('merged_collect','Collection + evaluation + judging','#91b6c6'),('train_update','Training update','#476c8b')]:
        values=[summary[m]['phase_hours'].get(ph,0) for m in RUNS]
        axs[1,1].bar(['GAE PPO','GRPO'],values,bottom=bottom,label=label,color=color,width=.6);bottom+=values
    for i,(m,h) in enumerate(zip(RUNS,bottom)):
        axs[1,1].text(i,h+.4,f'{h:.1f} h\n{summary[m]["actor_steps_through_30"]} actor steps',ha='center',fontsize=10)
    prettify(axs[1,1],'D  Recorded successful phases: all 30 updates','Hours');axs[1,1].set_ylim(0,max(bottom)*1.55);axs[1,1].legend(fontsize=9,loc='upper left')
    savefig(fig,out,'efficiency','Phase time includes pipeline overhead; this is not a GPU-hour benchmark. Checkpoint 30 has no saved policy evaluation.')
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    fig.suptitle('Critic learns useful summary-state values; initial-state transfer remains weak',fontsize=16,fontweight='bold')
    for key,label,color,ls in [('value/prefit/explained_variance','Pre-fit: used for actor advantages','#c75630','-'),('value/eval_after/explained_variance','After fit: same training batch','#8b8b8b','--'),('value/evalset/explained_variance','After fit: held-out greedy rollouts','#2679ad','-')]:
        rows=[c for c in critic_csv if key in c]
        axs[0,0].plot([r['update'] for r in rows],[r[key] for r in rows],label=label,color=color,ls=ls,marker='o',ms=3,lw=2)
    held=[c for c in critic_csv if 'value/evalset/explained_variance' in c]
    for stratum,label,color in [('','All states','#2679ad'),('initial/','Initial states','#8b8b8b'),('post_compaction/','Post-compaction states','#c75630')]:
        axs[0,1].plot([r['update'] for r in held],[r[f'value/evalset/{stratum}explained_variance'] for r in held],label=label,color=color,lw=2,marker='o',ms=4)
    for key,label,color,ls in [('value/evalset/brier_score','Critic','#c75630','-'),('heldout_train_prior_brier','Training-prevalence constant','#2679ad','--'),('heldout_oracle_brier','Held-out oracle constant','#8b8b8b',':')]:
        axs[1,0].plot([r['update'] for r in held],[r[key] for r in held],label=label,color=color,ls=ls,lw=2,marker='o',ms=4)
    for m in RUNS:
        rr=list(train[m].values())
        axs[1,1].plot([r['behavior_checkpoint'] for r in rr],[r['accuracy'] for r in rr],label=LABEL[m],color=COLOR[m],lw=1.7)
    for ax in axs.flat:
        ax.set_xlabel('Critic / policy update');ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    prettify(axs[0,0],'A  Explained variance: fitting versus prediction','Explained variance');axs[0,0].legend(fontsize=8)
    prettify(axs[0,1],'B  Held-out explained variance by state','Explained variance');axs[0,1].axhline(0,color='#777777',ls=':',lw=1);axs[0,1].legend(fontsize=9)
    prettify(axs[1,0],'C  Held-out Brier score (lower is better)','Brier score');axs[1,0].legend(fontsize=8)
    prettify(axs[1,1],'D  Stochastic training rollout success','Correct / rollouts',True);axs[1,1].set_xlabel('Behavior checkpoint (update − 1)');axs[1,1].legend(fontsize=8)
    for ax in [axs[0,1], axs[1,0]]:
        ax.set_xticks([r['update'] for r in held])
    savefig(fig,out,'critic','Critic metrics weight each rollout equally. Greedy held-out rollouts differ from stochastic training; post-fit EV is not the actor baseline quality.')


if __name__=='__main__':
    main()
