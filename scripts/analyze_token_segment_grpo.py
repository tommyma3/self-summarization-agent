"""Compare saved runs without modifying trajectories or loading models.
Run with .venv/bin/python scripts/analyze_token_segment_grpo.py.
"""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from analyze_gae_ppo_vs_grpo import compact_rows, jsonl, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/analysis/token-segment-grpo-assets'
RUNS = {'token': 'qwen-bcplus-token-mc-ppo', 'gae': 'qwen-bcplus-compact-separate-gae-ppo', 'grpo': 'qwen-bcplus-train'}
LABELS = {'token':'Token MC PPO', 'gae':'Segment GAE PPO', 'grpo':'GRPO'}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev, outcomes, training, summaries, sources, checks = {}, {}, {}, {}, [], {}
    reference = None
    for method, name in RUNS.items():
        run = ROOT / 'artifacts/train' / name
        for filename in ['eval_metrics.jsonl','step_metrics.jsonl','phase_timings.jsonl']:
            path = run / filename
            raw = path.read_bytes()
            (OUT/f'{method}-{filename}').write_bytes(raw)
            sources.append({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        ev[method] = jsonl(OUT/f'{method}-eval_metrics.jsonl')
        assert [r['iteration'] for r in ev[method]] == list(range(30))
        outcomes[method] = {}
        for metric in ev[method]:
            i = metric['iteration']
            rows = compact_rows(run/'rollouts'/f'iteration-{i:05d}.eval.jsonl')
            byid = {r['query_id']:r for r in rows}
            assert len(rows) == len(byid) == metric['eval_total'] == 100
            assert sum(r['outcome']=='correct_answer' for r in rows) == metric['eval_correct']
            assert sum(r['outcome']=='malformed_tool_call' for r in rows) == metric['eval_malformed']
            assert sum(r['parse_error'] for r in rows) == metric['eval_parse_errors'] == 0
            if reference is None: reference = byid
            assert set(byid)==set(reference)
            assert all(r['query']==reference[q]['query'] for q,r in byid.items())
            assert all(r['sampling_profile_id']==reference[q]['sampling_profile_id'] for q,r in byid.items())
            outcomes[method][i] = byid
            if i%10==0: print(method,'verified eval',i,flush=True)
        training[method] = {}
        for u in range(1,31):
            rows = compact_rows(run/'rollouts'/f'iteration-{u:05d}.judged.jsonl')
            assert len(rows)==200
            ids = sorted({r['query_id'] for r in rows})
            assert len(ids)==50 and not set(ids)&set(reference)
            training[method][u] = dict(query_ids=ids, correct=sum(r['outcome']=='correct_answer' for r in rows), malformed=sum(r['outcome']=='malformed_tool_call' for r in rows))
        steps=jsonl(OUT/f'{method}-step_metrics.jsonl')
        phases=jsonl(OUT/f'{method}-phase_timings.jsonl')
        summary={}
        for label,rs in [('early',ev[method][1:11]),('all_post_initial',ev[method][1:]),('late',ev[method][20:]),('last',ev[method][-1:])]:
            summary[label]={k:float(np.mean([r[k] for r in rs])) for k in ['eval_accuracy','eval_malformed','eval_avg_total_generated_tokens','eval_avg_budget_consumed_tokens','eval_avg_search_calls','eval_avg_summary_count']}
        summary['best'] = max(ev[method],key=lambda r:r['eval_accuracy'])
        summary['successful_phase_hours']={phase:sum(r['elapsed_seconds'] for r in phases if r['phase']==phase and r.get('exit_code',0)==0)/3600 for phase in {r['phase'] for r in phases}}
        summary['actor_steps']=sum(r['optimizer_step_count'] for r in steps)
        summary['critic_seconds_mean']=float(np.mean([r.get('value/critic_subprocess_seconds',0) for r in steps]))
        summary['late_train_accuracy']=sum(training[method][u]['correct'] for u in range(21,31))/2000
        if method!='grpo':
            pre=json.loads((run/'value_pretraining/dataset_manifest.json').read_text())
            assert not (set(pre['train_query_ids'])|set(pre['validation_query_ids']))&set(reference)
            summary['pretraining_rollouts']=pre['requested_rollout_count']
        summaries[method]=summary
        print(method,'complete',flush=True)
    checks['training_tasks_match_every_update']=all(training[m][u]['query_ids']==training['grpo'][u]['query_ids'] for m in RUNS for u in range(1,31))
    checks['evaluation_outcomes_verified']=9000
    checks['training_outcomes_read']=18000
    checks['eval_train_and_pretraining_disjoint']=True
    qids=sorted(reference)
    rng=np.random.default_rng(0)
    paired={}
    for a,b in itertools.combinations(RUNS,2):
        result={}
        for name,window in [('final',[29]),('early',list(range(1,11))),('late',list(range(20,30))),('all_post_initial',list(range(1,30)))]:
            diff=np.array([np.mean([int(outcomes[a][i][q]['outcome']=='correct_answer')-int(outcomes[b][i][q]['outcome']=='correct_answer') for i in window]) for q in qids])
            boot=diff[rng.integers(0,100,(20000,100))].mean(axis=1)
            result[name]={'difference':float(diff.mean()),'question_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist()}
            if name=='final':
                wins=int((diff>0).sum());losses=int((diff<0).sum())
                result[name].update(a_only=wins,b_only=losses,mcnemar_exact_p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.)
        paired[f'{a}_minus_{b}']=result
    body={'runs':summaries,'paired':paired,'checks':checks,'limitations':['One training seed per method; bootstrap conditions on these runs.','Windows and checkpoint maxima are retrospective.','Successful pipeline phase time is not GPU-hours and excludes failed attempts and restart gaps.','No fresh token-integrity or judge-semantic audit.']}
    (OUT/'summary.json').write_text(json.dumps(body,indent=2)+'\n')
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    write_csv(OUT/'per-question.csv',[dict(method=m,checkpoint=i,query_id=q,correct=int(r['outcome']=='correct_answer'),malformed=int(r['outcome']=='malformed_tool_call')) for m in RUNS for i,rs in outcomes[m].items() for q,r in rs.items()])
    colors={'token':'#8556b5','gae':'#cc6336','grpo':'#2679ad'}
    fig,axs=plt.subplots(1,3,figsize=(14,4.3))
    for m,rows in ev.items():
        x=[r['iteration'] for r in rows]
        axs[0].plot(x,[100*r['eval_accuracy'] for r in rows],label=LABELS[m],color=colors[m],linewidth=1.7)
        axs[1].plot(x,[r['eval_malformed'] for r in rows],color=colors[m],linewidth=1.7)
        ph=jsonl(OUT/f'{m}-phase_timings.jsonl')
        hours=[sum(p['elapsed_seconds'] for p in ph if p.get('exit_code',0)==0 and (p['phase']=='value_pretraining' or p.get('iteration',999)<=i))/3600 for i in x]
        axs[2].plot(hours,[100*r['eval_accuracy'] for r in rows],color=colors[m],linewidth=1.7)
    for ax,title in zip(axs,['Held-out accuracy','Malformed trajectories','Accuracy vs recorded pipeline cost']):
        ax.set_title(title,loc='left',fontsize=12);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[0].set(ylabel='Correct (%)',xlabel='Evaluated checkpoint',ylim=(0,55))
    axs[1].set(ylabel='Malformed (%)',xlabel='Evaluated checkpoint',ylim=(0,80))
    axs[2].set(ylabel='Correct (%)',xlabel='Successful phase hours to produce checkpoint',ylim=(0,55))
    axs[0].legend(fontsize=9,loc='lower right')
    fig.text(.02,.015,'Same 100 questions; greedy decoding; one training seed each. Checkpoint 30 has no saved policy evaluation. Cost is not normalized GPU-hours.',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1))
    for suffix in ['png','svg','pdf']: fig.savefig(OUT/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps({'paired':paired,'checks':checks,'critic_mean_seconds':{m:s['critic_seconds_mean'] for m,s in summaries.items()}},indent=2))

if __name__=='__main__': main()
