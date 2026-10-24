"""Paired, stratified inference from immutable per-example predictions."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from toolrl.data import read_json,read_jsonl,write_json


def paired_bootstrap(left,right,repeats=10000,seed=20261013):
    if len({r['id'] for r in left})!=len(left) or len({r['id'] for r in right})!=len(right):raise ValueError('Duplicate IDs')
    if not left or not right:raise ValueError('Empty predictions')
    left={r['id']:r for r in left};right={r['id']:r for r in right}
    if left.keys()!=right.keys():raise ValueError('Unpaired IDs')
    ids=sorted(left)
    for i in ids:
        if left[i]['call_count']!=right[i]['call_count']:raise ValueError('Stratum mismatch')
    d=np.array([int(left[i]['call_multiset_exact'])-int(right[i]['call_multiset_exact']) for i in ids])
    strata=[[j for j,i in enumerate(ids) if (left[i]['call_count']==1)==single] for single in (True,False)]
    rng=np.random.default_rng(seed);draw=np.zeros(repeats)
    for ix in strata:
        if ix:draw+=d[rng.choice(ix,size=(repeats,len(ix)),replace=True)].sum(axis=1)
    draw/=len(ids)
    return {'difference_pp':float(d.mean()*100),'ci95_pp':(np.quantile(draw,[0.025,0.975])*100).tolist(),'samples':len(ids),'seed':seed,'bootstrap_repeats':repeats}


def analyze(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    table=[]; predictions={}
    expected_runs={'base','sft'}|{f'{m}_{s}' for m in ('continue','sparse','dense') for s in (42,43,44)}
    actual_runs={p.name for p in (root/'eval').iterdir() if p.is_dir()}
    if actual_runs!=expected_runs:raise ValueError('Final analysis requires base + SFT + all nine paired runs')
    for folder in sorted((root/'eval').iterdir()):
        if not folder.is_dir() or not (folder/'metrics.json').exists():continue
        m=read_json(folder/'metrics.json'); rows=read_jsonl(folder/'predictions.jsonl')
        config=read_json(folder/'config.json')
        expected=read_json(config['manifest'])['counts']['test']
        if len(rows)!=expected:raise ValueError(f'Incomplete held-out denominator: {folder}')
        if [r['id'] for r in rows]!=read_json(config['manifest'])['indices']['test']:raise ValueError('Held-out IDs/order differ')
        if read_json(folder/'status.json')['status']!='complete':raise ValueError('Incomplete evaluation')
        predictions[folder.name]=rows
        table.append({'run':folder.name,'n':len(rows),'correct':m['correct'],'cem_percent':100*m['call_multiset_exact'],
                      'single_correct':m['single']['correct'],'multi_correct':m['multi']['correct'],
                      'format_percent':100*m['format_ok'],'deployable_percent':100*m['deployable_exact'],
                      'truncated_percent':100*m['truncated']})
        failures=[r for r in rows if not r['call_multiset_exact']]
        write_json(out/f'{folder.name}_errors.json',{'error_counts':dict(Counter(r['prediction_error'] or 'call_mismatch' for r in failures)),
                    'diagnostic_counts':dict(Counter(e for r in failures for e in r.get('diagnostics',{}).get('error_types',[]))),
                    'all_failure_ids':[r['id'] for r in failures],'examples':sorted(failures,key=lambda r:r['id'])[:10]})
    if not table:raise ValueError('No complete held-out predictions')
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    aggregates={}
    for method in ('continue','sparse','dense'):
        values=[r['cem_percent'] for r in table if r['run'].startswith(method+'_')]
        assert len(values)==3
        aggregates[method]={'seed_count':3,'mean_CEM_percent':float(np.mean(values)),'sample_std_pp':float(np.std(values,ddof=1)),'values_percent':values}
    write_json(out/'seed_summary.json',aggregates)
    contrasts={}
    for seed in (42,43,44):
        for other in ('sparse','continue'):
            a,b=f'dense_{seed}',f'{other}_{seed}'
            if a in predictions and b in predictions:contrasts[f'{a}-{b}']=paired_bootstrap(predictions[a],predictions[b])
    # Average across the fixed three seeds, paired on the same stratified IDs.
    if all(f'{m}_{s}' in predictions for s in (42,43,44) for m in ('dense','sparse')):
        ids=sorted(r['id'] for r in predictions['dense_42'])
        maps={k:{r['id']:r for r in v} for k,v in predictions.items()}
        assert all(sorted(maps[f'{m}_{s}'])==ids for s in (42,43,44) for m in ('dense','sparse'))
        d=np.mean([[int(maps[f'dense_{s}'][i]['call_multiset_exact'])-int(maps[f'sparse_{s}'][i]['call_multiset_exact']) for i in ids] for s in (42,43,44)],axis=0)
        rng=np.random.default_rng(20261013);sample=np.zeros(10000)
        for single in (True,False):
            ix=[j for j,i in enumerate(ids) if (maps['dense_42'][i]['call_count']==1)==single]
            if ix:sample+=d[rng.choice(ix,size=(10000,len(ix)),replace=True)].sum(axis=1)
        sample/=len(ids)
        contrasts['fixed_three_seed_mean_dense_minus_sparse']={'difference_pp':float(d.mean()*100),'ci95_pp':(np.quantile(sample,[.025,.975])*100).tolist(),
                    'scope':'sample uncertainty conditional on these three seeds; not full training variance'}
    write_json(out/'contrasts.json',contrasts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));ax.bar([r['run'] for r in table],[r['cem_percent'] for r in table]);ax.set_ylabel('Strict call exact match (%)');ax.tick_params(axis='x',rotation=40);fig.tight_layout();fig.savefig(out/'strict_success.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    resources=[]
    for run in sorted((root/'runs').iterdir()):
        if not run.is_dir():continue
        report=read_json(run/'report.json');resources.append({'run':run.name,**report})
        if not (run/'rollouts.jsonl').exists():continue
        rows=read_jsonl(run/'rollouts.jsonl');groups=[rows[i:i+4] for i in range(0,len(rows),4)]
        x=[];zero=[];reward=[]
        for i in range(0,len(groups),20):
            window=groups[i:i+20];x.append(i+len(window))
            zero.append(np.mean([g[0]['group_std']<=1e-8 for g in window]))
            reward.append(np.mean([r['reward']['total'] for g in window for r in g]))
        axes[0].plot(x,zero,label=run.name);axes[1].plot(x,reward,label=run.name)
    axes[0].set_ylabel('Zero-variance group fraction');axes[1].set_ylabel('Mean training reward')
    for ax in axes:ax.set_xlabel('Request exposures');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'reward_signal.png',dpi=160);plt.close(fig)
    write_json(out/'resources.json',resources)
    fig,ax=plt.subplots(figsize=(9,4))
    for method in ('continue','sparse','dense'):
        for seed in (42,43,44):
            points=[]
            for folder in (root/'validation').glob(f'{method}_{seed}_step*'):
                step=int(folder.name.split('step')[-1]);m=read_json(folder/'metrics.json')
                points.append((step*(2 if method=='continue' else 1),100*m['call_multiset_exact']))
            if points:
                points.sort();ax.plot(*zip(*points),marker='o',label=f'{method}_{seed}')
    ax.set_xlabel('Request exposures');ax.set_ylabel('Validation CEM (%)');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'validation_curve.png',dpi=160);plt.close(fig)
    return table
