"""Deterministic, unrepaired predictions and independently scored metrics."""
import json
from pathlib import Path
import torch
from transformers import set_seed
from toolrl.data import load_prepared,write_json,append_jsonl,read_json
from toolrl.runtime import load_model,tokenizer,Monitor,init_run,fingerprint_adapter,clear
from toolrl.eval_strict import evaluate_prediction
from toolrl.rewards_train import score
from toolrl.metrics import diagnostics


def metrics(rows):
    result={'samples':len(rows)}
    for key in ('call_multiset_exact','sequence_exact','format_ok','deployable_exact','truncated'):
        result[key]=sum(bool(r[key]) for r in rows)/len(rows) if rows else None
    for name,predicate in [('single',lambda r:r['call_count']==1),('multi',lambda r:r['call_count']>1)]:
        subset=[r for r in rows if predicate(r)]
        result[name]={'samples':len(subset),'correct':sum(r['call_multiset_exact'] for r in subset)}
    result['correct']=sum(r['call_multiset_exact'] for r in rows)
    result['average_tokens']=sum(r['generated_tokens'] for r in rows)/len(rows) if rows else None
    totals={k:sum(r.get('diagnostics',{}).get(k,0) for r in rows) for k in ('key_tp','key_fp','key_fn','value_correct','value_total')}
    denom=2*totals['key_tp']+totals['key_fp']+totals['key_fn']
    result['parameter_key_micro_f1']=2*totals['key_tp']/denom if denom else 1.0
    result['parameter_value_accuracy']=totals['value_correct']/totals['value_total'] if totals['value_total'] else None
    result['parameter_counts']=totals
    for key in ('tool_names_exact','extra_call','duplicate_call'):
        result[key]=sum(r.get('diagnostics',{}).get(key,False) for r in rows)/len(rows) if rows else None
    return result


def generate(model,tok,prompt,cap,sample=False):
    inputs=tok.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
    inputs={k:v.to(model.device) for k,v in inputs.items()}
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=cap,do_sample=sample,
                              **({'temperature':0.8,'top_p':0.95,'top_k':0} if sample else {}),
                              pad_token_id=tok.pad_token_id,use_cache=True,disable_compile=True)
    ids=output[0,inputs['input_ids'].shape[1]:].tolist()
    text=tok.decode(ids,skip_special_tokens=True)
    return text,ids


def generate_group(model,tok,prompt,cap):
    return generate_batch(model,tok,[prompt]*4,cap,True)


def generate_batch(model,tok,prompts,cap,sample=False):
    inputs=tok.apply_chat_template(prompts,tokenize=True,add_generation_prompt=True,padding=True,return_dict=True,return_tensors='pt')
    inputs={k:v.to(model.device) for k,v in inputs.items()}
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=cap,do_sample=sample,
                              **({'temperature':0.8,'top_p':0.95,'top_k':0} if sample else {}),
                              pad_token_id=tok.pad_token_id,use_cache=True,disable_compile=True)
    eos=model.generation_config.eos_token_id
    endings=set(eos if isinstance(eos,list) else [eos])
    group=[]
    for item in output[:,inputs['input_ids'].shape[1]:].tolist():
        end=next((i+1 for i,v in enumerate(item) if v in endings),len(item))
        ids=item[:end];group.append((tok.decode(ids,skip_special_tokens=True),ids))
    return group


def evaluate(config,prepared,out,adapter=None,split='val',limit=None,lock=None,probe=False):
    if split in ('test','official'):
        if lock is None:raise ValueError('Held-out generation requires a frozen final-evaluation lock')
        locked=read_json(lock)
        assert locked['config']==config
        assert fingerprint_adapter(adapter) in locked['adapters']
        if limit is not None or probe:raise ValueError('Never subset held-out evaluation')
    records=load_prepared(prepared,split,config)
    if probe:
        records=[r for r in records if r['call_count']==1][:20]+[r for r in records if r['call_count']>1][:20]
    elif limit is not None:records=records[:limit]
    out=init_run(out,config,{'adapter':str(adapter),'adapter_hashes':fingerprint_adapter(adapter),'split':split,'probe':probe,'ids':[r['id'] for r in records]})
    model=None
    try:
        set_seed(20261013)
        tok=tokenizer(config);model=load_model(config,adapter,False);model.eval()
        predictions=[]
        with Monitor(out/'gpu.jsonl') as monitor:
            batch_size=1 if probe else config['eval_batch_size']
            for start in range(0,len(records),batch_size):
                batch=records[start:start+batch_size]
                outputs=generate_group(model,tok,batch[0]['prompt'],config['completion_cap']) if probe else generate_batch(model,tok,[r['prompt'] for r in batch],config['completion_cap'])
                paired=[(batch[0],i,result) for i,result in enumerate(outputs)] if probe else [(r,0,result) for r,result in zip(batch,outputs)]
                for row,index,(text,ids) in paired:
                    e=evaluate_prediction(text,row['ground_truth'])
                    dense=score(text,row['ground_truth'],'dense');sparse=score(text,row['ground_truth'],'sparse')
                    record={'id':row['id'],'source':row['source'],'candidate_index':index,'call_count':row['call_count'],
                        'prediction':text,'reference':row['ground_truth'],'token_ids':ids,'generated_tokens':len(ids),
                        'format_ok':dense['format']==1,'deployable_exact':e['call_multiset_exact'] and dense['format']==1,
                        'truncated':len(ids)==config['completion_cap'] and '</tool_call>' not in text,
                        'hit_generation_limit':len(ids)==config['completion_cap'],'dense_reward':dense,'sparse_reward':sparse,
                        'diagnostics':diagnostics(text,row['ground_truth']),**e}
                    append_jsonl(out/'predictions.jsonl',record);predictions.append(record)
                print(f'eval {split} {len(predictions)}/{len(records)*(4 if probe else 1)}',flush=True)
        report=metrics(predictions)
        if probe:
            import numpy as np
            report['reward_diagnostics']={}
            for kind in ('dense','sparse'):
                groups=[predictions[i:i+4] for i in range(0,len(predictions),4)]
                scores=[[r[f'{kind}_reward']['total'] for r in g] for g in groups]
                report['reward_diagnostics'][kind]={'groups':len(groups),'nonzero_std_fraction':float(np.mean([np.std(x,ddof=1)>1e-8 for x in scores])),
                    'best_of_4_CEM':sum(g[max(range(4),key=lambda i:g[i][f'{kind}_reward']['total'])]['call_multiset_exact'] for g in groups)/len(groups)}
        write_json(out/'metrics.json',report);write_json(out/'resources.json',monitor.summary)
        write_json(out/'status.json',{'status':'complete'})
        return report
    except Exception as exc:
        write_json(out/'status.json',{'status':'failed','error':repr(exc)});raise
    finally:
        del model;clear()
