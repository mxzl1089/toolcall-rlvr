"""Read-only frozen data preparation and deterministic exposure schedules."""
import hashlib
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
from toolrl.verify_split import verify,sha256_file
from toolrl.eval_strict import parse_tool_calls

MANIFEST_SHA='56c16d33022b2dfbcfd3fba0fc9fb92960ff647940efd861741d6404def61d00'

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def append_jsonl(path,row):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def read_jsonl(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def tokenized(record,tokenizer):
    p=tokenizer.apply_chat_template(record['prompt'],tokenize=True,add_generation_prompt=True)
    whole=tokenizer.apply_chat_template(record['prompt']+[{'role':'assistant','content':record['ground_truth']}],tokenize=True,add_generation_prompt=False)
    if whole[:len(p)]!=p:raise ValueError(f'Prompt boundary mismatch {record["id"]}')
    return {'input_ids':whole,'attention_mask':[1]*len(whole),'labels':[-100]*len(p)+whole[len(p):]}

def schedule(records,seed,count):
    if count>len(records):raise ValueError('exposure schedule exceeds frozen pool')
    ordered=sorted(records,key=lambda r:r['id'])
    random.Random(seed).shuffle(ordered)
    return ordered[:count]

def prepare(config,out):
    from transformers import AutoTokenizer
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    manifest=Path(config['manifest']); source=Path(config['source_dir'])
    assert sha256_file(manifest)==config['manifest_sha256'],'Frozen manifest changed'
    verified=verify(source/'train.parquet',source/'test.parquet',manifest,manifest.with_name('prospective_manifest_v2.json'))
    assert verified['passed'],verified['failures']
    m=read_json(manifest)
    assert sha256_file(Path(config['model'])/'tokenizer_config.json')==m['source']['tokenizer_config_sha256']
    tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    rows=pq.read_table(source/'train.parquet').to_pylist()
    lookup={int(r['extra_info']['index']):r for r in rows}
    official={int(r['extra_info']['index']):r for r in pq.read_table(source/'test.parquet').to_pylist()}
    out.mkdir(parents=True)
    summary={}
    for split in ('train','val','test','official'):
        ids=m['official_short_tool_test_indices'] if split=='official' else m['indices'][split]
        assert len(ids)==len(set(ids))
        lengths=[]; records=[]
        for i in ids:
            r=(official if split=='official' else lookup)[i]
            calls=parse_tool_calls(r['reward_model']['ground_truth'])
            record={'id':i,'source':'official' if split=='official' else 'train','prompt':r['prompt'],'ground_truth':r['reward_model']['ground_truth'],'call_count':len(calls)}
            tokens=tokenized(record,tok)
            p=sum(x==-100 for x in tokens['labels'])
            assert p<=768 and len(tokens['input_ids'])<=config['max_length'],(split,i,p,len(tokens['input_ids']))
            assert any(x!=-100 for x in tokens['labels'])
            record.update(prompt_tokens=p,full_tokens=len(tokens['input_ids']))
            records.append(record);lengths.append((i,p,len(tokens['input_ids'])))
            append_jsonl(out/f'{split}.jsonl',record)
        summary[split]={'count':len(records),'single':sum(r['call_count']==1 for r in records),'max_prompt':max(x[1] for x in lengths),'max_full':max(x[2] for x in lengths),'lengths':lengths}
        if split=='train':
            for seed in (42,43,44):write_json(out/f'schedule_{seed}.json',[r['id'] for r in schedule(records,seed,config['exposures'])])
    write_json(out/'verification.json',verified)
    write_json(out/'summary.json',summary)
    write_json(out/'fingerprints.json',{p.name:sha256_file(p) for p in out.glob('*.jsonl')})
    return summary

def load_prepared(folder,split,config=None):
    folder=Path(folder); path=folder/f'{split}.jsonl'
    assert sha256_file(path)==read_json(folder/'fingerprints.json')[path.name]
    rows=read_jsonl(path)
    if config is not None:
        assert sha256_file(Path(config['manifest']))==config['manifest_sha256']
        manifest=read_json(config['manifest'])
        expected=manifest['official_short_tool_test_indices'] if split=='official' else manifest['indices'][split]
        assert [r['id'] for r in rows]==expected,'Prepared records do not match the frozen manifest'
    return rows
