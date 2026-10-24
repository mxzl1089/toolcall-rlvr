"""Read-only audit for label parameters missing from supplied tool schemas."""
import json,re
from pathlib import Path
from toolrl.data import load_prepared,write_json
from toolrl.eval_strict import parse_tool_calls

def schemas(instruction):
    found=list(re.finditer(r'^\d+\.\s+Name:\s*(.+)$',instruction,re.M))
    result={};unparsed=[]
    for i,m in enumerate(found):
        section=instruction[m.end():found[i+1].start() if i+1<len(found) else len(instruction)]
        match=re.search(r'^Parameters:\s*(\{[^\n]*\})\s*$',section,re.M)
        if not match:unparsed.append(m.group(1).strip());continue
        try:result[m.group(1).strip()]=json.loads(match.group(1))
        except ValueError:unparsed.append(m.group(1).strip())
    return result,unparsed

def audit(prepared,out):
    report={}
    for split in ('train','val','test','official'):
        bad=[];unknown=[];ok=[]
        for r in load_prepared(prepared,split):
            tools,unparsed=schemas(r['prompt'][0]['content'])
            reasons=[]
            for call in parse_tool_calls(r['ground_truth']):
                if call['name'] not in tools:
                    unknown.append({'id':r['id'],'tool':call['name'],'unparsed':unparsed});continue
                keys=set(call['parameters'])-set(tools[call['name']])
                if keys:reasons.append({'tool':call['name'],'unexpected_keys':sorted(keys),'schema_keys':sorted(tools[call['name']])})
            if reasons:bad.append({'id':r['id'],'issues':reasons,'label':r['ground_truth']})
            else:ok.append(r['id'])
        report[split]={'samples':len(load_prepared(prepared,split)),'invalid_keys_count':len(bad),'invalid_key_ids':[r['id'] for r in bad],
                       'unparsed_count':len(unknown),'unparsed':unknown,'bad_records':bad,'no_key_violation_ids':ok}
    write_json(out,report)
    print(json.dumps({k:{x:v[x] for x in ('samples','invalid_keys_count','invalid_key_ids','unparsed_count')} for k,v in report.items()},ensure_ascii=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);a=p.parse_args();audit(a.prepared,a.out)
