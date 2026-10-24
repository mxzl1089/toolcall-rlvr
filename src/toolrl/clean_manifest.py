"""Authorized v3: deterministic schema-contract exclusions, never relabel or resplit."""
import ast
from datetime import datetime
from pathlib import Path
from toolrl.audit_schema import schemas
from toolrl.data import read_json,write_json,load_prepared,MANIFEST_SHA
from toolrl.verify_split import sha256_file
from toolrl.eval_strict import parse_tool_calls

RULES={
 'version':'schema-valid-v1',
 'unknown_parameter':'exclude sample if any labelled parameter key is absent from its tool schema',
 'primitive_types':'str/string=>str; int/integer=>int excluding bool; float/number=>int or float excluding bool; bool/boolean=>bool; dict/object=>dict; list/array=>list',
 'containers':'List[T] recursively validates elements; Tuple[T1,T2] uses fixed-length JSON list; Dict is dict; Union accepts any member',
 'optional':'comma optional denotes omissible field, not nullable value; no required-key inference from defaults',
 'unsupported':'exclude samples with used parameter types not safely representable/recognized (e.g. Callable)',
 'preserve':'never coerce numeric strings, decode JSON-in-string, repair labels, randomize indices or move samples between splits',
 'model_independence':'all rules fixed without using prediction correctness; v2 diagnostic val outputs are not inputs',
}

def type_matches(value,type_text):
    text=type_text.strip().replace(', optional','').strip()
    try:tree=ast.parse(text,mode='eval').body
    except SyntaxError:return None
    def check(v,node):
        if isinstance(node,ast.Name):
            name=node.id.lower()
            primitives={'str':str,'string':str,'int':int,'integer':int,'bool':bool,'boolean':bool,'dict':dict,'object':dict,'list':list,'array':list}
            if name in primitives:return type(v)==primitives[name]
            if name in ('float','number'):return type(v) in (int,float)
            return None
        if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name):
            name=node.value.id.lower();sub=node.slice
            parts=sub.elts if isinstance(sub,ast.Tuple) else [sub]
            if name=='union':
                values=[check(v,n) for n in parts]
                return True if True in values else None if None in values else False
            if name=='list' and len(parts)==1:
                if type(v)!=list:return False
                values=[check(x,parts[0]) for x in v]
                return None if None in values else all(values)
            if name=='tuple':
                if type(v)!=list or len(v)!=len(parts):return False
                values=[check(x,n) for x,n in zip(v,parts)]
                return None if None in values else all(values)
        return None
    return check(value,tree)

def violations(record):
    tools,unparsed=schemas(record['prompt'][0]['content']);errors=[]
    for call in parse_tool_calls(record['ground_truth']):
        if call['name'] not in tools:
            errors.append({'code':'unparsed_schema','tool':call['name']});continue
        for k,v in call['parameters'].items():
            if k not in tools[call['name']]:errors.append({'code':'unknown_parameter','tool':call['name'],'parameter':k});continue
            spec=tools[call['name']][k]
            t=spec.get('type','') if isinstance(spec,dict) else ''
            match=type_matches(v,t)
            if match is not True:errors.append({'code':'unsupported_type' if match is None else 'type_conflict','tool':call['name'],'parameter':k,'expected':t,'actual':type(v).__name__})
    return errors

def create(parent,prepared,out):
    parent=Path(parent);out=Path(out)
    assert sha256_file(parent)==MANIFEST_SHA
    if out.exists():raise FileExistsError(out)
    v2=read_json(parent)
    v3={**v2,'version':'v3','parent_version':'v2','parent_sha256':sha256_file(parent),
        'created_at':datetime.now().astimezone().isoformat(),'cleaning_rules':RULES,
        'cleaning_authorized_by_user':True,'not_a_random_resplit':True,'indices':{},'removed_from_v2':{},'counts':{}}
    for split in ('train','val','test','official'):
        records=load_prepared(prepared,split);kept=[];removed=[]
        for r in records:
            errs=violations(r)
            if errs:removed.append({'id':r['id'],'reasons':errs})
            else:kept.append(r['id'])
        if split=='official':v3['official_short_tool_test_indices']=kept
        else:v3['indices'][split]=kept
        v3['removed_from_v2'][split]=removed
        accepted=[r for r in records if r['id'] in set(kept)]
        v3['counts'][split]=len(kept)
        v3.setdefault('strata',{})[split]={'single':sum(r['call_count']==1 for r in accepted),'multi':sum(r['call_count']>1 for r in accepted)}
    # The old removed_ids are retained as v1->v2 history, not reapplied to v3.
    write_json(out,v3)
    print({'counts':v3['counts'],'strata':v3['strata'],'sha256':sha256_file(out)})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();create(a.parent,a.prepared,a.out)
