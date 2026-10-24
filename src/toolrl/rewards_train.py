"""Contract-aligned sparse/dense rewards, adapted from ToolRL's matching idea.

Original reference: ToolRL rlla.py, Apache-2.0, ByteDance 2024.
Changes: typed equality, unique-block parsing and optimal unordered assignment.
Evaluation is independently implemented in eval_strict.py.
"""
import json
import math
import re
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment


def typed(value):
    if isinstance(value,dict): return ('dict',tuple(sorted((k,typed(v)) for k,v in value.items())))
    if isinstance(value,list): return ('list',tuple(map(typed,value)))
    return (type(value).__name__,value)


def _pairs(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError('duplicate_json_key')
        out[k]=v
    return out


def _bad(x): raise ValueError('invalid_json_constant')


def _float(x):
    v=float(x)
    return v if math.isfinite(v) else _bad(x)


def parse(text):
    if not isinstance(text,str):raise ValueError('non_string_output')
    if text.count('<tool_call>')!=1 or text.count('</tool_call>')!=1:
        raise ValueError('invalid_tool_call_count')
    a=text.index('<tool_call>')+len('<tool_call>'); b=text.index('</tool_call>')
    if b<a: raise ValueError('invalid_tool_call_order')
    body=text[a:b].strip()
    if not body:raise ValueError('empty_tool_call')
    calls=[]
    for line in body.splitlines():
        c=json.loads(line,object_pairs_hook=_pairs,parse_constant=_bad,parse_float=_float)
        if type(c)!=dict or set(c)!={'name','parameters'} or type(c['name'])!=str or type(c['parameters'])!=dict:
            raise ValueError('invalid_call_schema')
        calls.append(c)
    return calls


def format_score(text, reference):
    # Default tool-call-only branch of the upstream regex; these data have no response target.
    if '<response>' in reference:raise ValueError('response labels are outside this protocol')
    return float(bool(re.fullmatch(r'<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>',text.strip(),re.S)))


def jaccard(a,b):
    a,b=Counter(list(a)),Counter(list(b))
    denom=sum((a|b).values())
    return sum((a&b).values())/denom if denom else 1.0


def matching(g,p):
    weights=np.zeros((len(g),len(p)),dtype=float)
    for i,x in enumerate(g):
        for j,y in enumerate(p):
            if x['name']!=y['name']:continue
            a,b=x['parameters'],y['parameters']
            weights[i,j]=jaccard(a,b)+sum(k in b and typed(v)==typed(b[k]) for k,v in a.items())
    rows,cols=linear_sum_assignment(weights,maximize=True)
    pairs=[(int(i),int(j)) for i,j in zip(rows,cols) if g[i]['name']==p[j]['name']]
    return float(sum(weights[i,j] for i,j in pairs)),pairs


def score(prediction,reference,mode):
    if mode not in ('sparse','dense'):raise ValueError(f'Unknown reward mode {mode}')
    g=parse(reference)
    try:p=parse(prediction)
    except (ValueError,TypeError,RecursionError) as e:
        return {'format':0.0,'correctness':-3.0,'total':-3.0,'error':str(e),'pairs':[]}
    exact=Counter(map(typed,g))==Counter(map(typed,p))
    pairs=[]
    if mode=='sparse':c=3.0 if exact else -3.0
    else:
        param,pairs=matching(g,p)
        names=jaccard([x['name'] for x in g],[x['name'] for x in p])
        denom=1+sum(1+len(x['parameters']) for x in g)
        c=-3+6*(names+param)/denom
    f=format_score(prediction,reference)
    return {'format':f,'correctness':c,'total':f+c,'error':None,'pairs':pairs}
