"""Independent diagnostic metrics; not used to construct training rewards."""
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from toolrl.eval_strict import parse_tool_calls,ToolCallParseError,_typed_json

def diagnostics(prediction,reference):
    g=parse_tool_calls(reference)
    zero={'tool_names_exact':False,'extra_call':False,'duplicate_call':False,
          'key_tp':0,'key_fp':0,'key_fn':sum(len(x['parameters']) for x in g),
          'value_correct':0,'value_total':sum(len(x['parameters']) for x in g)}
    try:p=parse_tool_calls(prediction)
    except ToolCallParseError as e:return {**zero,'error_types':[e.code]}
    gc=Counter(x['name'] for x in g);pc=Counter(x['name'] for x in p)
    duplicate=any(n>Counter(_typed_json(x) for x in g)[k] and n>1 for k,n in Counter(_typed_json(x) for x in p).items())
    result={**zero,'tool_names_exact':gc==pc,'extra_call':len(p)>len(g),'duplicate_call':duplicate,'key_fn':0}
    errors=[]
    if gc!=pc:errors.append('wrong_tool_multiset')
    if len(p)>len(g):errors.append('extra_call')
    if len(p)<len(g):errors.append('missing_call')
    if duplicate:errors.append('duplicate_call')
    w=np.zeros((len(g),len(p)))
    for i,x in enumerate(g):
        for j,y in enumerate(p):
            if x['name']==y['name']:
                a,b=x['parameters'],y['parameters']
                w[i,j]=len(set(a)&set(b))+sum(k in b and _typed_json(v)==_typed_json(b[k]) for k,v in a.items())+1
    ii,jj=linear_sum_assignment(w,maximize=True)
    pairs=[(i,j) for i,j in zip(ii,jj) if g[i]['name']==p[j]['name']]
    usedg={i for i,j in pairs};usedp={j for i,j in pairs}
    for i,j in pairs:
        a,b=g[i]['parameters'],p[j]['parameters']
        result['key_tp']+=len(set(a)&set(b));result['key_fp']+=len(set(b)-set(a));result['key_fn']+=len(set(a)-set(b))
        if set(a)-set(b):errors.append('missing_parameter')
        if set(b)-set(a):errors.append('extra_parameter')
        for k in set(a)&set(b):
            if _typed_json(a[k])==_typed_json(b[k]):result['value_correct']+=1
            elif type(a[k])!=type(b[k]):errors.append('wrong_value_type')
            else:errors.append('wrong_value')
    result['key_fn']+=sum(len(x['parameters']) for i,x in enumerate(g) if i not in usedg)
    result['key_fp']+=sum(len(x['parameters']) for j,x in enumerate(p) if j not in usedp)
    result['error_types']=sorted(set(errors))
    return result
