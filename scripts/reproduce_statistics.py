"""Recompute main accuracy, error counts and paired CI from published per-prediction records."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def analyze(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    methods=['continue','sparse','dense'];seeds=[42,43,44]
    lookup={(r['method'],int(r['seed']),int(r['id'])):r for r in rows}
    assert len(lookup)==len(rows)==159*3*3, 'Incomplete or duplicated predictions'
    ids=sorted({int(r['id']) for r in rows});assert len(ids)==159
    for m in methods:
        for s in seeds:assert all((m,s,i) in lookup for i in ids)
    correct=lambda r: int(r['strict'].lower()=='true')
    accuracy={m:{'correct':sum(correct(r) for r in rows if r['method']==m),'n':477} for m in methods}
    errors={}
    for m in methods:
        layer={}
        for r in rows:
            if r['method']==m and not correct(r):
                name=r['failure_layer'];layer[name]=layer.get(name,0)+1
        errors[m]=layer
    differences=np.mean([[correct(lookup['dense',s,i])-correct(lookup['sparse',s,i]) for i in ids] for s in seeds],axis=0)
    rng=np.random.default_rng(20261013);draw=np.zeros(10000)
    for single in [True,False]:
        indices=[j for j,i in enumerate(ids) if (int(lookup['dense',42,i]['reference_calls'])==1)==single]
        for i in ids:
            assert len({lookup[m,s,i]['reference_calls'] for m in methods for s in seeds})==1
        draw+=differences[rng.choice(indices,size=(10000,len(indices)),replace=True)].sum(axis=1)
    draw/=len(ids)
    return {'accuracy':accuracy,'first_failure_counts':errors,'dense_minus_sparse_pp':float(differences.mean()*100),'paired_bootstrap95_pp':(np.quantile(draw,[.025,.975])*100).tolist(),'scope':'test-question uncertainty conditional on three trained seeds'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='results/per_prediction.csv');p.add_argument('--out');a=p.parse_args()
    result=analyze(a.input);text=json.dumps(result,indent=2)
    print(text)
    if a.out:
        path=Path(a.out)
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text+'\n',encoding='utf-8')
