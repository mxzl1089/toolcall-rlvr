"""Rescore unchanged rollout text under four reward configurations."""
import argparse
from collections import defaultdict
import numpy as np
from toolrl.data import read_jsonl,write_json
from toolrl.rewards_train import score
from toolrl.eval_strict import evaluate_prediction
from pathlib import Path
def summarize(paths):
    groups=defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):groups[(str(path),row['group_id'])].append(row)
    stats={}
    for mode in ['sparse','dense']:
        for include_format in [True,False]:
            d={'groups':len(groups),'candidates':0,'nonzero_advantage_groups':0,'positive_candidates':0,'strict_wrong_positive':0,'all_wrong_groups':0,'all_wrong_nonzero_groups':0}
            for rows in groups.values():
                assert len(rows)==4 and len({r['candidate_index'] for r in rows})==4
                assert len({r['id'] for r in rows})==1
                rewards=[score(r['prediction'],r['reference'],mode) for r in rows]
                values=np.asarray([r['total'] if include_format else r['correctness'] for r in rewards],dtype=np.float32)
                adv=(values-values.mean())/(values.std(ddof=1)+1e-4)
                wrong=np.array([not evaluate_prediction(r['prediction'],r['reference'])['call_multiset_exact'] for r in rows])
                nz=bool(np.any(adv!=0));d['candidates']+=4;d['nonzero_advantage_groups']+=int(nz)
                d['positive_candidates']+=int((adv>0).sum());d['strict_wrong_positive']+=int(((adv>0)&wrong).sum())
                d['all_wrong_groups']+=int(wrong.all());d['all_wrong_nonzero_groups']+=int(wrong.all() and nz)
            stats[f"{mode}_{'with_format' if include_format else 'correctness_only'}"]=d
    return stats
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('rollouts',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
    if Path(a.out).exists():raise FileExistsError(a.out)
    write_json(a.out,summarize(a.rollouts))
