"""Run one controlled advantage-intervention branch."""
import argparse
from toolrl.data import read_json
from toolrl_mechanism.train_mechanism import train
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/protocol.json')
    p.add_argument('--prepared',required=True);p.add_argument('--adapter',required=True);p.add_argument('--out',required=True)
    p.add_argument('--method',choices=['identity','gate','random'],required=True)
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    a=p.parse_args();train(read_json(a.config),a.prepared,a.out,a.method,a.seed,200,a.adapter)
