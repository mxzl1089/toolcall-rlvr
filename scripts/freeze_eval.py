"""Freeze a model/config pair before full held-out evaluation."""
import argparse
from toolrl.data import read_json, write_json
from toolrl.runtime import fingerprint_adapter
from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/protocol.json');p.add_argument('--adapter',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    if Path(a.out).exists():raise FileExistsError(a.out)
    hashes=fingerprint_adapter(a.adapter)
    if not hashes:raise ValueError('No adapter files found')
    write_json(a.out,{'config':read_json(a.config),'adapters':[hashes]})
