"""Prepare a new frozen 973-example, two-epoch continuous run."""
from common import *
import argparse
from toolrl.data import load_prepared
from toolrl.runtime import fingerprint_adapter
if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--config',default='configs/protocol.json');a.add_argument('--prepared',required=True);a.add_argument('--adapter',required=True)
    args=a.parse_args();c=read(args.config)
    config_dir=ROOT/'config'
    if (config_dir/'freeze.json').exists():raise FileExistsError('Existing frozen extension; use another checkout for a new run')
    data=load_prepared(args.prepared,'train',c);assert len(data)==973
    adapter=Path(args.adapter).resolve();prepared=Path(args.prepared).resolve()
    assert fingerprint_adapter(adapter),'No common SFT adapter'
    c.update(exposures=1946,seeds=[20261022])
    p={'config':c,'seed':20261022,'steps':1946,'epochs':2,'checkpoints':[200,500,973,1450,1946],'prepared':str(prepared),'adapter':str(adapter)}
    save(config_dir/'protocol.json',p)
    save(config_dir/'schedule.json',exposure_order([r['id'] for r in data],p['seed'],2))
    paths=[config_dir/'protocol.json',config_dir/'schedule.json',Path(c['manifest']),Path(c['model'])/'tokenizer_config.json']
    paths+=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    paths+=list(prepared.glob('*.jsonl'))+[prepared/'fingerprints.json']
    paths+=list(adapter.glob('*.safetensors'))+list(adapter.glob('*.json'))
    save(config_dir/'freeze.json',{'files':{str(x.resolve()):sha(x) for x in paths}})
    print('Frozen 1946-request run prepared. Run preflight, then continuous training from the repository root.')
