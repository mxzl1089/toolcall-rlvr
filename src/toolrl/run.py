"""Explicit stage CLI. Default operations never train or access held-out predictions."""
import argparse
import os
from pathlib import Path
from toolrl.data import read_json,write_json,prepare

def main():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/protocol.json')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--out',required=True)
    a=sub.add_parser('train');a.add_argument('--prepared',required=True);a.add_argument('--out',required=True)
    a.add_argument('--method',choices=['sft','continue','sparse','dense'],required=True);a.add_argument('--seed',type=int,default=42)
    a.add_argument('--steps',type=int,required=True);a.add_argument('--horizon',type=int);a.add_argument('--adapter')
    a.add_argument('--longest',action='store_true');a.add_argument('--force-length',action='store_true')
    a=sub.add_parser('eval');a.add_argument('--prepared',required=True);a.add_argument('--out',required=True)
    a.add_argument('--adapter');a.add_argument('--split',choices=['val','test','official'],default='val')
    a.add_argument('--limit',type=int);a.add_argument('--lock');a.add_argument('--probe',action='store_true')
    args=p.parse_args();c=read_json(args.config)
    assert c['group_size']==4 and c['beta']==0 and c['loss_type']=='grpo'
    assert c['completion_cap'] in (128,160) and c['max_length']==960
    if args.command=='prepare':prepare(c,args.out)
    elif args.command=='train':
        from toolrl.train import train
        train(c,args.prepared,args.out,args.method,args.seed,args.steps,args.adapter,args.horizon,args.longest,args.force_length)
    else:
        from toolrl.evaluate import evaluate
        evaluate(c,args.prepared,args.out,args.adapter,args.split,args.limit,args.lock,args.probe)

if __name__=='__main__':main()
