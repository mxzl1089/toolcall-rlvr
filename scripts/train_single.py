"""Single continuous dense-GRPO trajectory, full-state checkpoints and audited resume."""
from common import *
import argparse
import math
import shutil
import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback, set_seed
from trl import GRPOTrainer, GRPOConfig
from toolrl.data import load_prepared, append_jsonl
from toolrl.runtime import load_model, tokenizer, Monitor, fingerprint_adapter
from toolrl.rewards_train import score
from toolrl.eval_strict import evaluate_prediction
from toolrl.metrics import diagnostics


class AuditedTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        actual = result['advantages'].detach().float().cpu().numpy()
        assert len(actual)==4 and np.allclose(actual, self.expected, atol=2e-5, rtol=2e-5)
        append_jsonl(self.audit_out/'advantage_checks.jsonl', {'group':self.group, 'actual_advantages':actual.tolist(), 'max_error':float(np.max(np.abs(actual-self.expected)))})
        return result


def trim_to_checkpoint(out, step):
    stamp = str(time.time_ns())
    archive = out/'recovery'/stamp
    for name, key, exclusive in [('rollouts.jsonl','group_id',True), ('advantage_checks.jsonl','group',True), ('groups.jsonl','group',True), ('updates.jsonl','step',False), ('trainer_log.jsonl','step',False), ('gradients.jsonl','step',False)]:
        p = out/name
        if not p.exists():
            continue
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p,archive/name)
        kept=[]
        # A process killed during append may leave exactly one partial trailing line.
        lines=p.read_text(encoding='utf-8').splitlines()
        for i,line in enumerate(lines):
            try: r=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1: raise
                continue
            if r[key]<step if exclusive else r[key]<=step:
                kept.append(line)
        p.write_text('\n'.join(kept)+ ('\n' if kept else ''),encoding='utf-8')
    # Preserve uncommitted checkpoint directories before Trainer reuses their names.
    for p in (out/'checkpoints').glob('checkpoint-*'):
        if int(p.name.split('-')[-1])>step:
            archive.mkdir(parents=True, exist_ok=True)
            p.rename(archive/p.name)
    return str(archive)


class Audit(TrainerCallback):
    def __init__(self,out,requested,marks,group_counter,writer):
        self.out=out; self.requested=requested; self.marks=marks; self.counter=group_counter; self.writer=writer; self.paused=False
    def on_train_begin(self,args,state,control,model=None,**kw):
        self.prev={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.start=time.perf_counter()
    def on_pre_optimizer_step(self,args,state,control,model=None,**kw):
        squared=0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), 'Nonfinite gradient'
                squared+=float(p.grad.detach().float().square().sum())
        norm=math.sqrt(squared)
        append_jsonl(self.out/'gradients.jsonl',{'step':state.global_step+1,'post_clip_grad_norm':norm})
        self.writer.add_scalar('audit/post_clip_grad_norm',norm,state.global_step+1)
    def on_step_end(self,args,state,control,model=None,**kw):
        current={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        delta=sum((current[n]-p).abs().sum().item() for n,p in self.prev.items())
        assert math.isfinite(delta)
        append_jsonl(self.out/'updates.jsonl',{'step':state.global_step,'lora_delta_l1':delta,'elapsed_segment_sec':time.perf_counter()-self.start})
        self.writer.add_scalar('audit/lora_delta_l1',delta,state.global_step)
        self.prev=current
        assert self.counter[0]==state.global_step, (self.counter[0],state.global_step)
        if state.global_step in self.marks or state.global_step>=self.requested:
            control.should_save=True
        if (ROOT/'STOP').exists():
            control.should_save=True; control.should_training_stop=True; self.paused=True
        if state.global_step>=self.requested:
            control.should_training_stop=True
    def on_save(self,args,state,control,**kw):
        folder=Path(args.output_dir)/f'checkpoint-{state.global_step}'
        needed=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json']
        assert all((folder/n).is_file() for n in needed), 'Incomplete full-state checkpoint'
        save(folder/'committed.json',{'step':state.global_step,'groups':self.counter[0], 'files':{n:sha(folder/n) for n in needed}})
        self.writer.flush()
    def on_log(self,args,state,control,logs=None,**kw):
        if logs:
            for k,v in logs.items():
                if isinstance(v,(int,float)):
                    assert math.isfinite(v), (k,v)
                    self.writer.add_scalar('trainer/'+k,v,state.global_step)
            append_jsonl(self.out/'trainer_log.jsonl',{'step':state.global_step,**logs})


def main():
    a=argparse.ArgumentParser(); a.add_argument('--preflight',action='store_true'); a.add_argument('--stop-at',type=int)
    args=a.parse_args(); freeze_check(); p=protocol(); c=p['config']; seed=p['seed']
    from torch.utils.tensorboard import SummaryWriter
    train=load_prepared(p['prepared'],'train',c); lookup={r['id']:r for r in train}
    total=4 if args.preflight else p['steps']
    ids=[r['id'] for r in sorted(train,key=lambda r:(r['prompt_tokens'],r['full_tokens'],r['id']),reverse=True)[:4]] if args.preflight else read(ROOT/'config/schedule.json')
    records=[lookup[i] for i in ids]; assert len(records)==total
    stop=args.stop_at or total
    out=ROOT/'artifacts'/('preflight' if args.preflight else 'train'); out.mkdir(parents=True,exist_ok=True)
    checkpoint=latest_checkpoint(out/'checkpoints')
    step=read(checkpoint/'committed.json')['step'] if checkpoint else 0
    if step>=stop:
        raise RuntimeError('Requested stop already reached; supervisor should reuse completed artifacts')
    if checkpoint:
        archive=trim_to_checkpoint(out,step)
        append_jsonl(out/'resumes.jsonl',{'from_step':step,'checkpoint':str(checkpoint),'archived':archive})
    elif (out/'rollouts.jsonl').exists():
        raise RuntimeError('Partial training without committed checkpoint: manual review required')
    set_seed(seed); tok=tokenizer(c); model=load_model(c,p['adapter'],True)
    initial={n:v.detach().float().cpu().clone() for n,v in model.named_parameters() if v.requires_grad}
    count=[step]
    writer=SummaryWriter(str(ROOT/'tensorboard'/('preflight' if args.preflight else 'train')),purge_step=step+1 if step else None)
    trainer=None
    def reward(completions,ground_truth,sample_id,completion_ids,**kwargs):
        assert len(completions)==4 and len(set(sample_id))==1
        group=count[0]
        assert group<len(records) and sample_id[0]==records[group]['id'], ('schedule drift',group,sample_id)
        texts=[x[0]['content'] if isinstance(x,list) else x for x in completions]
        scores=[score(t,g,'dense') for t,g in zip(texts,ground_truth)]
        values=np.array([r['total'] for r in scores],dtype=np.float32)
        mean=float(values.mean()); std=float(values.std(ddof=1)); advantage=(values-mean)/(std+1e-4)
        trainer.expected=advantage; trainer.group=group
        results=[evaluate_prediction(t,g) for t,g in zip(texts,ground_truth)]
        correct=sum(r['call_multiset_exact'] for r in results)
        wrong_positive=sum(a>0 and not e['call_multiset_exact'] for a,e in zip(advantage,results))
        for i,(t,g,s,e) in enumerate(zip(texts,ground_truth,scores,results)):
            append_jsonl(out/'rollouts.jsonl',{'group_id':group,'epoch_index':group//973,'id':int(sample_id[i]),'candidate_index':i,
                'prediction':t,'reference':g,'reward':s,'evaluation':e,'diagnostics':diagnostics(t,g),
                'group_mean':mean,'group_std':std,'advantage':float(advantage[i]),
                'generated_tokens':len(completion_ids[i]),'token_ids':completion_ids[i],
                'hit_generation_limit':len(completion_ids[i])>=c['completion_cap']})
        append_jsonl(out/'groups.jsonl',{'group':group,'id':int(sample_id[0]),'strict_correct_candidates':correct,'reward_mean':mean,'reward_std':std,'wrong_positive':int(wrong_positive)})
        for k,v in {'correct_candidates':correct,'reward_mean':mean,'nonzero_advantage_group':int(std>1e-8),'wrong_positive_candidates':wrong_positive}.items():
            writer.add_scalar('rollout/'+k,v,group+1)
        count[0]+=1
        return values.tolist()
    cb=Audit(out,stop,set(p['checkpoints'])|{2,4} if args.preflight else set(p['checkpoints']),count,writer)
    conf=GRPOConfig(output_dir=str(out/'checkpoints'),max_steps=total,per_device_train_batch_size=1,
        gradient_accumulation_steps=4,num_generations=4,steps_per_generation=4,
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        report_to='none',save_strategy='steps',save_steps=100,save_total_limit=None,logging_steps=1,
        optim='adamw_torch',weight_decay=0.0,lr_scheduler_type='linear',warmup_steps=0,max_grad_norm=1.0,
        dataloader_pin_memory=False,dataloader_num_workers=0,disable_tqdm=True,seed=seed,data_seed=seed,
        torch_empty_cache_steps=1,max_completion_length=c['completion_cap'],learning_rate=c['rl_lr'],
        temperature=c['temperature'],top_p=c['top_p'],top_k=None,shuffle_dataset=False,use_vllm=False,
        beta=0.0,loss_type='grpo',scale_rewards='group',epsilon=.2,num_iterations=1,mask_truncated_completions=False,
        generation_kwargs={'min_new_tokens':c['completion_cap']} if args.preflight else None)
    ds=Dataset.from_list([{'prompt':r['prompt'],'ground_truth':r['ground_truth'],'sample_id':r['id']} for r in records])
    trainer=AuditedTrainer(model=model,args=conf,processing_class=tok,train_dataset=ds,reward_funcs=reward,callbacks=[cb]); trainer.audit_out=out
    for n,v in trainer.model.named_parameters():
        if n in initial:
            v.data=initial[n].to(device=v.device,dtype=torch.float32)
            assert torch.equal(v.detach().cpu(),initial[n])
    if not (out/'run.json').exists():
        save(out/'run.json',{'ids':ids,'seed':seed,'total_steps':total,'common_sft_hashes':fingerprint_adapter(p['adapter']),
            'exact_common_FP32_restore':True,'trainable_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad),'preflight':args.preflight})
    save(out/'status.json',{'status':'training','resume_step':step,'pid':os.getpid()})
    try:
        with Monitor(out/f'gpu_{step}.jsonl') as monitor:
            trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        end=trainer.state.global_step
        assert end==stop or cb.paused
        assert checkpoint_valid(out/'checkpoints'/f'checkpoint-{end}')
        save(out/f'resources_{step}_{end}.json',monitor.summary)
        save(out/'status.json',{'status':'paused' if cb.paused else ('complete' if end==total else 'preflight_segment_complete'),'steps':end,'requested_steps':total})
        print(json.dumps(read(out/'status.json')),flush=True)
    except Exception as exc:
        save(out/'status.json',{'status':'failed','error':repr(exc)}); raise
    finally:
        writer.close()


if __name__=='__main__': main()
