"""Isolated GRPO mechanism interventions; the frozen trainer is never edited."""
import json
import math
import time
import hashlib
import subprocess
import psutil
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback,set_seed
from trl import GRPOTrainer,GRPOConfig
from toolrl.data import load_prepared,schedule,write_json,append_jsonl
from toolrl.runtime import load_model,tokenizer,Monitor,init_run,fingerprint_adapter,clear
from toolrl.rewards_train import score
from toolrl.eval_strict import evaluate_prediction
from toolrl.verify_split import sha256_file


RANDOM_MASK_PROBABILITY = 189 / 346


def random_u(seed, group, sample_id):
    payload = f'mechanism-v1|random-mask|{seed}|{group}|{sample_id}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') / 2**64


class AuditedGRPO(GRPOTrainer):
    def _generate_and_score_completions(self,inputs):
        result=super()._generate_and_score_completions(inputs)
        actual=result['advantages'].detach().float().cpu().numpy()
        if not np.allclose(actual,self.audit_expected,atol=2e-5,rtol=2e-5):
            raise RuntimeError('Logged and actual TRL advantages disagree')
        if len(actual)!=4 or self.audit_mask not in (0, 1):
            raise RuntimeError('Broken mechanism group/mask contract')
        applied=actual*self.audit_mask
        if self.audit_mask==0 and np.any(applied):
            raise RuntimeError('Masked advantages remain nonzero')
        append_jsonl(self.audit_out/'advantage_checks.jsonl',{'group':self.audit_group,
            'actual_advantages':actual.tolist(),'applied_advantages':applied.tolist(),
            'mask':self.audit_mask,'max_error':float(np.max(np.abs(actual-self.audit_expected)))})
        result['advantages']=result['advantages']*self.audit_mask
        return result


class AuditCallback(TrainerCallback):
    def __init__(self,out,stop_after):self.out=out;self.stop_after=stop_after;self.prev=None;self.started=0;self.zero_groups=0;self.stop_reason=None;self.low_ram_since=None;self.high_gpu_since=None
    def on_train_begin(self,args,state,control,model=None,**kw):
        self.prev={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.started=time.perf_counter()
    def on_step_end(self,args,state,control,model=None,**kw):
        current={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        delta=sum((current[n]-v).abs().sum().item() for n,v in self.prev.items())
        if not math.isfinite(delta):raise RuntimeError('Nonfinite parameter update')
        append_jsonl(self.out/'updates.jsonl',{'step':state.global_step,'lora_delta_l1':delta,'elapsed':time.perf_counter()-self.started})
        ram=psutil.virtual_memory().available/2**30
        gpu_used=None
        try:
            gpu_used=int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=4,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout.strip().splitlines()[0])
        except (ValueError,IndexError,subprocess.TimeoutExpired,FileNotFoundError):pass
        clock=time.perf_counter()
        self.low_ram_since=(self.low_ram_since or clock) if ram<2.0 else None
        self.high_gpu_since=(self.high_gpu_since or clock) if gpu_used is not None and gpu_used>3891 else None
        append_jsonl(self.out/'resource_guard.jsonl',{'step':state.global_step,'ram_available_GiB':ram,
            'gpu_used_MiB':gpu_used,'low_ram_since':self.low_ram_since,'high_gpu_since':self.high_gpu_since})
        if (self.low_ram_since is not None and clock-self.low_ram_since>30) or (self.high_gpu_since is not None and clock-self.high_gpu_since>30):
            self.stop_reason='resource_limit'
            control.should_training_stop=True
        self.prev=current
        if self.zero_groups>=40:
            self.stop_reason='stopped_no_signal'
            control.should_training_stop=True
        if state.global_step>=self.stop_after:control.should_training_stop=True
    def on_log(self,args,state,control,logs=None,**kw):
        if logs:
            for key,value in logs.items():
                if isinstance(value,(int,float)) and not math.isfinite(value):raise RuntimeError(f'Nonfinite {key}')
            append_jsonl(self.out/'trainer_log.jsonl',{'step':state.global_step,**logs})


def train(config,prepared,out,method,seed,steps,adapter=None,horizon=None,longest=False,force_length=False):
    if method not in ('identity','gate','random'):raise ValueError(method)
    if adapter is None:raise ValueError('A common SFT adapter is required')
    if longest or force_length:raise ValueError('Mechanism branches must use the frozen ordinary schedule')
    set_seed(seed)
    original=load_prepared(prepared,'train',config)
    ordered=schedule(original,seed,len(original))
    count=steps
    if count>len(ordered):raise ValueError('Requested exposure exceeds frozen pool')
    records=ordered[:count]
    metadata={'method':method,'seed':seed,'steps':steps,'horizon':horizon or steps,'ids':[r['id'] for r in records],
              'initial_adapter':str(adapter) if adapter else None,'initial_adapter_hashes':fingerprint_adapter(adapter),
              'resource_longest':longest,'force_full_completion':force_length,
              'random_mask_probability':RANDOM_MASK_PROBABILITY,
              'protocol':'mechanism-v1'}
    out=init_run(out,config,metadata)
    write_json(out/'mechanism_source_hashes.json',{
        'train_mechanism.py':sha256_file(Path(__file__)),
        'rewards_train.py':sha256_file(Path(score.__code__.co_filename)),
        'eval_strict.py':sha256_file(Path(evaluate_prediction.__code__.co_filename))})
    tok=tokenizer(config);model=None;trainer=None
    try:
        model=load_model(config,adapter,True)
        initial={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        cb=AuditCallback(out,steps)
        common=dict(output_dir=str(out/'checkpoints'),max_steps=horizon or steps,per_device_train_batch_size=1,
                    bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                    report_to='none',save_strategy='steps',save_steps=50,save_total_limit=4,logging_steps=1,
                    optim='adamw_torch',weight_decay=0.0,lr_scheduler_type='linear',warmup_steps=0,max_grad_norm=1.0,
                    dataloader_pin_memory=False,dataloader_num_workers=0,disable_tqdm=True,seed=seed,data_seed=seed,
                    torch_empty_cache_steps=1)
        if True:
            group_counter=[0]
            def tool_reward(completions,ground_truth,sample_id,completion_ids,**kwargs):
                if len(completions)!=4 or len(set(sample_id))!=1:raise RuntimeError('Broken 4-candidate grouping')
                group=group_counter[0]
                if group>=len(records) or sample_id[0]!=records[group]['id']:raise RuntimeError('Unexpected request order')
                texts=[c[0]['content'] if isinstance(c,list) else c for c in completions]
                scores=[score(t,g,'dense') for t,g in zip(texts,ground_truth)]
                values=np.array([r['total'] for r in scores],dtype=np.float32)
                # Log the identical ddof/epsilon convention used by installed TRL.
                std=float(values.std(ddof=1));mean=float(values.mean())
                cb.zero_groups=cb.zero_groups+1 if std<=1e-8 else 0
                trainer.audit_expected=(values-mean)/(std+1e-4)
                trainer.audit_group=group
                strict=[evaluate_prediction(text,g)['call_multiset_exact'] for text,g in zip(texts,ground_truth)]
                all_wrong=not any(strict)
                u=random_u(seed,group,int(sample_id[0]))
                mask=0 if (method=='gate' and all_wrong) or (method=='random' and std>1e-8 and u<RANDOM_MASK_PROBABILITY) else 1
                trainer.audit_mask=mask
                append_jsonl(out/'group_masks.jsonl',{'group':group,'id':int(sample_id[0]),
                    'method':method,'all_wrong':all_wrong,'raw_std':std,'mask':mask,'random_u':u,
                    'strict_correct_candidates':sum(strict)})
                for i,(text,g,r) in enumerate(zip(texts,ground_truth,scores)):
                    e=evaluate_prediction(text,g)
                    append_jsonl(out/'rollouts.jsonl',{'group_id':group,'id':int(sample_id[i]),'candidate_index':i,
                        'prediction':text,'reference':g,'reward':r,'group_mean':mean,'group_std':std,
                        'token_ids':completion_ids[i],'generated_tokens':len(completion_ids[i]),
                        'hit_generation_limit':len(completion_ids[i])>=config['completion_cap'],
                        'advantage':float((values[i]-mean)/(std+1e-4)),
                        'advantage_applied':float((values[i]-mean)/(std+1e-4))*mask,
                        'group_all_wrong':all_wrong,'mask':mask,'random_u':u,'evaluation':e})
                group_counter[0]+=1
                return values.tolist()
            args=GRPOConfig(**common,gradient_accumulation_steps=4,num_generations=4,steps_per_generation=4,
                max_completion_length=config['completion_cap'],learning_rate=config['rl_lr'],
                temperature=config['temperature'],top_p=config['top_p'],top_k=None,shuffle_dataset=False,
                use_vllm=False,beta=0.0,loss_type='grpo',scale_rewards='group',epsilon=0.2,num_iterations=1,
                mask_truncated_completions=False,generation_kwargs={'min_new_tokens':config['completion_cap']} if force_length else None)
            data=Dataset.from_list([{'prompt':r['prompt'],'ground_truth':r['ground_truth'],'sample_id':r['id']} for r in records])
            trainer=AuditedGRPO(model=model,args=args,processing_class=tok,train_dataset=data,reward_funcs=tool_reward,callbacks=[cb])
            trainer.audit_out=out
        # TRL casts QLoRA adapters to BF16 during construction. Restore the exact
        # common FP32 adapter before optimizer creation, avoiding unequal starts
        # and avoiding rounding being counted as a policy update.
        for n,p in trainer.model.named_parameters():
            if n in initial:
                p.data=initial[n].to(device=p.device,dtype=torch.float32)
                assert torch.equal(p.detach().cpu(),initial[n])
        write_json(out/'initialization_check.json',{'exact_common_fp32_adapter':True,
                   'trainable_parameters':sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)})
        with Monitor(out/'gpu.jsonl') as monitor:
            result=trainer.train()
        assert trainer.state.global_step==steps or cb.stop_reason in ('stopped_no_signal','resource_limit')
        delta=sum((p.detach().float().cpu()-initial[n]).abs().sum().item() for n,p in trainer.model.named_parameters() if n in initial)
        trainer.model.save_pretrained(out/'adapter')
        assert group_counter[0]==trainer.state.global_step
        report={'steps':trainer.state.global_step,'requested_steps':steps,'stop_reason':cb.stop_reason,'train_loss':float(result.training_loss),'parameter_delta_l1':delta,**monitor.summary}
        write_json(out/'report.json',report)
        write_json(out/'status.json',{'status':cb.stop_reason or 'complete','adapter_hashes':fingerprint_adapter(out/'adapter')})
        print(json.dumps(report),flush=True)
        return report
    except Exception as exc:
        write_json(out/'status.json',{'status':'failed','error':repr(exc)})
        raise
    finally:
        del trainer,model
        clear()
