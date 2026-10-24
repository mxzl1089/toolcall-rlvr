"""Actual Transformers SFT and TRL GRPO entrypoints with audit logs."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer,TrainingArguments,TrainerCallback,DataCollatorForSeq2Seq,set_seed
from trl import GRPOTrainer,GRPOConfig
from toolrl.data import load_prepared,schedule,tokenized,write_json,append_jsonl
from toolrl.runtime import load_model,tokenizer,Monitor,init_run,fingerprint_adapter,clear
from toolrl.rewards_train import score
from toolrl.eval_strict import evaluate_prediction


class OrderedSFT(Trainer):
    def _get_train_sampler(self,train_dataset=None):
        from torch.utils.data import SequentialSampler
        return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)


class AuditedGRPO(GRPOTrainer):
    def _generate_and_score_completions(self,inputs):
        result=super()._generate_and_score_completions(inputs)
        actual=result['advantages'].detach().float().cpu().numpy()
        if not np.allclose(actual,self.audit_expected,atol=2e-5,rtol=2e-5):
            raise RuntimeError('Logged and actual TRL advantages disagree')
        append_jsonl(self.audit_out/'advantage_checks.jsonl',{'group':self.audit_group,
            'actual_advantages':actual.tolist(),'max_error':float(np.max(np.abs(actual-self.audit_expected)))})
        return result


class AuditCallback(TrainerCallback):
    def __init__(self,out,stop_after):self.out=out;self.stop_after=stop_after;self.prev=None;self.started=0;self.zero_groups=0;self.stop_reason=None
    def on_train_begin(self,args,state,control,model=None,**kw):
        self.prev={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.started=time.perf_counter()
    def on_step_end(self,args,state,control,model=None,**kw):
        now={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        delta=sum((now[n]-v).abs().sum().item() for n,v in self.prev.items())
        if not math.isfinite(delta):raise RuntimeError('Nonfinite parameter update')
        append_jsonl(self.out/'updates.jsonl',{'step':state.global_step,'lora_delta_l1':delta,'elapsed':time.perf_counter()-self.started})
        self.prev=now
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
    if method not in ('sft','continue','sparse','dense'):raise ValueError(method)
    if method!='sft' and adapter is None:raise ValueError('A common SFT adapter is required')
    set_seed(seed)
    original=load_prepared(prepared,'train',config)
    if longest:
        ordered=sorted(original,key=lambda r:(r['full_tokens'] if method in ('sft','continue') else r['prompt_tokens'],r['full_tokens'],r['id']),reverse=True)
    else:ordered=schedule(original,seed,len(original))
    count=steps*2 if method in ('sft','continue') else steps
    if count>len(ordered):raise ValueError('Requested exposure exceeds frozen pool')
    records=ordered[:count]
    if method=='continue' and steps*2!=config['exposures'] and not longest:
        # Small preflight continuation runs are tagged in run.json, never final results.
        pass
    metadata={'method':method,'seed':seed,'steps':steps,'horizon':horizon or steps,'ids':[r['id'] for r in records],
              'initial_adapter':str(adapter) if adapter else None,'initial_adapter_hashes':fingerprint_adapter(adapter),
              'resource_longest':longest,'force_full_completion':force_length}
    out=init_run(out,config,metadata)
    tok=tokenizer(config);model=None;trainer=None
    try:
        model=load_model(config,adapter,True)
        initial={n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        cb=AuditCallback(out,steps)
        common=dict(output_dir=str(out/'checkpoints'),max_steps=horizon or steps,per_device_train_batch_size=1,
                    bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                    report_to='none',save_strategy='steps',save_steps=25 if method=='continue' else 50,save_total_limit=4,logging_steps=1,
                    optim='adamw_torch',weight_decay=0.0,lr_scheduler_type='linear',warmup_steps=0,max_grad_norm=1.0,
                    dataloader_pin_memory=False,dataloader_num_workers=0,disable_tqdm=True,seed=seed,data_seed=seed,
                    torch_empty_cache_steps=1)
        if method in ('sft','continue'):
            rows=[tokenized(r,tok) for r in records]
            assert all(len(r['input_ids'])<=config['max_length'] for r in rows)
            write_json(out/'mask_audit.json',[{'id':records[i]['id'],'length':len(r['labels']),'masked':r['labels'].count(-100),'supervised':sum(v!=-100 for v in r['labels'])} for i,r in enumerate(rows)])
            args=TrainingArguments(**common,gradient_accumulation_steps=2,learning_rate=config['sft_lr'] if method=='sft' else config['rl_lr'])
            trainer=OrderedSFT(model=model,args=args,train_dataset=Dataset.from_list(rows),
                               data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100),callbacks=[cb])
        else:
            group_counter=[0]
            def tool_reward(completions,ground_truth,sample_id,completion_ids,**kwargs):
                if len(completions)!=4 or len(set(sample_id))!=1:raise RuntimeError('Broken 4-candidate grouping')
                group=group_counter[0]
                if group>=len(records) or sample_id[0]!=records[group]['id']:raise RuntimeError('Unexpected request order')
                texts=[c[0]['content'] if isinstance(c,list) else c for c in completions]
                scores=[score(t,g,method) for t,g in zip(texts,ground_truth)]
                values=np.array([r['total'] for r in scores],dtype=np.float32)
                # Log the identical ddof/epsilon convention used by installed TRL.
                std=float(values.std(ddof=1));mean=float(values.mean())
                cb.zero_groups=cb.zero_groups+1 if std<=1e-8 else 0
                trainer.audit_expected=(values-mean)/(std+1e-4)
                trainer.audit_group=group
                for i,(text,g,r) in enumerate(zip(texts,ground_truth,scores)):
                    e=evaluate_prediction(text,g)
                    append_jsonl(out/'rollouts.jsonl',{'group_id':group,'id':int(sample_id[i]),'candidate_index':i,
                        'prediction':text,'reference':g,'reward':r,'group_mean':mean,'group_std':std,
                        'token_ids':completion_ids[i],'generated_tokens':len(completion_ids[i]),
                        'hit_generation_limit':len(completion_ids[i])>=config['completion_cap'],
                        'advantage':float((values[i]-mean)/(std+1e-4)),'evaluation':e})
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
        assert trainer.state.global_step==steps or cb.stop_reason=='stopped_no_signal'
        delta=sum((p.detach().float().cpu()-initial[n]).abs().sum().item() for n,p in trainer.model.named_parameters() if n in initial)
        trainer.model.save_pretrained(out/'adapter')
        if method in ('sparse','dense'):
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
