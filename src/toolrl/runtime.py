"""Offline QLoRA loading, per-run provenance and low-overhead resource logging."""
import gc
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from importlib.metadata import version
from toolrl.data import write_json,append_jsonl
from toolrl.verify_split import sha256_file

def environment():
    names=('torch','transformers','trl','peft','bitsandbytes','numpy','scipy','datasets','accelerate')
    return {'python':platform.python_version(),'packages':{n:version(n) for n in names}}

def fingerprint_adapter(folder):
    if folder is None:return {}
    p=Path(folder)
    return {x.name:sha256_file(x) for x in p.iterdir() if x.suffix in ('.safetensors','.json')}

def tokenizer(config):
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    if t.pad_token_id is None:t.pad_token=t.eos_token
    t.padding_side='left'
    return t

def load_model(config,adapter=None,trainable=True):
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import LoraConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
        device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=trainable,gradient_checkpointing_kwargs={'use_reentrant':False})
    if adapter:model=PeftModel.from_pretrained(model,str(adapter),is_trainable=trainable)
    elif trainable:model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.config.use_cache=not trainable
    if trainable:
        assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)>0
    return model

def clear():
    import torch
    gc.collect();torch.cuda.empty_cache()

class Monitor:
    def __init__(self,path):self.path=Path(path);self.stop=threading.Event();self.thread=None
    def __enter__(self):
        import torch
        torch.cuda.reset_peak_memory_stats();self.start=time.perf_counter()
        def sample():
            while not self.stop.is_set():
                try:
                    s=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    append_jsonl(self.path,{'elapsed':time.perf_counter()-self.start,'sample':s.stdout.strip(),'returncode':s.returncode})
                except Exception as e:append_jsonl(self.path,{'monitor_error':str(e)})
                self.stop.wait(1)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        import torch
        self.stop.set();self.thread.join(timeout=6)
        self.summary={'seconds':time.perf_counter()-self.start,'allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
                      'shared_memory_observed':False,'note':'nvidia-smi sampling is not proof that Windows shared GPU memory was unused'}

def init_run(out,config,metadata):
    out=Path(out)
    if out.exists():raise FileExistsError(f'Never overwrite a run: {out}')
    out.mkdir(parents=True)
    write_json(out/'config.json',config)
    write_json(out/'environment.json',environment())
    sources=Path(__file__).parent
    write_json(out/'source_hashes.json',{p.name:sha256_file(p) for p in sources.glob('*.py')})
    write_json(out/'run.json',metadata)
    write_json(out/'status.json',{'status':'running'})
    return out
