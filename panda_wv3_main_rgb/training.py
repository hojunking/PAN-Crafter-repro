"""FP32 AdamW with original warmup/cosine and complete atomic restart state."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import torch
from donor30.artifacts import _cpu
from panda_method.calibration import state_dict_sha256
from panda_stage2_condq.training import TrainingSpec,rng_state,restore_rng,_finite_tree
from panda_stage2_condq.stream import TrainStream,StreamSpec
from ta2.evaluation import file_sha
from directshift48.reporting import atomic_json
from . import plan


def model_hash(model):
    return plan.digest({role:state_dict_sha256(getattr(model,role))
        for role in ('aligner','teacher','student') if getattr(model,role,None) is not None})


def stream_for(cfg,registered,device):
    seeds=cfg['stream_seeds']
    return TrainStream(registered,StreamSpec(seeds['rr_data'],seeds['rr_augmentation']),
                       microbatch_size=cfg['microbatch'],effective_batch_size=cfg['batch'],device=device)


class Trainer:
    def __init__(self,cfg,model):
        plan.validate(cfg)
        self.cfg,self.model,self.updates,self.poisoned=deepcopy(cfg),model,0,False
        self.schedule=TrainingSpec(lr=cfg['lr'],total_updates=cfg['horizon'],warmup_updates=cfg['warmup'],
            weight_decay=cfg['weight_decay'],effective_batch_size=cfg['batch'],microbatch_size=cfg['microbatch'])
        if cfg['stage']=='stage1':
            groups=[]
            if cfg['alignment']=='LEARNED':
                groups.append({'params':model.aligner.parameters(),'lr':cfg['lr_aligner'],'peak_lr':cfg['lr_aligner'],'role':'A'})
            groups.append({'params':model.teacher.parameters(),'lr':cfg['lr'],'peak_lr':cfg['lr'],'role':'T'})
        else:
            groups=[{'params':model.student.parameters(),'lr':cfg['lr'],'peak_lr':cfg['lr'],'role':'S'}]
        self.optimizer=torch.optim.AdamW(groups,betas=tuple(cfg['betas']),eps=cfg['adam_eps'],weight_decay=cfg['weight_decay'])
        self.frozen = ({k:state_dict_sha256(getattr(model,k)) for k in ('aligner','teacher')}
                       if cfg['stage']=='stage2' else {})

    def audit(self):
        if self.poisoned:
            raise ValueError('Poisoned optimizer state requires restoration from a saved checkpoint')
        for role,sha in self.frozen.items():
            m=getattr(self.model,role)
            if any(p.requires_grad or p.grad is not None for p in m.parameters()) or state_dict_sha256(m)!=sha:
                raise ValueError('Frozen reference changed: '+role)
        _finite_tree(self.optimizer.state_dict())
        _finite_tree(self.model.state_dict())

    def step(self,batches):
        if self.poisoned or self.updates>=self.cfg['horizon'] or sum(len(b.pan) for b in batches)!=self.cfg['batch']:
            raise ValueError('Invalid optimizer state, horizon or effective batch')
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        rate=self.schedule.rate(self.updates+1)
        for g in self.optimizer.param_groups:
            g['lr']=rate*g['peak_lr']/self.cfg['lr']
        result={}
        try:
            for b in batches:
                terms=self.model(b)
                if not all(bool(torch.isfinite(v).all()) for v in terms.values()):
                    raise FloatingPointError('Nonfinite loss/diagnostics')
                fraction=len(b.pan)/self.cfg['batch']
                (terms['loss']*fraction).backward()
                for k,v in terms.items(): result[k]=result.get(k,0.)+float(v.detach())*fraction
            for g in self.optimizer.param_groups:
                for p in g['params']:
                    if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                        raise FloatingPointError('Nonfinite gradient before optimizer update')
            self.optimizer.step();self.updates+=1
        except BaseException:
            self.poisoned=True
            raise
        self.optimizer.zero_grad(set_to_none=True)
        if self.updates%100==0:self.audit()
        return dict(result,step=self.updates,lr=rate)


def save(path,trainer,stream,pins,*,immutable=False,evaluated=False):
    trainer.audit()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value={'schema':'WV3_MAIN_RGB_EVALUATED_MODEL_V1' if evaluated else 'WV3_MAIN_RGB_COMPLETE_CHECKPOINT_V1',
        'config':trainer.cfg,'pins':pins,'updates':trainer.updates,
        'model':_cpu(trainer.model.state_dict()),'model_sha256':model_hash(trainer.model)}
    if not evaluated:
        value.update(optimizer=_cpu(trainer.optimizer.state_dict()),stream=stream.state_dict(),rng=rng_state())
    fd,tmp=tempfile.mkstemp(prefix='.checkpoint-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(value,f);f.flush();os.fsync(f.fileno())
        if immutable:
            os.link(tmp,path) # exclusive publication, never overwrite a selected model
        else:os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    receipt={'path':str(path),'file_sha256':file_sha(path),'step':trainer.updates,
             'model_sha256':value['model_sha256'],'config_sha256':trainer.cfg['config_sha256']}
    atomic_json(path.with_suffix('.json'),receipt)
    return receipt


def load(path,trainer,stream,pins):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if file_sha(path)!=meta['file_sha256']:raise ValueError('Checkpoint file hash mismatch')
    x=torch.load(path,map_location='cpu',weights_only=False)
    if (x.get('schema')!='WV3_MAIN_RGB_COMPLETE_CHECKPOINT_V1' or x['config']!=trainer.cfg
            or x['pins']!=pins or x['updates']!=meta['step']
            or not 0<=x['updates']<=trainer.cfg['horizon']):
        raise ValueError('Resume config/code/data/evaluator/update mismatch')
    trainer.model.load_state_dict(x['model'],strict=True)
    trainer.optimizer.load_state_dict(x['optimizer'])
    trainer.updates=x['updates'];stream.load_state_dict(x['stream']);restore_rng(x['rng'])
    if model_hash(trainer.model)!=x['model_sha256'] or x['model_sha256']!=meta['model_sha256']:
        raise ValueError('Restored model differs from checkpoint identity')
    expected=trainer.schedule.rate(trainer.updates) if trainer.updates else trainer.cfg['lr']
    for g in trainer.optimizer.param_groups:
        if g['lr']!=expected*g['peak_lr']/trainer.cfg['lr']:
            raise ValueError('Restored LR is inconsistent with absolute update')
    trainer.audit()
    return meta
