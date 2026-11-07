"""Disposable native CUDA updates, exact restoration, and baseline numerical parity."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import torch
from directshift48.reporting import atomic_json,utc_now
from donor30.artifacts import _same,_cpu
from panda_stage2_condq.training import configure_precision
from ta2.evaluation import evaluator_identity
from . import plan,assets,training
from .model import Stage1,Stage2
from .release import verify


def differences(a,b,path=''):
    if _same(a,b):return {}
    if isinstance(a,torch.Tensor):
        return {path:float((a.cpu()-b.detach().cpu()).abs().max())}
    if isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys():
        result={}
        for k in a:result.update(differences(a[k],b[k],path+'/'+str(k)))
        return result
    if isinstance(a,(list,tuple)) and isinstance(b,type(a)) and len(a)==len(b):
        result={}
        for k,(x,y) in enumerate(zip(a,b)):result.update(differences(x,y,path+'/'+str(k)))
        return result
    return {path:repr((a,b))}


def numerically_same(a,b):
    # Bilinear CUDA backward is not bitwise repeatable under the existing
    # reference policy. This tolerance applies ONLY after an exact restoration.
    # Model, optimizer, RNG and stream at the saved boundary must remain exact.
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:return False
        if not a.is_floating_point():return _same(a,b)
        return torch.allclose(a.cpu(),b.detach().cpu(),rtol=1e-6,atol=1e-8)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(numerically_same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a) is type(b) and len(a)==len(b) and all(numerically_same(x,y) for x,y in zip(a,b))
    return _same(a,b)


def baseline_parity(server, source, registered, root, pins, device):
    if server == 's1':
        from panda_wv3_main_rgb.model import Stage2 as Original
        from panda_wv3_main_rgb.plan import resolve as original_resolve
        cfg = plan.resolve(server,'H2_REF',1)
        donor,stats = assets.prepare_reference(cfg,registered,root,pins,device=device)
        new = Stage2(cfg,donor,stats).to(device)
        old = Original(original_resolve('s1','M0_OURS',1),donor,stats).to(device)
    else:
        from panda_wv3_ablation_rgb.model import Stage1 as Original
        from panda_wv3_ablation_rgb.plan import resolve as original_resolve
        cfg = plan.resolve(server,'H1_REF',1)
        new = Stage1(cfg,registered.inventory.manifest()).to(device)
        old = Original(original_resolve('s3','S1_A5_ALL',1),registered.inventory.manifest()).to(device)
    new.eval();old.eval()
    stream = training.stream_for(cfg,registered,device)
    batch = stream.next_update()[0]
    with torch.no_grad():
        a,b = new(batch),old(batch)
        common = set(a)&set(b)
        if any(not torch.equal(a[k],b[k]) for k in common):
            raise ValueError('Baseline training forward differs from original RGB W112 recipe')
        parity = {}
        for split in ('rr','fr'):
            sample = source.batch(split,[0],device=device)
            a,b = new.native_prediction(sample),old.native_prediction(sample)
            if not _same(a,b):
                raise ValueError('Baseline native inference differs: '+split)
            parity[split] = {'max_abs':0.,'shape':list(a['y'].shape)}
        if server == 's1':
            zero_cfg = plan.resolve(server,'H2_GT',1)
            zero = Stage2(zero_cfg,donor,stats).to(device)
            terms = zero(batch)
            if not torch.equal(terms['loss'],terms['rec']) or float(terms['weight_mean'])!=1.:
                raise ValueError('Uniform GT endpoint must have unit weight and unchanged support')
            if not torch.equal(terms['valid_fraction'],new(batch)['valid_fraction']):
                raise ValueError('Zero endpoint changed Teacher/Student support intersection')
    return {'baseline_training_terms_exact':sorted(common),'native_parity':parity,
            'zero_endpoint_checked':server=='s1'}


def run(server,device='cuda:0'):
    configure_precision();release=verify(environment=True)
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Actual CUDA preflight required')
    root=plan.campaign_root(server);folder=root/'preflight';folder.mkdir(parents=True,exist_ok=True)
    source,registered=assets.sources()
    pins={'source_release':release['source_sha256'],'evaluator':evaluator_identity(),
          'train_inventory':registered.inventory.manifest(),
          'split_manifests':{s:source.manifest_sha(s) for s in ('train','val','rr','fr')}}
    parity=baseline_parity(server,source,registered,root,pins,device)
    cfg=plan.resolve(server,'H2_A0' if server=='s1' else 'H1_SPA1',1)
    if cfg['stage']=='stage2':
        donor,stats=assets.prepare_reference(cfg,registered,root,pins,device=device)
        net=Stage2(cfg,donor,stats).to(device)
    else:
        net=Stage1(cfg,registered.inventory.manifest()).to(device)
    trainer=training.Trainer(cfg,net);stream=training.stream_for(cfg,registered,device)
    with tempfile.TemporaryDirectory(dir=folder) as scratch:
        path=Path(scratch)/'resume.pt'
        first=trainer.step(stream.next_update());training.save(path,trainer,stream,pins)
        resume_bytes=path.stat().st_size
        evaluated=Path(scratch)/'evaluated.pt'
        training.save(evaluated,trainer,stream,pins,evaluated=True)
        evaluated_bytes=evaluated.stat().st_size
        boundary={'model':_cpu(net.state_dict()),'optimizer':_cpu(trainer.optimizer.state_dict()),
                  'stream':deepcopy(stream.state_dict()),'rng':training.rng_state()}
        second=trainer.step(stream.next_update())
        expected_model=_cpu(net.state_dict());expected_opt=_cpu(trainer.optimizer.state_dict())
        expected_stream=deepcopy(stream.state_dict())
        training.load(path,trainer,stream,pins)
        restored={'model':_cpu(net.state_dict()),'optimizer':_cpu(trainer.optimizer.state_dict()),
                  'stream':deepcopy(stream.state_dict()),'rng':training.rng_state()}
        exact={k:_same(boundary[k],restored[k]) for k in boundary}
        if not all(exact.values()):
            raise ValueError('Checkpoint boundary restoration differs: '+repr(exact))
        replay=trainer.step(stream.next_update())
        diagnostic={'boundary_exact':exact,'second':second,'replayed':replay,
            'stream_exact':_same(expected_stream,stream.state_dict()),
            'model_bitwise_equal':_same(expected_model,net.state_dict()),
            'optimizer_bitwise_equal':_same(expected_opt,trainer.optimizer.state_dict()),
            'model_differences':differences(expected_model,net.state_dict()),
            'optimizer_differences':differences(expected_opt,trainer.optimizer.state_dict()),
            'replay_tolerance':{'rtol':1e-6,'atol':1e-8}}
        atomic_json(folder/'resume_diagnostic.json',diagnostic)
        if (second.keys()!=replay.keys()
                or any(not math.isclose(second[k],replay[k],rel_tol=1e-6,abs_tol=1e-8) for k in second)
                or not diagnostic['stream_exact']
                or not numerically_same(expected_model,net.state_dict())
                or not numerically_same(expected_opt,trainer.optimizer.state_dict())):
            raise ValueError('CUDA replay exceeded existing bilinear-backward tolerance')
        trainer.audit()
    receipt=dict(status='VERIFIED',at=utc_now(),source_release=release['source_sha256'],
        run_id=cfg['run_id'],server=server,disposable_only=True,production_updates=0,
        first_update=first,second_update=second,checkpoint_boundary_exact=exact,
        cuda_replay_tolerance=diagnostic['replay_tolerance'],baseline_parity=parity,
        checkpoint_bytes={'evaluated':evaluated_bytes,'complete_resume':resume_bytes})
    atomic_json(folder/'verified.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s1','s3'),required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    print(json.dumps(run(args.server,args.device)))
