"""Source-bound MAIN-A acceptance, without claiming an unexecuted GPU pass."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import io
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import default_collate

from fh12.model import build_model, state_hash
from fh12.losses import student_losses as original_losses, routed_student_backward
from fh12.training import rng_state, restore_rng, tensor_cpu_tree
from maina_hqnr.data import BatchStream,build_dataset,stream_manifest,epoch_loader,completed_updates

# Fixed before comparison, never adjusted in response to a failed measurement.
TOLERANCES = dict(cpu=dict(rtol=0.,atol=0.),cuda=dict(rtol=1e-6,atol=1e-7))


@contextmanager
def isolated_probe(*models):
    state=rng_state();modes=[model.training for model in models]
    try:
        yield
    finally:
        for model,training in zip(models,modes):model.train(training)
        restore_rng(state)


def rng_equal(left,right):
    lc,rc=left['cuda'],right['cuda']
    return (left['python']==right['python'] and left['numpy'][0]==right['numpy'][0]
        and np.array_equal(left['numpy'][1],right['numpy'][1]) and left['numpy'][2:]==right['numpy'][2:]
        and torch.equal(left['torch'],right['torch']) and ((lc is None and rc is None) or
        (lc is not None and rc is not None and len(lc)==len(rc) and all(torch.equal(a,b) for a,b in zip(lc,rc)))))


def assert_tree_equal(left,right):
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or not torch.equal(left.cpu(),right.cpu()):raise AssertionError('Tensor state mismatch')
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise AssertionError('State keys mismatch')
        for key in left:assert_tree_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        if type(left)!=type(right) or len(left)!=len(right):raise AssertionError('State sequence mismatch')
        for a,b in zip(left,right):assert_tree_equal(a,b)
    elif left!=right:
        raise AssertionError('Scalar state mismatch')


def _comparison(left,right,tolerance,label):
    if left.shape!=right.shape or not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
        raise AssertionError(label+': invalid shape/nonfinite')
    error=float((left.detach()-right.detach()).abs().max())
    if not torch.allclose(left,right,**tolerance):raise AssertionError(label+': numerical mismatch '+str(error))
    return error


def compare_base(model,original,teacher,batch,q,qref,tau,case,device):
    from maina_hqnr.training import objectives
    tol=TOLERANCES[torch.device(device).type];gt,_,ms,lp,pan,meta=batch
    gt,ms,lp,pan=[v.to(device) for v in (gt,ms,lp,pan)]
    model.zero_grad(set_to_none=True);original.zero_grad(set_to_none=True)
    out,terms=objectives(model,teacher,batch,q,qref,tau,case,device)
    native=original(pan,ms,lp)
    with torch.no_grad():tout=teacher(pan,ms,lp)
    w=(qref/(qref+q[meta[:,0].to(device),meta[:,1].to(device)])).detach()
    old=original_losses(native,tout,gt,tau,w)
    error={key:_comparison(out[key],native[key],tol,key) for key in ('x_in','pan_aligned','L','H','ms_base','delta','y')}
    for key in ('L_U','L_A','hard_i','soft_i','edge_i','difficulty','advantage','q_weights'):
        error['loss_'+key]=_comparison(terms[key],old[key],tol,'loss '+key)
    routed_student_backward(model,terms);routed_student_backward(original,old)
    for (name,p),(other,qparam) in zip(model.named_parameters(),original.named_parameters()):
        if name!=other:raise AssertionError('Original/new parameter ordering differs')
        error['grad_'+name]=_comparison(p.grad,qparam.grad,tol,'gradient '+name)
    return dict(passed=True,max_abs_error=max(error.values()),comparisons=len(error),tolerance=tol,
                includes=['frontend','shared_warp','output','loss','U_gradient','A_gradient'])


def smoke_roundtrip(model,teacher,dataset,q,qref,tau,case,cfg,device,workers=4):
    """Two actual batch48 updates, disk-format full-state roundtrip and replay."""
    from maina_hqnr.training import make_optimizer,make_scheduler,objectives,verify_restored_state
    tol=TOLERANCES[torch.device(device).type];stream=BatchStream(len(dataset),48,cfg['seed'])
    opt=make_optimizer(model,cfg);sch=make_scheduler(opt,cfg)
    # A short DataLoader still receives the exact original index/view tuples.
    def one_step():
        loader=epoch_loader(dataset,stream,cfg['seed'],workers=workers,pin_memory=torch.device(device).type=='cuda')
        batch=next(iter(loader));model.train();opt.zero_grad(set_to_none=True)
        _,losses=objectives(model,teacher,batch,q,qref,tau,case,device)
        routed_student_backward(model,losses)
        opt.step();sch.step()
        if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
        stream.cursor+=1
    one_step();snapshot=io.BytesIO()
    torch.save(dict(full_state=True,precision='fp32',update=1,model=tensor_cpu_tree(model.state_dict()),
        optimizer=tensor_cpu_tree(opt.state_dict()),scheduler=sch.state_dict(),sampler=stream.state_dict(),rng=rng_state()),snapshot)
    one_step();wanted=copy.deepcopy(tensor_cpu_tree(model.state_dict()))
    wanted_opt=copy.deepcopy(tensor_cpu_tree(opt.state_dict()));wanted_rng=rng_state();wanted_stream=stream.state_dict()
    snapshot.seek(0);restored=torch.load(snapshot,map_location='cpu',weights_only=False)
    model.load_state_dict(restored['model'],strict=True);opt.load_state_dict(restored['optimizer'])
    sch.load_state_dict(restored['scheduler']);stream.load_state_dict(restored['sampler']);restore_rng(restored['rng'])
    verify_restored_state(restored,{},stream,sch);one_step()
    maximum=max(_comparison(v.detach().cpu(),wanted[k],tol,k) for k,v in model.state_dict().items())
    # Optimizer counters are exact; floating moment differences obey the same
    # predeclared CUDA tolerance as weights (CUDA scatter backward is not bitwise).
    observed=tensor_cpu_tree(opt.state_dict())
    if observed['param_groups']!=wanted_opt['param_groups']:raise AssertionError('Resume parameter groups changed')
    for idx,state in observed['state'].items():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):_comparison(value,wanted_opt['state'][idx][key],tol,'optimizer '+key)
            elif value!=wanted_opt['state'][idx][key]:raise AssertionError('Optimizer state mismatch')
    assert_tree_equal(stream.state_dict(),wanted_stream)
    if not rng_equal(rng_state(),wanted_rng) or sch.last_epoch!=2:raise AssertionError('Resume RNG/scheduler mismatch')
    return dict(passed=True,updates=2,batch_size=48,workers=workers,resumed_from_update=1,
        max_abs_error=maximum,tolerance=tol,optimizer_verified=True,rng_verified=True,sampler_verified=True)


def run_acceptance(root,server,bindings,device='cuda'):
    """Production admission calls CUDA; CPU is explicitly TEST_ONLY, never PASSED."""
    from maina_hqnr.common import source_identity,object_sha,apply_runtime_policy
    from maina_hqnr.plan import make_case,build_config
    from maina_hqnr.assets import load_training_reference,validate_bindings
    from maina_hqnr.training import fresh_model
    root=Path(root);dev=torch.device(device)
    if dev.type not in TOLERANCES or (dev.type=='cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Requested acceptance device unavailable; no implicit fallback')
    validate_bindings(bindings,root,server,rehash=True);runtime=apply_runtime_policy(root)
    # Do not perturb caller RNG, model state, controller cursor or production init.
    with isolated_probe():
        teacher,reference,q,_bridge=load_training_reference(bindings,device=dev)
        teacher.eval().requires_grad_(False);teacher_hash=state_hash(teacher.state_dict())
        dataset=build_dataset(bindings['dataset_manifest'],'train',root=root)
        q=torch.as_tensor(q,dtype=torch.float32,device=dev)
        case=make_case(server,0,'BASE');cfg=build_config(case,root=root,bindings=bindings)
        model,initial=fresh_model(cfg,teacher);model.to(dev)
        original,_=build_model('PLH',104,[1,2,2],cfg['seed'],role='S',teacher_aligner_state=teacher.aligner.state_dict())
        original.to(dev)
        stream=BatchStream(len(dataset),48,cfg['seed'])
        batch=default_collate([dataset[index] for index in stream.remaining_batches()[0][:2]])
        parity=compare_base(model,original,teacher,batch,q,reference['q_ref'],reference['tau_R'],case,dev)
        del original
        hashes=[]
        for code in ('BASE','AL05','AL15','BE005','BE020','ED0006','ED006'):
            arm=make_case(server,0,code);armcfg=build_config(arm,root=root,bindings=bindings)
            other,init=fresh_model(armcfg,teacher)
            hashes.append(dict(case=code,U=init['hashes']['U'],A=init['hashes']['A'],
                stream=object_sha(stream_manifest(len(dataset),48,arm['seed']))))
            del other
        if len({(h['U'],h['A'],h['stream']) for h in hashes})!=1:raise AssertionError('Seven variants not paired')
        # The smoke only mutates this isolated in-memory model.
        resume=smoke_roundtrip(model,teacher,dataset,q,reference['q_ref'],reference['tau_R'],case,cfg,dev)
        if state_hash(teacher.state_dict())!=teacher_hash or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
            raise AssertionError('Frozen F1 was mutated or received gradients')
        return dict(passed=dev.type=='cuda',status='PASSED' if dev.type=='cuda' else 'CPU_TEST_ONLY',
            device=str(dev),runtime_policy=runtime,source_identity=source_identity(root),
            bindings_sha256=object_sha(bindings),tolerances=TOLERANCES[dev.type],parity=parity,
            initialization_and_stream=hashes,resume=resume,teacher_unchanged=True,
            performance_reproduction_claimed=False,production_state_written=False)
