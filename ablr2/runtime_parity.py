"""Measured old/new CPU numerical bridge, never an implicit source allowlist.

The same standalone probe is run in two separate interpreters importing each
release's own code. It performs no campaign training/data/Sheet writes. Evidence
is synthetic numerical parity, not a claim of CUDA bitwise reproducibility or
real C03 initialization pairing (the latter must also inspect the run artifacts).
"""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys

SCHEMA='ABLR2X_RUNTIME_NUMERICAL_PARITY_v1'
PROBE_VERSION='FULL_WIDTH_D123_D122_NATIVE64_C03_C07_ADAMW_2STEP_v1'
RUNTIME_KEYS=('torch','numpy','scipy','skimage','cuda','cudnn','tf32_matmul','tf32_cudnn',
    'runtime_policy_sha256','cudnn_benchmark','cudnn_deterministic','deterministic_algorithms')
NUMERIC_KEYS=('teacher_initial','teacher_forward','teacher_loss','teacher_gradient','teacher_optimizer',
    'C03','C07','native_views','native_stream','lp_recipe','components','schedules','evaluation','validation')
UNCHANGED_CORE=('ablr2/model.py','ablr2/losses.py','pa/aligner.py','pa/warp.py',
    'pa/losses.py','pa/offset.py','fh12/calibration.py',
    'fh12/model.py','fh12/training.py','fh12/data.py','model/pancrafter_paper.py',
    'model/pancrafter.py','model/swin.py','tools/repair_lpan.py',
    'qg40/calibration.py','qg40/data.py','qg40/model.py','qg40/losses.py','qg40/evaluation.py',
    'tools/metrics/eval_rr.py','tools/metrics/eval_fr.py','tools/metrics/q2n.py','tools/metrics/jqm.py')


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)


def _measurements(values):
    for key in ('teacher_forward','teacher_gradient','teacher_optimizer','native_views'):
        if not _digest(values.get(key)):raise ValueError('Missing measured tensor digest: '+key)
    if (set(values.get('components',{}))!={f'C{i:02}' for i in range(17)}
            or values.get('evaluation',{}).get('n_scenes')!=20
            or not isinstance(values.get('validation'),(int,float)) or not math.isfinite(values['validation'])):
        raise ValueError('Incomplete measured component/evaluation coverage')
    for key in ('C03','C07'):
        branch=values.get(key,{})
        if (len(branch.get('observed',[]))!=2 or not _digest(branch.get('final'))
                or any(not _digest(branch.get('initial',{}).get(k)) for k in ('U','A','full'))
                or branch['final']==branch['initial']['full']
                or any(not _digest(row.get('gradient')) or not _digest(row.get('output'))
                    or not math.isfinite(row.get('gradient_l1',float('nan'))) or row['gradient_l1']<=0
                    for row in branch['observed'])):
            raise ValueError('Missing/non-updating real optimizer/gradient evidence: '+key)


def probe(root,server):
    """Called only in an isolated child before importing any release module."""
    root=Path(root).resolve()
    # A direct script starts with .../ablr2 on sys.path; its model.py must not
    # shadow the release's namespace package model/pancrafter_paper.py.
    own_directory=Path(__file__).resolve().parent
    sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=own_directory]
    sys.path.insert(0,str(root))
    import tempfile
    import numpy as np
    import h5py
    import torch
    from ablr2.common import apply_runtime_policy,source_identity
    from ablr2.plan import verify_lane,sensor_spec,component_config,build_config,cases_for
    from ablr2.model import build_model,state_hash
    from ablr2.losses import teacher_loss,student_losses,routed_student_backward
    from ablr2.training import make_optimizer,make_scheduler,cosine_factor,aligner_factor
    from ablr2.data import ABLR2Dataset
    from fh12.training import BatchStream
    from fh12.data import RECIPE
    from tools.repair_lpan import make_lpan
    from ablr2.evaluation import rr_metrics,validation_ergas
    if server not in ('s1','s2'):raise ValueError('Legacy bridge applies only WV3/s1 or QB/s2')
    torch.set_num_threads(1);apply_runtime_policy(root)
    sensor=verify_lane(server);spec=sensor_spec(sensor);bands=spec.num_bands
    origin=source_identity(root)
    selected={c.case_id:c for c in cases_for(server) if c.sweep=='P01'}
    generator=np.random.default_rng(820123)
    raw_pan=generator.uniform(40,1900,(3,1,64,64)).astype(np.float32)
    raw_ms=generator.uniform(100,1700,(3,bands,16,16)).astype(np.float32)
    raw_gt=generator.uniform(100,1700,(3,bands,64,64)).astype(np.float32)
    raw_lp=make_lpan(raw_pan.astype(np.float64)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='ablr2-parity-') as temporary:
        source,lp=Path(temporary)/'native.h5',Path(temporary)/'lp.h5'
        with h5py.File(source,'w') as out:
            out['pan']=raw_pan;out['ms']=raw_ms;out['gt']=raw_gt;out['lms']=raw_gt
        with h5py.File(lp,'w') as out:out['lpan']=raw_lp
        dataset=ABLR2Dataset(source,lp,spec=spec,split='train')
        views=[dataset.get_view(i,r,augment=True) for i in range(3) for r in range(4)]
        values={'native_views':state_hash({str(i)+'/'+str(k):v for i,row in enumerate(views) for k,v in enumerate(row)})}
        rows=[dataset.base(i) for i in (0,1)]
        gt,ms,low,pan=[torch.stack([r[i] for r in rows]) for i in (0,2,3,4)]
        teacher,init=build_model(bands=bands,seed=selected['TPLUS'].seed,role='T')
        values['teacher_initial']=init['hashes']
        out=teacher(pan,ms,low)
        values['teacher_forward']=state_hash(out)
        losses=teacher_loss(teacher,out,gt,pan,ms,1,torch.Generator().manual_seed(314159),consistency_weight=1e-4,bands=bands)
        values['teacher_loss']={k:float(losses[k]) for k in ('total','rec','off')}
        losses['total'].backward()
        grads={n:p.grad for n,p in teacher.named_parameters() if p.grad is not None}
        if not grads or not sum(float(g.abs().sum()) for g in grads.values())>0:raise ValueError('Degenerate Teacher parity gradient')
        values['teacher_gradient']=state_hash(grads)
        optimizer=make_optimizer(teacher,build_config(selected['TPLUS']))
        optimizer.step();values['teacher_optimizer']=state_hash(teacher.state_dict())
        teacher.eval().requires_grad_(False)
        with torch.no_grad():teacher_out=teacher(pan,ms,low)
        for key in ('C03','C07'):
            case=selected[key];component=component_config(key)
            model,init=build_model(bands=bands,seed=case.seed,role='S',component=component,
                teacher_aligner_state=teacher.aligner.state_dict())
            optimizer=make_optimizer(model,build_config(case));scheduler=make_scheduler(optimizer)
            initial=state_hash(model.state_dict());observed=[]
            for step in range(2):
                optimizer.zero_grad(set_to_none=True);out=model(pan,ms,low)
                losses=student_losses(out,teacher_out if key=='C07' else None,gt,component,
                    tau_R=.17 if key=='C07' else None,q_weights=torch.tensor([.2,.8]) if key=='C07' else None,bands=bands)
                routed_student_backward(model,losses)
                grad={n:p.grad for n,p in model.named_parameters() if p.grad is not None}
                norm=sum(float(g.abs().sum()) for g in grad.values())
                if not np.isfinite(norm) or norm<=0:raise ValueError('Degenerate Student parity gradient')
                observed.append(dict(output=state_hash(out),gradient=state_hash(grad),
                    losses={k:float(losses[k]) for k in ('L_U','L_A','hard','soft','edge')},gradient_l1=norm))
                optimizer.step();scheduler.step()
            final=state_hash(model.state_dict())
            if initial==final:raise ValueError('Parity optimizer fixture did not update')
            values[key]=dict(initial=init['hashes'],observed=observed,final=final,
                scheduler=scheduler.state_dict(),optimizer_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups])
            # LambdaLR serializes a placeholder for each lambda; no functions.
            del model,optimizer,scheduler
        stream=BatchStream(spec.nominal_train_n,48,selected['C03'].seed)
        values['native_stream']=dict(order=state_hash({'order':stream.order,'rotations':stream.rotations}),
            first_batches=list(stream.remaining_batches())[:3])
        values['lp_recipe']=dict(recipe=RECIPE,tensor=state_hash({'LP':torch.from_numpy(raw_lp)}))
        values['components']={f'C{i:02}':component_config(f'C{i:02}') for i in range(17)}
        values['schedules']={str(t):[cosine_factor(t),aligner_factor(t),aligner_factor(t,'BASE_TIMES_1over3_AFTER_24240')]
            for t in (0,1,99,100,24239,24240,49490,49999,50000)}
        base=np.arange(256*256,dtype=np.float64).reshape(1,1,256,256)/100+20
        truth=np.broadcast_to(base,(20,bands,256,256));pred=truth+np.sin(base/10)
        metrics=rr_metrics(pred,truth,spec,include_q=True)
        values['evaluation']={k:v for k,v in metrics.items() if k not in ('protocol',)}
        dataset.split='val'
        values['validation']=validation_ergas(teacher,dataset,'cpu')
    after=source_identity(root)
    if after!=origin:
        changed=[k for k in set(origin.get('files',{}))|set(after.get('files',{}))
            if origin.get('files',{}).get(k)!=after.get('files',{}).get(k)]
        changed += [k for k in RUNTIME_KEYS if origin.get(k)!=after.get(k)]
        raise ValueError('Source changed during numerical probe: '+', '.join(sorted(changed)))
    return dict(schema=SCHEMA,probe_version=PROBE_VERSION,server=server,sensor=sensor,source_identity=origin,
        imported_root=str(root),device='cpu',nondegenerate=True,numerics=values,numerics_sha256=_hash(values))


def compare_runtime_releases(origin_root,consumer_root,server,output_path=None):
    origin_root,consumer_root=Path(origin_root).resolve(),Path(consumer_root).resolve()
    script=consumer_root/'ablr2/runtime_parity.py'
    if server not in ('s1','s2') or not script.is_file():raise ValueError('Explicit supported releases required')
    results=[]
    for root in (origin_root,consumer_root):
        process=subprocess.run([sys.executable,str(script),'--probe-root',str(root),'--server',server],
            cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),
            capture_output=True,text=True,timeout=300,check=True)
        results.append(json.loads(process.stdout.strip().splitlines()[-1]))
    receipt=dict(schema=SCHEMA,probe_version=PROBE_VERSION,producer_sha256=_sha(script),
        origin_source_identity=results[0]['source_identity'],consumer_source_identity=results[1]['source_identity'],
        origin_root=str(origin_root),consumer_root=str(consumer_root),
        measurements=dict(original_sha256=results[0]['numerics_sha256'],consumer_sha256=results[1]['numerics_sha256']),
        original=results[0],consumer=results[1],original_sha256=_hash(results[0]),consumer_sha256=_hash(results[1]),
        passed=results[0]['numerics']==results[1]['numerics'],scope='Measured CPU synthetic contract; real run init/stream pairing additionally required')
    validate_runtime_parity(receipt,results[0]['source_identity'],results[1]['source_identity'],consumer_root)
    if output_path:
        from ablr2.common import immutable_json
        immutable_json(output_path,receipt)
    return receipt


def validate_runtime_parity(receipt,origin,consumer,root):
    """Reject PASS-only/missing/changed measurements and mismatched runtime flags."""
    if (receipt.get('schema')!=SCHEMA or receipt.get('probe_version')!=PROBE_VERSION
            or receipt.get('passed') is not True or receipt.get('producer_sha256')!=_sha(Path(root)/'ablr2/runtime_parity.py')
            or receipt.get('origin_source_identity')!=origin or receipt.get('consumer_source_identity')!=consumer
            or Path(receipt.get('consumer_root','')).resolve()!=Path(root).resolve()):
        raise ValueError('Measured runtime parity producer/protocol missing or changed')
    for label,identity in (('original',origin),('consumer',consumer)):
        result=receipt.get(label,{})
        numerics=result.get('numerics',{})
        if (result.get('source_identity')!=identity or result.get('schema')!=SCHEMA
                or result.get('probe_version')!=PROBE_VERSION or result.get('device')!='cpu'
                or result.get('nondegenerate') is not True or result.get('server') not in ('s1','s2')
                or set(numerics)!=set(NUMERIC_KEYS) or result.get('numerics_sha256')!=_hash(numerics)
                or receipt.get(label+'_sha256')!=_hash(result)):
            raise ValueError('Runtime parity measured evidence/source differs')
        recorded_root=receipt['origin_root' if label=='original' else 'consumer_root']
        if Path(result.get('imported_root','')).resolve()!=Path(recorded_root).resolve():
            raise ValueError('Runtime parity imported release path differs')
        if receipt.get('measurements',{}).get(label+'_sha256')!=result['numerics_sha256']:
            raise ValueError('Runtime parity measurement digest differs')
        _measurements(numerics)
    if (receipt['original']['server']!=receipt['consumer']['server']
            or receipt['original']['sensor']!=receipt['consumer']['sensor']
            or receipt['original']['numerics']!=receipt['consumer']['numerics']
            or any(origin.get(k)!=consumer.get(k) for k in RUNTIME_KEYS)):
        raise ValueError('Runtime numerical behavior or numerical environment differs')
    if any(not _digest(origin.get('files',{}).get(name))
            or origin['files'][name]!=consumer.get('files',{}).get(name) for name in UNCHANGED_CORE):
        raise ValueError('Core model/loss/warp/RNG source changed; representative CPU bridge is insufficient')
    return receipt


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--probe-root',required=True,type=Path)
    parser.add_argument('--server',required=True,choices=('s1','s2'));args=parser.parse_args()
    print(json.dumps(probe(args.probe_root,args.server),sort_keys=True,allow_nan=False))
