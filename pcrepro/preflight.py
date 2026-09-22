"""Per-dataset admission: a missing QB input never blocks valid WV3/GF2."""
from pathlib import Path
import os
import shutil
import time
from pcrepro.common import (ROOT,atomic_json,read_json,read,camp,immutable_json,object_sha,
    source_identity,apply_runtime_policy,utcnow,resolved_path)
from pcrepro.plan import RECIPE,RECIPE_SHA256,DATASETS,SENSORS,verify_server,verify_sources

def disk_guard(root,estimated_bytes=0):
    target=(Path(root)/'work_dir').resolve()
    while not target.exists():target=target.parent
    free=shutil.disk_usage(target).free
    required=max(20*1024**3,int(estimated_bytes*1.25)+2*1024**3)
    return dict(allowed=free>=required,free_bytes=free,required_bytes=required,measured_path=str(target),automatic_pruning=False)

def execute(root,server,dataset,bindings,device='cuda',stopcheck=None):
    verify_server(server);verify_sources(root);apply_runtime_policy(root)
    if dataset not in DATASETS:raise ValueError('Unknown dataset')
    start=time.monotonic();folder=camp(root,server)
    supplied=read_json(bindings) if not isinstance(bindings,dict) else bindings
    binding=supplied.get('datasets',{}).get(dataset)
    if not binding:raise ValueError('BLOCKED_DATASET: explicit '+dataset+' original data binding missing')
    if isinstance(binding,str):binding=read_json(resolved_path(binding,root))
    from pcrepro.data import validate_manifest
    if stopcheck and stopcheck():raise InterruptedError('Safe preflight pause')
    manifest=validate_manifest(binding,root,verify_files=True,stopcheck=stopcheck)
    if manifest.get('dataset')!=dataset:raise ValueError('Wrong dataset manifest')
    # The native FR evaluator must be importable before spending 50K updates.
    from tools.metrics.eval_fr import load_dlpan
    from pcrepro.evaluation import _evaluator_identity
    wald=load_dlpan(os.environ.get('PANCRAFTER_DLPAN',str(Path(root).parent/'DLPan-Toolbox')))
    evaluator=_evaluator_identity(wald)
    import torch
    if device!='cuda' or not torch.cuda.is_available():raise ValueError('Production preflight requires actual CUDA; no CPU substitution')
    gpu=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
        total_memory=torch.cuda.get_device_properties(0).total_memory,torch=torch.__version__,cuda=torch.version.cuda)
    from pcrepro.model import build_model
    model=build_model(SENSORS[dataset]['bands'],seed=2025,max_pixel=SENSORS[dataset]['max_dn'])
    params=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values())
    # two resume states, exact model/fullstate, validation model, two RR/FR outputs.
    expected=state_bytes*15+2*20*SENSORS[dataset]['bands']*(256**2+512**2)*4
    disk=disk_guard(root,expected)
    if not disk['allowed']:raise OSError('INSUFFICIENT_DISK: '+str(disk))
    if stopcheck and stopcheck():raise InterruptedError('Safe preflight pause')
    # A throwaway tiny geometry/gradient smoke. Never an optimizer update,
    # checkpoint or published metric; real training reinitializes from its seed.
    from pcrepro.training import mars_loss
    bands=SENSORS[dataset]['bands'];maximum=SENSORS[dataset]['max_dn']
    model=model.to(device).train()
    raw=torch.rand(1,1,64,64,device=device,dtype=torch.float64)*maximum
    pan=raw.float().mul(2/maximum).sub(1)
    ms=torch.rand(1,bands,16,16,device=device)*2-1
    gt=torch.rand(1,bands,64,64,device=device)*2-1
    loss=mars_loss(model,pan,ms,gt,raw)['loss']
    loss.backward();torch.cuda.synchronize()
    if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Preflight MARs forward/backward nonfinite')
    smoke=dict(passed=True,batch_pairs=1,pan_size=64,ms_size=16,optimizer_updates=0,
        official_metrics=False,full_batch48_memory_verified=False)
    del model,loss,pan,ms,gt,raw
    torch.cuda.empty_cache()
    if stopcheck and stopcheck():raise InterruptedError('Safe preflight pause')
    data_path=folder/'datasets'/f'{dataset}.json'
    immutable_json(data_path,manifest)
    value=dict(complete=True,dataset=dataset,server=server,recipe_sha256=RECIPE_SHA256,
        data_manifest=str(data_path),data_sha=object_sha(manifest),source_identity=source_identity(root),
        total_params=params,trainable_params=trainable,params_m=params/1e6,estimated_run_bytes=expected,
        gpu=gpu,disk=disk,smoke=smoke,evaluator=evaluator,
        preprocessing_seconds=time.monotonic()-start,at_utc=utcnow())
    atomic_json(folder/'preflight'/f'{dataset}.json',value);return value
