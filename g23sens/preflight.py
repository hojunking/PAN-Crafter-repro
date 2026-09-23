"""Actual-device acceptance before BASE admission, isolated from experimental RNG."""
from pathlib import Path

from g23sens.common import (ROOT,camp,read,atomic_json,object_sha,source_identity,
                           apply_runtime_policy,utcnow,selected_gpu_uuid)
from g23sens.plan import verify_sources,make_case,build_config


def run_preflight(root=ROOT,server='s4',device='cuda',overrides=None):
    from g23sens.assets import prepare_bindings
    from g23sens.training import smoke_validate
    import torch
    root=Path(root);verify_sources(root);apply_runtime_policy(root)
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA smoke is required before admission')
    bindings=prepare_bindings(root,server,overrides)
    source=source_identity(root);folder=camp(root,server)
    expected=dict(source_identity=source,bindings_sha256=object_sha(bindings),
        gpu_uuid=selected_gpu_uuid(),device_name=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version())
    previous=read(folder/'preflight.json')
    if previous.get('status')=='PASSED' and all(previous.get(k)==v for k,v in expected.items()):
        return previous
    case=make_case(server,0,'BASE');cfg=build_config(case,root,bindings)
    path=folder/'preflight/config.json';atomic_json(path,cfg)
    numerical=smoke_validate(path,root=root,device=device)
    from g23sens.model import initial_snapshot,build_model
    from g23sens.data import NativeDataset
    from g23sens.evaluation import evaluate_checkpoint
    from kdv.teacher_assets import load_run_model,freeze
    from model.pancrafter_paper import PANCrafterPaper
    from g23sens.diagnostics import isolated_probe
    teacher,_=load_run_model(bindings['teacher']['run_path'],'best_hqnr',PANCrafterPaper,
                            bindings['teacher']['checkpoint_sha256'])
    freeze(teacher)
    from copy import deepcopy
    smoke_cfg=deepcopy(cfg);smoke_cfg['seed']=numerical['smoke_seed']
    model=build_model(smoke_cfg,initial_snapshot(smoke_cfg,teacher)).to(device).eval()
    datasets={key:NativeDataset(bindings,key,root=root) for key in ('rr','fr')}
    # The actual native RR/FR populations, with the very same U+A for both.
    # This is an isolated random initialization smoke, never a result row.
    from g23sens.model import state_hash
    with isolated_probe(model,teacher):
        evaluation=evaluate_checkpoint(model,datasets,device,state_hash(model.state_dict()))
    result=dict(status='PASSED',at_utc=utcnow(),**expected,numerical=numerical,
        native_evaluation=dict(rr_n=evaluation['rr']['n_scenes'],fr_n=evaluation['fr']['n_scenes'],
            protocol=evaluation['metadata']['protocol'],checkpoint_sha256=evaluation['metadata']['checkpoint_sha256']),
        smoke_only=True,production_updates=0)
    atomic_json(folder/'preflight.json',result)
    return result
