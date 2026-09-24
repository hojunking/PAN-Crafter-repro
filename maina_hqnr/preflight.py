"""Device-bound acceptance; no remote actions, Teacher regeneration or training run."""
from pathlib import Path

from maina_hqnr.common import (ROOT,camp,read,atomic_json,object_sha,source_identity,
    apply_runtime_policy,utcnow,selected_gpu_uuid,verify_server)
from maina_hqnr.plan import verify_sources


def run_preflight(root=ROOT, server='s4', device='cuda', overrides=None):
    from maina_hqnr.assets import verify_assets
    from maina_hqnr.diagnostics import run_acceptance
    import torch
    root=Path(root); verify_server(server); verify_sources(root)
    apply_runtime_policy(root)
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Production admission requires actual CUDA acceptance; CPU tests are not sufficient')
    bindings=verify_assets(server,root,overrides,persist=True)
    source=source_identity(root); folder=camp(root,server)
    expected=dict(source_identity=source,bindings_sha256=object_sha(bindings),
        gpu_uuid=selected_gpu_uuid(),device_name=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    previous=read(folder/'preflight.json')
    if previous.get('status')=='PASSED' and all(previous.get(k)==v for k,v in expected.items()):
        return previous
    numerical=run_acceptance(root,server,bindings,device=device)
    if numerical.get('status')!='PASSED':
        raise RuntimeError('Actual-device numerical acceptance did not pass')
    result=dict(status='PASSED',at_utc=utcnow(),**expected,numerical=numerical,
        bindings_path=str(folder/'runtime_bindings.json'),smoke_only=True,production_updates=0)
    atomic_json(folder/'preflight.json',result)
    return result


def preflight(server, root=ROOT, overrides=None, device='cuda'):
    return run_preflight(root,server,device,overrides)
