"""Atomic local artifacts, explicit execution identity, no legacy campaign imports."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import socket
from fh12.common import ROOT,atomic_json,read_json,object_sha,sha256,utcnow,resolved_path
from pcrepro.plan import CAMPAIGN_ID,RECIPE,RECIPE_SHA256,verify_server,case_for,SOURCE_PLAN

def read(path,default=None):return read_json(path) if Path(path).is_file() else ({} if default is None else default)
def read_config(path):
    text=Path(path).read_text()
    try:return json.loads(text)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(text)
def camp(root,server):verify_server(server);return Path(root)/'work_dir/_pcrepro'/server
def run_dir(case,root=ROOT):return Path(root)/'work_dir'/case_for(case).run_id
def machine_lock(kind='runner'):
    if kind not in ('runner','worker'):raise ValueError('Unknown local lock kind')
    return Path('/tmp')/f'pcrepro-{os.getuid()}-{socket.gethostname()}-{kind}.lock'
@contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield stream
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
def immutable_json(path,value):
    if Path(path).exists():
        if read_json(path)!=value:raise ValueError('Immutable reproduction artifact changed: '+str(path))
    else:atomic_json(path,value)
    return Path(path)
def apply_runtime_policy(root=ROOT):
    import torch
    policy=RECIPE['runtime']
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=policy['tf32_matmul']
    torch.backends.cudnn.allow_tf32=policy['tf32_cudnn']
    torch.backends.cudnn.benchmark=policy['cudnn_benchmark']
    torch.backends.cudnn.deterministic=policy['cudnn_deterministic']
    torch.use_deterministic_algorithms(policy['deterministic_algorithms'])
    return dict(policy=policy,recipe_sha256=RECIPE_SHA256)
def source_identity(root=ROOT):
    import torch,numpy,scipy,skimage
    root=Path(root)
    names=['fh12/common.py','fh12/upload.py','fh12/plan.py',
        'model/pancrafter_paper.py','model/pancrafter.py','model/swin.py',
        'gspread/gspread_upload.py','gspread/sheet_categories.py','reporting_extra/sensor_backfill.py',
        'tools/eval_dlpan.py',SOURCE_PLAN]
    paths=[root/n for n in names]+list((root/'pcrepro').glob('*.py'))+list((root/'pcrepro').glob('*.json'))
    paths+=list((root/'tools').glob('pcrepro_*'))+list((root/'tools/metrics').glob('*.py'))
    files={str(p.relative_to(root)):sha256(p) for p in sorted(paths) if not p.name.startswith('test_')}
    wald=Path(os.environ.get('PANCRAFTER_DLPAN',str(root.parent/'DLPan-Toolbox')))/'01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py'
    files['external/DLPan/wald_utilities.py']=sha256(wald) if wald.is_file() else 'UNAVAILABLE'
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.SubprocessError):commit='UNAVAILABLE'
    from importlib.metadata import version
    packages={name:version(name) for name in ('h5py','safetensors','timm','gspread','PyYAML')}
    runtime_readback=dict(dtype=str(torch.get_default_dtype()),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    return dict(files=files,content_sha256=object_sha(files),git_release=commit,recipe_sha256=RECIPE_SHA256,
        torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,skimage=skimage.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        runtime_policy=RECIPE['runtime'],runtime_readback=runtime_readback,packages=packages)
def append_event(path,event,**details):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with locked(path.with_suffix('.lock')):
        with path.open('a') as f:
            value=dict(at_utc=utcnow(),**{k:v for k,v in details.items() if k!='at_utc'})
            value.update(event=event,at_utc=details.get('at_utc',value['at_utc']))
            f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
