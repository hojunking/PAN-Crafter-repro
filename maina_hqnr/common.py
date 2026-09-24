"""Local ownership and immutable provenance. No cross-server coordination."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_POLICY = dict(precision='fp32', matmul_allow_tf32=False,
    cudnn_allow_tf32=True, cudnn_benchmark=False, cudnn_deterministic=False,
    num_threads=1)


class RuntimePaused(RuntimeError):
    """Recoverable local safe pause, never completion or numerical divergence."""


def verify_server(server):
    if server not in ('s4', 's5'):
        raise ValueError('MAIN-A owns only s4/s5; s1/s2/s3 are protected')
    return server


def camp(root, server):
    return Path(root)/'work_dir/maina_hqnr'/verify_server(server)


def run_dir(case, root=ROOT, attempt=0):
    from maina_hqnr.plan import case_for, validate_case
    case = case_for(case) if isinstance(case, str) else case
    validate_case(case)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError('Invalid attempt')
    return camp(root, case['server'])/'runs'/case['run_id']/f'attempt{attempt:03d}'


def cycle_dir(root, server, cycle):
    from maina_hqnr.plan import seed_for
    seed_for(server, cycle)
    return camp(root, server)/'cycles'/f'C{cycle:06d}'


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def object_sha(value):
    # Deliberately same canonical JSON convention as original fh12.common.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def read(path, default=None):
    return read_json(path, {} if default is None else default)


def read_config(path):
    import yaml
    value = Path(path).read_text()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return yaml.safe_load(value)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False)+'\n'
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
        parent = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Immutable identity changed: '+str(path))
    else:
        atomic_json(path, value)
    return value


@contextmanager
def locked(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimePaused('Local owner is already active: '+str(path)) from None
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_identity(root=ROOT):
    root = Path(root); names = set()
    for directory in ('maina_hqnr', 'fh12', 'fh20r1', 'pa', 'model'):
        names.update(str(p.relative_to(root)) for p in (root/directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.json', '.csv')
                     and not p.name.startswith('test_'))
    for pattern in ('tools/maina_hqnr*', 'tools/metrics/*.py'):
        names.update(str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file())
    names.update(name for name in ('tools/eval_dlpan.py', 'tools/repair_lpan.py',
        'gspread/gspread_upload.py', 'gspread/sheet_categories.py',
        'reporting_extra/sensor_sheet.py', 'reporting_extra/sensor_layout.py')
        if (root/name).is_file())
    files = {name: sha256(root/name) for name in sorted(names)}
    dlpan = Path(os.environ.get('PANCRAFTER_DLPAN', str(root.parent/'DLPan-Toolbox')))
    wald = dlpan/'01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py'
    files['external/DLPan/wald_utilities.py'] = sha256(wald) if wald.is_file() else 'UNAVAILABLE'
    try:
        release = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        release = 'UNCOMMITTED'
    versions = {}
    for name in ('torch','numpy','scipy','scikit-image','h5py','safetensors','diffusers','timm','PyYAML'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'UNAVAILABLE'
    return dict(files=files, content_sha256=object_sha(files), git_release=release,
                packages=versions, runtime_policy=RUNTIME_POLICY)


def apply_runtime_policy(root=ROOT):
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    return dict(RUNTIME_POLICY)


def selected_gpu_uuid():
    import re
    target = os.environ.get('PANCRAFTER_MAINA_GPU_UUID') or os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    if not re.fullmatch(r'(?:\d+|GPU-[A-Za-z0-9-]+)', target):
        raise ValueError('One explicit GPU is required; no CPU fallback or GPU list')
    value = subprocess.check_output(['nvidia-smi', '-i', target, '--query-gpu=uuid',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+', value):
        raise ValueError('GPU UUID is not unique')
    return value


def check_runtime(case, root=ROOT, update=None):
    from maina_hqnr.plan import validate_case
    validate_case(case)
    folder = camp(root, case['server'])
    if (folder/'STOP_NOW_SAFE').exists():
        raise RuntimePaused('STOP_NOW_SAFE requested; preserve full-state and cursor')
    parent = folder
    while not parent.exists(): parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < 8*1024**3:
        raise RuntimePaused(f'STORAGE_PAUSE: at least 8 GiB free required; available={free}')
    return False
