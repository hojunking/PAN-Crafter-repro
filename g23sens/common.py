"""Local ownership, immutable identities and pause policy (no time budget)."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_POLICY = dict(precision='fp32', matmul_allow_tf32=False,
    cudnn_allow_tf32=True, cudnn_benchmark=False, cudnn_deterministic=False,
    num_threads=1)


class RuntimePaused(RuntimeError):
    """Recoverable safe pause, never a completed training result."""


def verify_server(server):
    if server not in ('s4', 's5'):
        raise ValueError('G23 sensitivity owns only s4 and s5; s1-s3 are protected')
    return server


def camp(root, server):
    return Path(root)/'work_dir/g23sens'/verify_server(server)


def run_dir(run_id_or_case, root=ROOT, attempt=0):
    from g23sens.plan import case_for, validate_case
    case = case_for(run_id_or_case) if isinstance(run_id_or_case, str) else run_id_or_case
    validate_case(case)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError('Invalid attempt')
    return camp(root, case['server'])/'runs'/case['run_id']/f'attempt{attempt:03d}'


def cycle_dir(root, server, cycle):
    from g23sens.plan import seed_for
    seed_for(cycle)
    return camp(root, server)/'cycles'/f'C{cycle:06d}'


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


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
    value=Path(path).read_text()
    # JSON's exponent syntax (1e-06) is not YAML1.1's float syntax in PyYAML.
    try:return json.loads(value)
    except json.JSONDecodeError:return yaml.safe_load(value)


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
        try: os.fsync(parent)
        finally: os.close(parent)
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
        try: fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError: raise RuntimePaused('Local owner is already active: '+str(path))
        try: yield stream
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def source_identity(root=ROOT):
    """Transitive computational sources, including the external MTF dependency."""
    root = Path(root)
    names = set()
    for directory in ('g23sens', 'kdv', 'pa', 'model', 'feeders', 'fh12', 'qg40', 'pcrepro'):
        for path in (root/directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.json') and not path.name.startswith('test_'):
                names.add(str(path.relative_to(root)))
    for pattern in ('tools/g23sens_*', 'tools/metrics/*.py'):
        names.update(str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file())
    names.update(('main.py', 'train_kdv.py', 'tools/eval_dlpan.py',
        'gspread/gspread_upload.py', 'gspread/sheet_categories.py', 'reporting_extra/sensor_sheet.py',
        'reporting_extra/sensor_layout.py', 'reporting_extra/sensor_backfill.py',
        'config/PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S1234_FRESH50_v2.yaml'))
    bundle = root/'research_log/PANDA_G23_SENS_S45_UNLIMITED_2026-09-23'
    names.update(str(p.relative_to(root)) for p in bundle.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts)
    files = {name: sha256(root/name) for name in sorted(names)}
    dlpan = Path(os.environ.get('PANCRAFTER_DLPAN', str(root.parent/'DLPan-Toolbox')))
    wald = dlpan/'01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py'
    files['external/DLPan/wald_utilities.py'] = sha256(wald) if wald.is_file() else 'UNAVAILABLE'
    try:
        release = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root,
            text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError: release = 'UNCOMMITTED'
    import importlib.metadata
    versions = {}
    for name in ('torch','numpy','scipy','scikit-image','h5py','safetensors','diffusers','timm','PyYAML'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'UNAVAILABLE'
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
    """Resolve one local physical GPU without initializing a supervisor CUDA context."""
    target=os.environ.get('PANCRAFTER_G23SENS_GPU_UUID') or os.environ.get('CUDA_VISIBLE_DEVICES','0')
    import re
    if not re.fullmatch(r'(?:\d+|GPU-[A-Za-z0-9-]+)',target):
        raise ValueError('One explicit GPU is required, not a list or CPU fallback')
    value=subprocess.check_output(['nvidia-smi','-i',target,'--query-gpu=uuid',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+',value):raise ValueError('GPU UUID is not unique')
    return value


def check_runtime(case, root=ROOT, update=None):
    from g23sens.plan import validate_case
    validate_case(case)
    if (camp(root,case['server'])/'STOP_NOW_SAFE').exists():
        raise RuntimePaused('STOP_NOW_SAFE requested; retain full-state cursor')
    from g23sens.resources import ensure_space
    ensure_space(camp(root,case['server']))
    return False
