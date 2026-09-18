"""Small shared provenance and atomic-artifact helpers for FH12 only."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def resolved_path(path, root=ROOT):
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def before_deadline(deadline):
    if not deadline:
        return True
    end = dt.datetime.fromisoformat(str(deadline).replace('Z', '+00:00'))
    if end.tzinfo is None:
        raise ValueError('FH12 deadline must include a timezone')
    return dt.datetime.now(dt.timezone.utc) < end


def check_deadline(deadline):
    if not before_deadline(deadline):
        raise TimeoutError('FH12 immutable wall-clock deadline reached')


def source_identity(root=ROOT):
    """Bind new trainer, inference, metrics and local library versions, not git alone."""
    import torch
    import numpy
    import scipy
    import skimage
    root = Path(root)
    paths = [str(p.relative_to(root)) for p in sorted((root / 'fh12').glob('*.py'))]
    paths += ['model/pancrafter_paper.py', 'model/swin.py', 'model/pancrafter.py',
              'pa/aligner.py', 'pa/warp.py', 'pa/losses.py', 'pa/offset.py',
              'tools/repair_lpan.py', 'tools/eval_dlpan.py',
              'tools/metrics/eval_rr.py', 'tools/metrics/eval_fr.py', 'tools/metrics/q2n.py']
    files = {p: sha256(root / p) for p in paths if (root / p).is_file()}
    try:
        release = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        release = 'unavailable'
    return dict(files=files, content_sha256=object_sha(files), git_release=release,
                torch=torch.__version__, numpy=numpy.__version__, scipy=scipy.__version__,
                skimage=skimage.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                tf32_cudnn=bool(torch.backends.cudnn.allow_tf32))


def load_checkpoint_model(cfg, directory, device='cpu', expected_source=None):
    """All FH12 inference paths use the same wrapper and strict state loading."""
    from safetensors.torch import load_file
    from fh12.model import build_model
    directory = Path(directory)
    identity = read_json(directory / 'identity.json')
    if identity['model_sha256'] != sha256(directory / 'model.safetensors'):
        raise ValueError('FH12 checkpoint bytes do not match identity')
    if identity['config_sha256'] != object_sha(cfg):
        raise ValueError('FH12 checkpoint belongs to a different resolved config')
    if expected_source is not None and identity.get('source_identity') != expected_source:
        raise ValueError('FH12 checkpoint belongs to a different source/runtime release')
    state = load_file(str(directory / 'model.safetensors'))
    f = cfg['fh12']; m = cfg['model_args']
    aligner = {k[len('aligner.'):]: v for k, v in state.items() if k.startswith('aligner.')}
    model, _ = build_model(f['input_layout'], int(m['hidden_size']), m['depth'],
                           int(cfg['seed']), role=f['role'],
                           teacher_aligner_state=aligner if f['role'] == 'S' else None)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), identity
