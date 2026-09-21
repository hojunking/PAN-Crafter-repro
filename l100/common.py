"""LOCAL-T immutable identities; no imported G20 runtime exceptions or state."""
from pathlib import Path
import contextlib
import fcntl
import json
import os

from fh12.common import (ROOT, atomic_json, before_deadline, check_deadline,
                         object_sha, read_json, resolved_path, sha256, utcnow)

CAMPAIGN_ID = 'PANDA_GF2_L100_S345_LOCALT_20H_20260921_v2'
NUMERICAL_REVISION = 'QG40_SYNC_FREQ_C4_v1'


def read(path, default=None):
    return read_json(path) if Path(path).is_file() else ({} if default is None else default)


def camp(root, server):
    if server not in ('s3', 's4', 's5'):
        raise ValueError('LOCAL-T only permits s3, s4, s5')
    return Path(root) / 'work_dir/_l100' / server


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Immutable LOCAL-T artifact changed: ' + str(path))
    else:
        atomic_json(path, value)
    return path


def apply_runtime_policy(root=ROOT):
    import torch
    path = Path(root) / 'l100/runtime_policy.json'
    policy = read_json(path)
    if policy['precision'] != 'float32' or policy['mixed_precision'] != 'no':
        raise ValueError('LOCAL-T must retain FP32 / AMP OFF')
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = policy['tf32_matmul']
    torch.backends.cudnn.allow_tf32 = policy['tf32_cudnn']
    torch.backends.cudnn.benchmark = policy['cudnn_benchmark']
    torch.backends.cudnn.deterministic = policy['cudnn_deterministic']
    torch.use_deterministic_algorithms(policy['deterministic_algorithms'])
    return dict(policy=policy, sha256=sha256(path),
                tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def source_identity(root=ROOT):
    from qg40.common import source_identity as base_identity
    import torch
    root = Path(root)
    result = base_identity(root)
    # Only these immutable G20 numerical modules are reused, not its live
    # controller, migration/reference exceptions, preflight or upload logic.
    names = ['g20/data.py', 'g20/model.py', 'g20/evaluation.py', 'g20/plan.py',
             'g20/losses.py', 'g20/postrun.py', 'g20/common.py',
             'reporting_extra/sensor_sheet.py', 'reporting_extra/sensor_layout.py',
             'reporting_extra/sensor_backfill.py',
             'l100/runtime_policy.json']
    from l100.plan import SOURCE_SHAS
    names += list(SOURCE_SHAS)
    paths = [root / name for name in names]
    paths += sorted((root / 'l100').glob('*.py'))
    paths += sorted((root / 'tools').glob('l100_*.py'))
    for path in paths:
        if path.name.startswith('test_'):
            continue
        result['files'][str(path.relative_to(root))] = sha256(path)
    result.update(content_sha256=object_sha(result['files']), consumer_campaign=CAMPAIGN_ID,
                  numeric_method_revision=NUMERICAL_REVISION,
                  runtime_policy_sha256=sha256(root / 'l100/runtime_policy.json'),
                  cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                  cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    return result


def numerical_compatibility(origin, consumer):
    keys = ('files', 'content_sha256', 'numeric_method_revision', 'torch', 'numpy', 'scipy',
            'skimage', 'cuda', 'cudnn', 'tf32_matmul', 'tf32_cudnn', 'runtime_policy_sha256',
            'cudnn_benchmark', 'cudnn_deterministic', 'deterministic_algorithms')
    changed = [key for key in keys if origin.get(key) != consumer.get(key)]
    if changed:
        raise ValueError('LOCAL-T numerical/runtime mismatch: ' + ', '.join(changed))


def load_checkpoint_model(cfg, directory, device='cpu', expected_source=None):
    from safetensors.torch import load_file
    from g20.model import build_model
    directory = Path(directory)
    identity = read_json(directory / 'identity.json')
    weights = directory / 'model.safetensors'
    if identity.get('model_sha256') != sha256(weights) or identity.get('config_sha256') != object_sha(cfg):
        raise ValueError('LOCAL-T checkpoint bytes/config mismatch')
    if expected_source is not None and identity.get('source_identity') != expected_source:
        raise ValueError('LOCAL-T checkpoint execution release mismatch')
    field, args = cfg['l100'], cfg['model_args']
    state = load_file(str(weights), device='cpu')
    aligner = {key[len('aligner.'):]: value for key, value in state.items() if key.startswith('aligner.')}
    model, _ = build_model(field['input_layout'], args['hidden_size'], args['depth'], cfg['seed'],
                          role=field['role'], num_bands=field['num_bands'],
                          teacher_aligner_state=aligner if field['role'] == 'S' else None)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), identity


def append_event(path, event, **details):
    row = dict(campaign_id=CAMPAIGN_ID, event=event, at_utc=utcnow(), **details)
    path = Path(path)
    with locked(path.with_suffix('.lock')):
        with path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return row
