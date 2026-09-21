"""ABLR2 provenance, local leases and non-destructive artifact operations."""
from pathlib import Path
import contextlib
import datetime as dt
import fcntl
import json
import os
import re

from fh12.common import (ROOT, atomic_json, before_deadline, check_deadline,
                         object_sha, read_json, resolved_path, sha256, utcnow)

CAMPAIGN_ID = 'PANDA_ABL_S1WV3_S2QB_ADAPTIVE_LOOP_20260921_v2'
NUMERICAL_REVISION = 'ABLR2_MASKED_CPLUS3_COMPONENT_ROUTING_v1'


def read(path, default=None):
    return read_json(path) if Path(path).is_file() else ({} if default is None else default)


def read_config(path):
    """Immutable configs are JSON; PyYAML 1.1 misreads JSON 1e-08 as text."""
    content=Path(path).read_text()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(content)


def camp(root, server):
    from ablr2.plan import verify_lane
    sensor = verify_lane(server)
    return Path(root) / 'work_dir/ablr2' / sensor / server


def run_dir(run, root=ROOT):
    match = re.fullmatch(r'ABLR2_(WV3|QB)_(s1|s2)_[A-Za-z0-9_]+', run)
    if not match or (match[1], match[2]) not in (('WV3', 's1'), ('QB', 's2')):
        raise ValueError('Unsafe or cross-lane ABLR2 run identifier')
    return camp(root, match[2]) / 'runs' / run


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
    from fh12.data import write_immutable_json
    return write_immutable_json(Path(path), value)


def append_event(path, event, **details):
    row = {**details, 'campaign_id':CAMPAIGN_ID, 'event':event,
           'at_utc':details.get('at_utc',utcnow())}
    path = Path(path)
    with locked(path.with_suffix('.lock')):
        with path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return row


def apply_runtime_policy(root=ROOT):
    import torch
    path = Path(root) / 'ablr2/runtime_policy.json'
    policy = read_json(path)
    if policy['precision'] != 'float32' or policy['mixed_precision'] != 'no':
        raise ValueError('ABLR2 requires FP32, AMP OFF')
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = policy['tf32_matmul']
    torch.backends.cudnn.allow_tf32 = policy['tf32_cudnn']
    torch.backends.cudnn.benchmark = policy['cudnn_benchmark']
    torch.backends.cudnn.deterministic = policy['cudnn_deterministic']
    torch.use_deterministic_algorithms(policy['deterministic_algorithms'])
    return dict(policy=policy, sha256=sha256(path))


def source_identity(root=ROOT):
    from qg40.common import source_identity as base
    import torch
    root = Path(root)
    result = base(root)
    from ablr2.plan import SOURCE_SHAS
    paths = sorted((root / 'ablr2').glob('*.py')) + sorted((root / 'tools').glob('ablr2_*'))
    paths += [root / name for name in SOURCE_SHAS]
    paths += [root / 'ablr2/runtime_policy.json', root / 'ablr2/sensor_sources.json']
    paths += [root / 'reporting_extra/sensor_sheet.py', root / 'reporting_extra/sensor_layout.py',
              root / 'reporting_extra/sensor_backfill.py', root / 'gspread/gspread_upload.py']
    for path in paths:
        if path.is_file() and not path.name.startswith('test_'):
            result['files'][str(path.relative_to(root))] = sha256(path)
    result.update(content_sha256=object_sha(result['files']), consumer_campaign=CAMPAIGN_ID,
        numeric_method_revision=NUMERICAL_REVISION,
        runtime_policy_sha256=sha256(root / 'ablr2/runtime_policy.json'),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    return result


def numerical_compatibility(origin, consumer):
    keys = ('files', 'content_sha256', 'numeric_method_revision', 'torch', 'numpy', 'scipy',
        'skimage', 'cuda', 'cudnn', 'tf32_matmul', 'tf32_cudnn', 'runtime_policy_sha256',
        'cudnn_benchmark', 'cudnn_deterministic', 'deterministic_algorithms')
    changed = [k for k in keys if origin.get(k) != consumer.get(k)]
    if changed:
        raise ValueError('ABLR2 numerical/runtime mismatch: ' + ', '.join(changed))


def load_checkpoint_model(cfg, directory, device='cpu', expected_source=None):
    from safetensors.torch import load_file
    from ablr2.model import build_model
    from ablr2.plan import validate_config
    case = validate_config(cfg)
    directory = Path(directory)
    identity = read_json(directory / 'identity.json')
    weights = directory / 'model.safetensors'
    if identity.get('model_sha256') != sha256(weights) or identity.get('config_sha256') != object_sha(cfg):
        raise ValueError('ABLR2 checkpoint bytes/config mismatch')
    if expected_source is not None and identity.get('source_identity') != expected_source:
        raise ValueError('ABLR2 checkpoint execution release mismatch')
    state = load_file(str(weights), device='cpu')
    aligner = {k[len('aligner.'):]: v for k, v in state.items() if k.startswith('aligner.')}
    component = cfg['ablr2']['component']
    clone = case.role == 'S' and component['aligner'].startswith('CLONE_')
    model, _ = build_model(bands=case.num_bands, seed=cfg['seed'], role=case.role,
        component=component, width=case.width, depth=case.depth,
        teacher_aligner_state=aligner if clone else None)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), identity


class RuntimePaused(RuntimeError):
    """Save a completed-update boundary and return; this is not a failed seed."""


def runtime_context(cfg, root=ROOT, deadline_arg=None, resume=False):
    from ablr2.plan import validate_config
    case = validate_config(cfg)
    lease = read(camp(root, case.server_id) / 'lease.json')
    if (lease.get('campaign_id') != CAMPAIGN_ID or lease.get('server') != case.server_id
            or lease.get('sensor') != case.sensor or not lease.get('expires_utc')
            or not before_deadline(lease['expires_utc'])):
        raise RuntimePaused('WAIT_LEASE: operator must grant/renew the local lease')
    deadline = lease['expires_utc']
    if deadline_arg:
        a = dt.datetime.fromisoformat(str(deadline_arg).replace('Z', '+00:00'))
        b = dt.datetime.fromisoformat(deadline.replace('Z', '+00:00'))
        deadline = min(a, b).isoformat()
    return lease, deadline


def check_runtime(cfg, root=ROOT, completed_update=0):
    lease, deadline = runtime_context(cfg, root)
    field = cfg['ablr2']
    control = read(camp(root, field['server_id']) / 'control.json')
    if control.get('command') == 'STOP_NOW_SAFE':
        raise RuntimePaused('STOP_NOW_SAFE at completed update ' + str(completed_update))
    return deadline


def execution_identity(cfg, root=ROOT):
    """Label-independent numerical identity, after checking consumed reference bytes."""
    from ablr2.plan import validate_config, execution_signature
    from ablr2.references import validate_reference, teacher_endpoint
    case = validate_config(cfg, require_bound=True)
    data = read_json(cfg['ablr2']['dataset_manifest'])
    reference = None
    if case.requires_calibration:
        manifest, _, _, _ = validate_reference(case.reference_id, case.server_id, root, dataset_manifest=data)
        reference = {key:manifest[key] for key in ('teacher_checkpoint_sha256','q_cache_sha256',
                                                  'tau_R','q_ref','s_bar','s_bar_sha256')}
    elif case.requires_teacher:
        _, _, _, identity, _, _ = teacher_endpoint(run_dir(case.teacher_run_id,root),root,case.server_id)
        reference = dict(teacher_checkpoint_sha256=identity['model_sha256'])
    return execution_signature(cfg,source_sha256=source_identity(root)['content_sha256'],
                               data_sha256=object_sha(data),reference_identity=reference)
