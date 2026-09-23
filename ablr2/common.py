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
    match = re.fullmatch(r'ABLR2_(WV3|QB|GF2)_(s1|s2|s3)_[A-Za-z0-9_]+', run)
    if not match or (match[1], match[2]) not in (('WV3', 's1'), ('QB', 's2'), ('GF2','s3')):
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
    path=Path(path)
    lane=next((p.name for p in path.parents if p.name in ('s1','s2','s3') and p.parent.name in ('WV3','QB','GF2')),None)
    row = {**details, 'campaign_id':campaign_id(lane) if lane else CAMPAIGN_ID, 'event':event,
           'at_utc':details.get('at_utc',utcnow())}
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
    from ablr2.plan import SOURCE_SHAS,source_path,EXTENSION_SHAS,EXTENSION_BUNDLE
    paths = sorted((root / 'ablr2').glob('*.py')) + sorted((root / 'tools').glob('ablr2_*'))
    # Original SHA-pinned documents may have moved to the approved archive.
    # Preserve physical path provenance; never skip a missing required source.
    paths += [source_path(name,root) for name in SOURCE_SHAS]
    paths += [root/name for name in EXTENSION_SHAS]
    paths += [root/EXTENSION_BUNDLE/'SHA256SUMS.txt']
    paths += [root / 'ablr2/runtime_policy.json', root / 'ablr2/sensor_sources.json']
    paths += [root / 'reporting_extra/sensor_sheet.py', root / 'reporting_extra/sensor_layout.py',
              root / 'reporting_extra/sensor_backfill.py', root / 'gspread/gspread_upload.py',
              root / 'gspread/sheet_categories.py', root / 'model/se.py']
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


def campaign_id(server):
    from ablr2.plan import campaign_id as resolve
    return resolve(server)


def grant_until_stop(root,server,*,operator_authorized=False,now=None):
    """Explicit new authorization; neither extend nor replace an original lease."""
    from ablr2.plan import verify_lane,EXTENSION_ID
    sensor=verify_lane(server)
    if operator_authorized is not True:raise PermissionError('Explicit UNTIL_OPERATOR_STOP authorization required')
    current=dt.datetime.fromisoformat(now or utcnow())
    if current.tzinfo is None:raise ValueError('Authorization timestamp requires timezone')
    folder=camp(root,server)
    source=source_identity(root)
    with locked(folder/'authorization.lock'):
        previous=read(folder/'authorization.json')
        if previous and (previous.get('server')!=server or previous.get('campaign_id')!=campaign_id(server)):
            raise ValueError('Existing authorization belongs to another lane')
        if (previous.get('service_mode')=='UNTIL_OPERATOR_STOP' and previous.get('source_identity')==source
                and previous.get('revoked') is False):return previous
        old_lease=read(folder/'lease.json')
        value=dict(schema='ABLR2X_OPERATOR_AUTHORIZATION_v1',extension_id=EXTENSION_ID,
            campaign_id=campaign_id(server),server=server,sensor=sensor,service_mode='UNTIL_OPERATOR_STOP',
            granted_at_utc=current.isoformat(),first_granted_at_utc=previous.get('first_granted_at_utc',current.isoformat()),
            sequence=previous.get('sequence',0)+1,expires_utc=None,max_campaign_cycles=None,
            operator_action=True,automatic_renewal=False,revoked=False,source_identity=source,
            scope='New ABLR2X jobs only; original admitted runs retain original source/lease',
            previous_authorization_sha256=object_sha(previous) if previous else None,
            original_lease=old_lease or None,original_lease_sha256=object_sha(old_lease) if old_lease else None)
        immutable_json(folder/'authorizations'/f'{value["sequence"]:06}.json',value)
        atomic_json(folder/'authorization.json',value)
        append_event(folder/'authorization_ledger.jsonl','OPERATOR_UNTIL_STOP',authorization=value)
        return value


def authorization_context(root,server):
    """No automatic grant, renewal, control reset, or finite→continuous conversion."""
    from ablr2.plan import verify_lane,EXTENSION_ID
    sensor=verify_lane(server);folder=camp(root,server)
    authorization=read(folder/'authorization.json')
    if authorization:
        if (authorization.get('schema')!='ABLR2X_OPERATOR_AUTHORIZATION_v1'
                or authorization.get('extension_id')!=EXTENSION_ID
                or authorization.get('campaign_id')!=campaign_id(server)
                or authorization.get('server')!=server or authorization.get('sensor')!=sensor
                or authorization.get('service_mode')!='UNTIL_OPERATOR_STOP'
                or authorization.get('operator_action') is not True
                or authorization.get('automatic_renewal') is not False
                or authorization.get('expires_utc') is not None
                or authorization.get('max_campaign_cycles') is not None
                or authorization.get('revoked') is not False):
            raise RuntimePaused('WAIT_AUTHORIZATION: invalid or revoked until-stop receipt')
        history=read(folder/'authorizations'/f'{authorization["sequence"]:06}.json')
        if history!=authorization:raise RuntimePaused('WAIT_AUTHORIZATION: immutable receipt differs')
        registered=read(folder/'registration_ablr2x.json')
        if registered and registered.get('source_identity')!=authorization.get('source_identity'):
            raise RuntimePaused('WAIT_AUTHORIZATION: execution source requires explicit new authorization')
        return authorization,None
    lease=read(folder/'lease.json')
    if (lease.get('campaign_id')!=campaign_id(server) or lease.get('server')!=server
            or lease.get('sensor')!=sensor or not lease.get('expires_utc')
            or not before_deadline(lease['expires_utc'])):
        raise RuntimePaused('WAIT_LEASE: explicit finite lease or until-stop authorization required')
    return lease,lease['expires_utc']


def assert_compatible_source(origin,consumer,root,server):
    """A release change requires measured, source-bound parity, not hash exclusions."""
    if origin==consumer:return None
    from ablr2.migration import validate_source_bridge
    return validate_source_bridge(root,server,origin,consumer)


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
    lease,deadline = authorization_context(root,case.server_id)
    if deadline_arg:
        a = dt.datetime.fromisoformat(str(deadline_arg).replace('Z', '+00:00'))
        if a.tzinfo is None:raise ValueError('Deadline needs explicit timezone')
        deadline=min(a,dt.datetime.fromisoformat(deadline.replace('Z','+00:00'))).isoformat() if deadline else a.isoformat()
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
