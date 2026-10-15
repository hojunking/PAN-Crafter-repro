"""Explicit s1/R0 RTX 4090 variance acceptance, preserving historical failures.

This does not change the original cache, reference, training, clock, or parity
tolerances. A sealed local measurement may be consumed only after independent
strict checks of the original artifacts and data equality by references.py.
"""
from pathlib import Path
import hashlib

import numpy as np
import torch

from fh12.calibration import AXIS16, axis16_q, _write_npz_immutable
from fh12.common import check_deadline, object_sha, read_json, sha256
from fh12.data import AUGMENTATION
from g20.common import CAMPAIGN_ID, camp, source_identity
from qg40.reference_parity import (ATOL, RTOL, PARITY_COUNT, PARITY_SEED,
                                  identity_for, isolated_teacher, parity_indices)

AUTH_SCHEMA = 'G20_S1_R0_GPU_EXCEPTION_AUTH_v1'
RECEIPT_SCHEMA = 'G20_S1_R0_GPU_EXCEPTION_v1'
GPU_NAME = 'NVIDIA GeForce RTX 4090'
POLICY = dict(max_abs_error=dict(q=5e-5, output=2e-4, delta=5e-5),
              strict_atol=ATOL, strict_rtol=RTOL, q_count=PARITY_COUNT,
              q_views=[0, 1, 2, 3], probe_base_ids=[0, 1], same_device_repeats=2,
              training_changed=False, clock_changed=False, origin_artifacts_changed=False)
POLICY_V2 = dict(POLICY, max_abs_error=dict(q=5e-5, output=2e-4, delta=2e-4))
_PROBE_KEYS = ('pan', 'ms', 'lp', 'output', 'delta')
_ARRAY_KEYS = {'pan', 'ms', 'lp'} | {
    f'{phase}_{key}' for phase in ('historical', 'local', 'repeat')
    for key in ('q', 'output', 'delta')}


def authorization_path(root, server='s1', *, version=None):
    folder = camp(root, server)
    if version is None:
        version = 2 if ((folder / 'metric_exception_authorization.json').exists()
                        or (folder / 'parity_exception_authorization_v2.json').exists()) else 1
    if version not in (1, 2):
        raise ValueError('Unknown GPU exception authorization revision')
    return folder / ('parity_exception_authorization_v2.json' if version == 2
                     else 'parity_exception_authorization.json')


def policy_for(root, server='s1', *, version=None):
    path = authorization_path(root, server, version=version)
    return POLICY_V2 if path.name.endswith('_v2.json') else POLICY


def supplemental_authorization(root, server, auth):
    """Bind the second approval to its immutable first approval and metric grant."""
    from g20.metric_exception import authorization as metric_authorization, authorization_path as metric_path
    previous = authorization_path(root, server, version=1)
    metric = metric_authorization(root, server)
    if (auth.get('previous_authorization_sha256') != sha256(previous)
            or auth.get('metric_authorization_sha256') != sha256(metric_path(root, server))
            or auth.get('consumer_source_identity') != metric.get('consumer_source_identity')):
        raise ValueError('GPU exception supplemental authorization binding changed')
    return dict(previous_authorization_sha256=sha256(previous),
                metric_authorization_sha256=sha256(metric_path(root, server)))


def _gpu(device):
    try:
        if torch.device(device).type != 'cuda':
            raise ValueError('s1 R0 exception measurement/consumption requires CUDA')
        name = torch.cuda.get_device_name(torch.device(device))
    except (AssertionError, RuntimeError, TypeError) as error:
        raise ValueError('Cannot verify s1 R0 exception GPU') from error
    if name != GPU_NAME:
        raise ValueError('s1 R0 exception GPU differs: ' + name)
    return name


def authorization(root, server, origin, device=None):
    """Return a fully bound authorization, or None only when its file is absent.

    CPU report validation omits device. Executable measurement/consumption
    supplies it and must run on the explicitly authorized CUDA GPU.
    """
    from g20.references import R0_PINS
    path = authorization_path(root, server)
    if not path.is_file():
        return None
    auth = read_json(path)
    if (server != 's1' or origin.get('schema') != 'QG40_REFERENCE_v1'
            or origin.get('reference_id') != 'GF2_TA' or origin.get('server') != 's3'
            or origin.get('teacher_checkpoint_sha256') != R0_PINS['teacher_checkpoint_sha256']
            or origin.get('q_cache_sha256') != R0_PINS['q_cache_sha256']
            or object_sha(origin) != R0_PINS['calibration_id']):
        raise ValueError('GPU exception only authorizes the pinned original s1/R0 reference')
    expected = dict(schema=AUTH_SCHEMA, campaign_id=CAMPAIGN_ID, server_id='s1',
        user_approved=True, approval_text='승인한다',
        reference_id='R0', original_registration_sha256=sha256(camp(root, 's1') / 'registration.json'),
        consumer_source_identity=source_identity(root), gpu_name=GPU_NAME,
        teacher_checkpoint_sha256=R0_PINS['teacher_checkpoint_sha256'],
        q_cache_sha256=R0_PINS['q_cache_sha256'], origin_manifest_sha256=object_sha(origin),
        policy=policy_for(root, server))
    if auth.get('user_approved') is not True or any(auth.get(key) != value for key, value in expected.items()):
        raise ValueError('GPU exception authorization identity/policy mismatch')
    if path == authorization_path(root, server, version=2):
        supplemental_authorization(root, server, auth)
    if device is not None:
        _gpu(device)
    return auth


def _q_inputs(q, local, origin):
    q = np.asarray(q)
    count = local['splits']['train']['count']
    if (count < 3072 or q.shape != (count, 4) or list(q.shape) != origin.get('q_shape')
            or q.dtype != np.dtype('float32') or not np.isfinite(q).all() or (q < 0).any()):
        raise ValueError('GPU exception requires the unchanged valid full-train four-view q cache')
    return parity_indices(count)


def _protocol(q):
    ids = parity_indices(len(q))
    return dict(base_count=len(q), seed=PARITY_SEED, indices=ids.tolist(),
        indices_sha256=object_sha(ids.tolist()), views=[0, 1, 2, 3],
        augmentation=AUGMENTATION, augmentation_sha256=object_sha(AUGMENTATION),
        AXIS16=[list(value) for value in AXIS16], units='HR_PAN_PIXELS', atol=ATOL, rtol=RTOL)


def _array_identity(value):
    value = np.asarray(value)
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def probe_identity(probe):
    """Bind a portable probe's actual arrays independently of NPZ serialization."""
    return {key: _array_identity(probe[key]) for key in _PROBE_KEYS}


def verify_probe_binding(receipt, probe):
    """Verify bridge probe bytes against the historical arrays in the seal."""
    values = _probe_inputs(probe)
    path = Path(receipt['measurements_path'])
    if sha256(path) != receipt.get('measurements_sha256'):
        raise ValueError('GPU exception immutable measurement file changed')
    with np.load(path, allow_pickle=False) as arrays:
        if (probe_identity(values) != receipt.get('original_probe_identity')
                or any(not np.array_equal(values[key], arrays[key if key in ('pan', 'ms', 'lp')
                    else 'historical_' + key]) for key in _PROBE_KEYS)):
            raise ValueError('GPU exception original portable probe changed')
    return True


def _probe_inputs(probe, dataset=None):
    if set(probe) != set(_PROBE_KEYS):
        raise ValueError('GPU exception requires a complete original portable probe')
    values = {key: np.asarray(probe[key]) for key in _PROBE_KEYS}
    pan = values['pan']
    if pan.ndim != 4 or pan.shape[:2] != (2, 1) or any(side < 4 or side % 4 for side in pan.shape[2:]):
        raise ValueError('GPU exception probe geometry differs')
    h, w = pan.shape[2:]
    shapes = dict(pan=(2, 1, h, w), ms=(2, 4, h // 4, w // 4),
                  lp=(2, 1, h // 4, w // 4), output=(2, 4, h, w), delta=(2, 2))
    if any(value.shape != shapes[key] or value.dtype != np.dtype('float32')
           or not np.isfinite(value).all() for key, value in values.items()):
        raise ValueError('GPU exception probe shape/dtype/finite check failed')
    if dataset is not None:
        rows = [dataset.base(index) for index in (0, 1)]
        for key, column in (('pan', 4), ('ms', 2), ('lp', 3)):
            expected = torch.stack([row[column] for row in rows]).detach().cpu().numpy()
            if not np.array_equal(values[key], expected):
                raise ValueError('Original probe inputs differ from local train base IDs 0,1')
    return values


def _measure_q(model, dataset, q, device, deadline):
    if (not dataset.has_gt or dataset.base_count != len(q) or len(dataset) != len(q)
            or getattr(model, 'bands', None) != 4):
        raise ValueError('GPU exception q measurement requires the frozen C4 base-train Teacher')
    ids = parity_indices(len(q))
    measured = np.empty((PARITY_COUNT, 4), dtype=np.float32)
    with isolated_teacher(model):
        for view in range(4):
            check_deadline(deadline)
            rows = [dataset.get_view(int(index), view, augment=True) for index in ids]
            if any(torch.as_tensor(row[-1]).cpu().tolist() != [int(index), view, 1, 1]
                   for index, row in zip(ids, rows)):
                raise ValueError('GPU exception q view/base-ID metadata mismatch')
            ms = torch.stack([row[2] for row in rows]).to(device)
            pan = torch.stack([row[4] for row in rows]).to(device)
            online, _, _ = axis16_q(model, pan, ms)
            actual = online.detach().cpu().numpy()
            if actual.shape != (PARITY_COUNT,) or not np.isfinite(actual).all() or (actual < 0).any():
                raise ValueError('GPU exception online q is invalid')
            measured[:, view] = actual
    return measured


def _measure_probe(model, probe, device, deadline):
    check_deadline(deadline)
    with isolated_teacher(model):
        output = model(*[torch.from_numpy(probe[key]).to(device) for key in ('pan', 'ms', 'lp')])
    result = {key: np.asarray(probe[key]).copy() for key in ('pan', 'ms', 'lp')}
    result.update(output=output['y'].detach().cpu().numpy(), delta=output['delta'].detach().cpu().numpy())
    return _probe_inputs(result)


def _comparison(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or not actual.size
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError('GPU exception comparison shape/finite check failed')
    failed = int(np.count_nonzero(~np.isclose(actual, expected, atol=ATOL, rtol=RTOL)))
    return dict(status='FAILED' if failed else 'PASSED', passed=failed == 0,
        failed_count=failed, total_count=int(actual.size),
        max_abs_error=float(np.max(np.abs(actual - expected))), atol=ATOL, rtol=RTOL)


def _summaries(arrays, policy=POLICY):
    if set(arrays) != _ARRAY_KEYS:
        raise ValueError('GPU exception measurement array keys differ')
    for phase in ('historical', 'local', 'repeat'):
        probe = {key: arrays[key] for key in ('pan', 'ms', 'lp')}
        probe.update({key: arrays[phase + '_' + key] for key in ('output', 'delta')})
        _probe_inputs(probe)
        q = arrays[phase + '_q']
        if (q.shape != (PARITY_COUNT, 4) or q.dtype != np.dtype('float32')
                or not np.isfinite(q).all() or (q < 0).any()):
            raise ValueError('GPU exception measured q shape/dtype/finite/sign check failed')
    historical = {key: _comparison(arrays['local_' + key], arrays['historical_' + key])
                  for key in ('q', 'output', 'delta')}
    repeated = {key: _comparison(arrays['repeat_' + key], arrays['local_' + key])
                for key in ('q', 'output', 'delta')}
    historical_repeat = {key: _comparison(arrays['repeat_' + key], arrays['historical_' + key])
                         for key in ('q', 'output', 'delta')}
    if all(row['passed'] for row in historical.values()):
        raise ValueError('GPU exception requires a recorded strict historical FAILED comparison')
    if not all(row['passed'] for row in repeated.values()):
        raise ValueError('GPU exception same-local repeat failed original strict tolerances')
    for comparison in (historical, historical_repeat):
        for key, row in comparison.items():
            if row['max_abs_error'] > policy['max_abs_error'][key]:
                raise ValueError('GPU exception historical variance cap exceeded: ' + key)
    return historical, repeated, historical_repeat


def _bindings(q, origin, local, root, server, device):
    from g20.references import data_signature
    auth = authorization(root, server, origin, device)
    if auth is None:
        raise ValueError('Explicit s1 R0 GPU exception authorization is required')
    _q_inputs(q, local, origin)
    return dict(schema=RECEIPT_SCHEMA, status='ACCEPTED_WITH_PARITY_EXCEPTION',
        passed=False, accepted=True, synthetic_test=False, campaign_id=CAMPAIGN_ID,
        server_id=server, reference_id='R0', gpu_name=GPU_NAME,
        authorization_sha256=sha256(authorization_path(root, server)),
        original_registration_sha256=auth['original_registration_sha256'],
        consumer_source_identity=auth['consumer_source_identity'],
        origin_manifest_sha256=object_sha(origin), reference_identity=identity_for(origin),
        local_data_identity=object_sha(data_signature(local)), policy=auth['policy'], protocol=_protocol(q))


def _measurement_path(root, server):
    return camp(root, server) / 'references/R0/gpu_exception_measurements.npz'


def measure(model, dataset, q, probe, origin, local, root, server, device, deadline=None):
    """Measure twice without altering the historical q cache or portable probe."""
    check_deadline(deadline)
    _gpu(device)
    receipt = _bindings(q, origin, local, root, server, device)
    probe = _probe_inputs(probe, dataset)
    local_q = _measure_q(model, dataset, q, device, deadline)
    local_probe = _measure_probe(model, probe, device, deadline)
    repeat_q = _measure_q(model, dataset, q, device, deadline)
    repeat_probe = _measure_probe(model, probe, device, deadline)
    arrays = {key: probe[key].copy() for key in ('pan', 'ms', 'lp')}
    arrays.update(historical_q=np.asarray(q)[parity_indices(len(q))].copy(),
                  local_q=local_q, repeat_q=repeat_q)
    for key in ('output', 'delta'):
        arrays.update({f'historical_{key}': probe[key].copy(), f'local_{key}': local_probe[key],
                       f'repeat_{key}': repeat_probe[key]})
    historical, repeated, historical_repeat = _summaries(arrays, receipt['policy'])
    check_deadline(deadline)
    path = _measurement_path(root, server)
    _write_npz_immutable(path, arrays)
    receipt.update(measurements_path=str(path), measurements_sha256=sha256(path),
        array_identity={key: _array_identity(value) for key, value in arrays.items()},
        original_probe_identity=probe_identity(probe), local_probe_identity=probe_identity(local_probe),
        historical_comparison=historical, same_local_repeat=repeated,
        historical_repeat_comparison=historical_repeat)
    validate(receipt, q, origin, local, root, server, device)
    return receipt, local_probe


def _read_arrays(receipt, root, server):
    path = _measurement_path(root, server)
    if (Path(receipt.get('measurements_path', '')).resolve() != path.resolve()
            or receipt.get('measurements_sha256') != sha256(path)):
        raise ValueError('GPU exception immutable measurement file changed')
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def validate(receipt, q, origin, local, root, server, device=None):
    """Recompute every reported comparison from immutable measured arrays."""
    expected = _bindings(q, origin, local, root, server, device)
    if (receipt.get('passed') is not False or receipt.get('accepted') is not True
            or receipt.get('synthetic_test') is not False
            or any(receipt.get(key) != value for key, value in expected.items())):
        raise ValueError('GPU exception receipt identity/policy/status mismatch')
    arrays = _read_arrays(receipt, root, server)
    historical, repeated, historical_repeat = _summaries(arrays, expected['policy'])
    original_probe = {key: arrays[key] for key in ('pan', 'ms', 'lp')}
    local_probe = dict(original_probe)
    for key in ('output', 'delta'):
        original_probe[key] = arrays['historical_' + key]
        local_probe[key] = arrays['local_' + key]
    if (receipt.get('array_identity') != {key: _array_identity(value) for key, value in arrays.items()}
            or not np.array_equal(arrays['historical_q'], np.asarray(q)[parity_indices(len(q))])
            or receipt.get('original_probe_identity') != probe_identity(original_probe)
            or receipt.get('local_probe_identity') != probe_identity(local_probe)
            or receipt.get('historical_comparison') != historical
            or receipt.get('same_local_repeat') != repeated
            or receipt.get('historical_repeat_comparison') != historical_repeat):
        raise ValueError('GPU exception receipt measurements or historical failure record differs')
    return receipt


def verify_local(model, dataset, q, original_probe, receipt, origin, local,
                 root, server, device, deadline=None):
    """Require fresh local q/output/c to match the sealed local values strictly."""
    check_deadline(deadline)
    _gpu(device)
    validate(receipt, q, origin, local, root, server, device)
    original_probe = _probe_inputs(original_probe, dataset)
    if probe_identity(original_probe) != receipt['original_probe_identity']:
        raise ValueError('GPU exception original portable probe changed')
    arrays = _read_arrays(receipt, root, server)
    current_q = _measure_q(model, dataset, q, device, deadline)
    current_probe = _measure_probe(model, original_probe, device, deadline)
    measured = dict(q=current_q, output=current_probe['output'], delta=current_probe['delta'])
    comparisons = {key: _comparison(value, arrays['local_' + key]) for key, value in measured.items()}
    if not all(row['passed'] for row in comparisons.values()):
        raise ValueError('GPU exception fresh local replay failed original strict tolerances')
    check_deadline(deadline)
    return dict(schema='G20_R0_LOCAL_GPU_PARITY_v1', status='MEASURED', passed=True,
        scope='fresh local q/output/c versus exception measurements; historical failures retained',
        exception_receipt_sha256=object_sha(receipt), comparisons=comparisons, atol=ATOL, rtol=RTOL)
