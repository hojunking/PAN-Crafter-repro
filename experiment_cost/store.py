"""Integrity checked, server-local architecture profile store (no Torch import)."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .measurement import ROOT, architecture_signature, _digest


def _validate_measurement(measurement, signature):
    from .publication import _validated_fields
    _validated_fields(measurement)
    before = measurement.get('model_state_sha256_before')
    isolation = measurement.get('gpu_isolation_assertion')
    if (measurement.get('weight_basis') != 'FRESH_ARCHITECTURE_NOT_CHECKPOINT'
            or type(measurement.get('optimizer_updates')) is not int
            or measurement['optimizer_updates'] != 0
            or not isinstance(before, str) or re.fullmatch(r'[a-f0-9]{64}', before) is None
            or before != measurement.get('model_state_sha256_after')
            or not isinstance(isolation, dict) or isolation.get('server') != signature['server']
            or isolation.get('training_state_retained_in_memory') is not True
            or isolation.get('other_compute_pids') != []):
        raise ValueError('Unproven read-only isolated architecture measurement')
    if (measurement.get('batch') != 1 or measurement.get('precision') != 'FP32'
            or not str(measurement.get('environment', {}).get('device', '')).startswith('cuda')):
        raise ValueError('Production cost profile requires batch1 FP32 CUDA')
    for split, shape in signature['input_shapes'].items():
        observation = measurement['observations'][split]
        if observation.get('pan_shape') != shape:
            raise ValueError('Cost observation shape mismatch: ' + split)


def find_profile(config, *, server, cost_root=None):
    signature = architecture_signature(config)
    if signature['server'] != server:
        raise ValueError('Cost profile server mismatch')
    key = _digest(signature)
    root = Path(cost_root) if cost_root else ROOT/'work_dir/experiment_cost'/server
    path = root/'profiles'/f'{key}.json'
    if not path.exists():
        return None
    envelope = json.loads(path.read_text())
    proof = envelope.pop('profile_sha256', None)
    if proof != _digest(envelope):
        raise ValueError('Cost profile integrity mismatch')
    if (envelope.get('schema_version') != 'COST_SIDECAR_V1'
            or envelope.get('architecture_key') != key or envelope.get('signature') != signature
            or envelope.get('server') != server):
        raise ValueError('Cost profile architecture/source identity mismatch')
    measurement = envelope['measurement']
    _validate_measurement(measurement, signature)
    return {**envelope, 'profile_sha256': proof, 'path': str(path.resolve())}


def save_profile(config, measurement, *, server, cost_root=None):
    from sheet_publish_v2.transport import atomic_json
    signature = architecture_signature(config)
    if signature['server'] != server:
        raise ValueError('Server mismatch')
    _validate_measurement(measurement, signature)
    key = _digest(signature)
    root = Path(cost_root) if cost_root else ROOT/'work_dir/experiment_cost'/server
    path = root/'profiles'/f'{key}.json'
    body = {'schema_version': 'COST_SIDECAR_V1', 'server': server,
            'architecture_key': key, 'signature': signature, 'measurement': measurement,
            'created_at': datetime.now(timezone.utc).isoformat()}
    digest = _digest(body)
    # Keep all measurements; current pointer is atomic and has identical proof.
    atomic_json(root/'archive'/f'{key}-{digest}.json', {**body, 'profile_sha256': digest})
    atomic_json(path, {**body, 'profile_sha256': digest})
    return find_profile(config, server=server, cost_root=root)
