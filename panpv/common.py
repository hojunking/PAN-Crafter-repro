"""PANPV provenance utilities. Private scene manifests belong in ignored work_dir."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def load_donor(path, expected_sha, seed=261002):
    """Only an explicitly bound safetensors donor, never a pickled model fallback."""
    from safetensors.torch import load_file
    from deploy.pandep_shared.project.src.pan_shared.model import SingleSensorPLHUNet
    path = Path(path)
    if path.suffix != '.safetensors' or not path.is_file():
        raise ValueError('BLOCKED_DONOR: an existing .safetensors file is required')
    actual = file_sha(path)
    if actual != expected_sha:
        raise ValueError('BLOCKED_DONOR: checkpoint SHA256 mismatch')
    # WV3 is the original donor state-key namespace, NOT the target sensor.
    model = SingleSensorPLHUNet('WV3', width=104, depth=(1, 2, 2), seed=seed)
    model.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    return model
