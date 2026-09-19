"""QG40 provenance and immutable artifact helpers, independent of live WV3."""
from pathlib import Path
import contextlib
import fcntl
import json
import os

from fh12.common import (ROOT, atomic_json, before_deadline, check_deadline,
                         object_sha, read_json, resolved_path, sha256, utcnow)

CAMPAIGN_ID = 'PANDA_QG40_20260920_v1'
NUMERICAL_REVISION = 'QG40_SYNC_FREQ_C4_v1'


def read(path, default=None):
    path = Path(path)
    return read_json(path) if path.is_file() else ({} if default is None else default)


def camp(root, server):
    if server not in {'s1', 's2', 's3', 's4', 's5'}:
        raise ValueError('QG40 local server must be s1..s5')
    return Path(root) / 'work_dir/_qg40' / server


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
            raise ValueError(f'Immutable artifact differs; refusing overwrite: {path}')
        return path
    atomic_json(path, value)
    return path


def source_identity(root=ROOT):
    """All imported numerical sources plus full local runtime; no one-file proxy."""
    from fh12.common import source_identity as legacy_identity
    root = Path(root)
    result = legacy_identity(root)
    paths = sorted((root / 'qg40').glob('*.py'))
    paths += sorted((root / 'tools').glob('qg40_*.py'))
    paths += sorted((root / 'tools/metrics').glob('*.py'))
    paths += [root / 'tools/repair_qb_ms.py'] if (root / 'tools/repair_qb_ms.py').is_file() else []
    paths += [root / name for name in ('qg40/sensor_sources.json', 'qg40/SENSOR_SOURCE_EVIDENCE.md')
              if (root / name).is_file()]
    for path in paths:
        if path.name.startswith('test_') or path.name.endswith('_tests.py'):
            continue
        result['files'][str(path.relative_to(root))] = sha256(path)
    external = Path(os.environ.get('PANCRAFTER_DLPAN', str(root.parent / 'DLPan-Toolbox')))
    wald = external / '01-DL-toolbox(Pytorch)' / 'UDL' / 'pansharpening' / 'models' / 'APNN' / 'wald_utilities.py'
    result['files']['external/DLPan/wald_utilities.py'] = sha256(wald) if wald.is_file() else 'UNAVAILABLE'
    result.update(content_sha256=object_sha(result['files']),
                  numeric_method_revision=NUMERICAL_REVISION,
                  consumer_campaign=CAMPAIGN_ID)
    return result


def numerical_compatibility(origin, consumer):
    """Different local HEAD/path is not a waiver of numerical/runtime equality."""
    keys = ('files', 'content_sha256', 'numeric_method_revision', 'torch', 'numpy',
            'scipy', 'skimage', 'cuda', 'cudnn', 'tf32_matmul', 'tf32_cudnn')
    differences = [key for key in keys if origin.get(key) != consumer.get(key)]
    if differences:
        raise ValueError('QG40 reference numerical/runtime mismatch: ' + ', '.join(differences))


def load_checkpoint_model(cfg, directory, device='cpu', expected_source=None):
    from safetensors.torch import load_file
    from qg40.model import build_model
    directory = Path(directory)
    identity = read_json(directory / 'identity.json')
    weights = directory / 'model.safetensors'
    if identity.get('model_sha256') != sha256(weights) or identity.get('config_sha256') != object_sha(cfg):
        raise ValueError('QG40 checkpoint bytes/config mismatch')
    if expected_source is not None and identity.get('source_identity') != expected_source:
        raise ValueError('QG40 checkpoint execution release mismatch')
    field, args = cfg['qg40'], cfg['model_args']
    state = load_file(str(weights), device='cpu')
    aligner = {k[len('aligner.'):]: v for k, v in state.items() if k.startswith('aligner.')}
    model, _ = build_model(field['input_layout'], args['hidden_size'], args['depth'], cfg['seed'],
                          role=field['role'], num_bands=field['num_bands'],
                          teacher_aligner_state=aligner if field['role'] == 'S' else None)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), identity


def append_event(path, event, **details):
    path = Path(path)
    row = dict(campaign_id=CAMPAIGN_ID, event=event, at_utc=utcnow(), **details)
    with locked(path.with_suffix('.lock')):
        with path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return row
