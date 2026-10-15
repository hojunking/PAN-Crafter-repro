"""Append-only s1/R0 exception amendments; never rewrite registration/history."""
from __future__ import annotations

import os
from pathlib import Path

from g20.common import (CAMPAIGN_ID, camp, immutable_json, locked, object_sha,
                        read_json, sha256, source_identity, utcnow)


SCHEMA = 'G20_S1_R0_SOURCE_AMENDMENT_v1'
SCHEMA_V2 = 'G20_S1_R0_SOURCE_AMENDMENT_v2'
BLOCKED_REASON = 'ValueError: Local reference import failed; inspect reference_import.log'
ALLOWED_PATHS = frozenset({
    'g20/gpu_exception.py', 'g20/migration.py', 'g20/references.py',
    'g20/controller.py', 'g20/preflight.py', 'g20/upload.py', 'g20/metric_exception.py',
})
ARCHIVE_NAMES = ('status.json', 'preflight.json', 'method_regression.json',
                 'phase_coordinates.json')
RUNTIME_KEYS = ('torch', 'numpy', 'scipy', 'skimage', 'cuda', 'cudnn',
                'tf32_matmul', 'tf32_cudnn', 'numeric_method_revision',
                'consumer_campaign')


def _version(root, server):
    from g20.gpu_exception import authorization_path
    return 2 if authorization_path(root, server).name.endswith('_v2.json') else 1


def amendment_path(root, server='s1', *, version=None):
    version = _version(root, server) if version is None else version
    if version not in (1, 2):
        raise ValueError('Unknown source amendment revision')
    directory = 'source_amendment_v2' if version == 2 else 'source_amendment'
    return camp(root, server) / directory / 'receipt.json'


def _scope(server):
    if server != 's1':
        raise ValueError('Source amendment is authorized only for s1/R0')


def _source_changes(original, current):
    for label, source in (('registered', original), ('current', current)):
        files = source.get('files')
        if (not isinstance(files, dict) or not files
                or any(not isinstance(name, str) or not isinstance(value, str)
                       or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                       for name, value in files.items())
                or source.get('content_sha256') != object_sha(files)):
            raise ValueError('Invalid ' + label + ' source content hash map')
    if any(key not in original or key not in current for key in RUNTIME_KEYS):
        raise ValueError('Source amendment requires complete original runtime provenance')
    mutable = {'files', 'content_sha256', 'git_release'}
    if ({k: v for k, v in original.items() if k not in mutable}
            != {k: v for k, v in current.items() if k not in mutable}):
        raise ValueError('Source amendment cannot change runtime or numerical method')
    old, new = original['files'], current['files']
    changed = {name: dict(original_sha256=old.get(name), current_sha256=new.get(name))
               for name in sorted(old.keys() | new.keys()) if old.get(name) != new.get(name)}
    if (not changed or not set(changed).issubset(ALLOWED_PATHS)
            or any(row['current_sha256'] is None for row in changed.values())):
        raise ValueError('Source amendment changes files outside the audited G20 exception scope')
    if not {'g20/gpu_exception.py', 'g20/migration.py'}.issubset(new):
        raise ValueError('Source amendment exception implementation missing')
    return changed


def _binding(root, server, registration, current, *, version=1):
    from g20.controller import shared_window
    from g20.gpu_exception import AUTH_SCHEMA, GPU_NAME, authorization_path, policy_for
    from g20.plan import R0_PINS, registry_sha256

    _scope(server)
    folder = camp(root, server)
    if read_json(folder / 'registration.json') != registration:
        raise ValueError('Original registration changed during source amendment')
    original_sha = sha256(folder / 'registration.json')
    identity = registration['identity']
    if (identity.get('campaign_id') != CAMPAIGN_ID or identity.get('server_id') != server
            or identity.get('device') != 'cuda'
            or identity.get('window') != shared_window(root).to_dict()
            or identity.get('registry_sha256') != registry_sha256()
            or identity.get('sensor_spec_sha256') != object_sha(read_json(folder / 'sensor_spec.json'))):
        raise ValueError('Source amendment cannot change registration, clock, registry or sensor')
    changes = _source_changes(identity['source_identity'], current)
    auth_path = authorization_path(root, server, version=version)
    auth = read_json(auth_path)
    expected = dict(schema=AUTH_SCHEMA, campaign_id=CAMPAIGN_ID, server_id=server,
                    reference_id='R0', original_registration_sha256=original_sha,
                    consumer_source_identity=current, gpu_name=GPU_NAME,
                    teacher_checkpoint_sha256=R0_PINS['teacher_sha256'],
                    q_cache_sha256=R0_PINS['q_cache_sha256'],
                    origin_manifest_sha256=R0_PINS['calibration_id'],
                    policy=policy_for(root, server, version=version))
    if (any(auth.get(key) != value for key, value in expected.items())
            or auth.get('user_approved') is not True or auth.get('approval_text') != '승인한다'):
        raise ValueError('Source amendment lacks exact s1/R0 user authorization')
    result = dict(schema=SCHEMA if version == 1 else SCHEMA_V2,
                campaign_id=CAMPAIGN_ID, server_id=server,
                reference_id='R0', original_registration_sha256=original_sha,
                original_identity=identity, consumer_source_identity=current,
                changed_files=changes, authorization_sha256=sha256(auth_path),
                training_changed=False, clock_changed=False,
                origin_artifacts_changed=False)
    if version == 2:
        from g20.gpu_exception import supplemental_authorization
        binding = supplemental_authorization(root, server, auth)
        prior_path = amendment_path(root, server, version=1)
        prior = read_json(prior_path)
        prior_source = prior['consumer_source_identity']
        if not _validate_version(root, server, registration, prior_source, version=1):
            raise ValueError('Second source amendment requires an immutable first amendment')
        result.update(binding, previous_amendment_sha256=sha256(prior_path),
                      previous_consumer_source_identity=prior_source)
    return result


def _blocked_snapshot(state):
    if (state.get('campaign_id') != CAMPAIGN_ID or state.get('server_id') != 's1'
            or state.get('status') != 'BLOCKED_INTEGRITY'
            or state.get('reason') != BLOCKED_REASON
            or state.get('runs') != {} or state.get('blocks') != {}
            or state.get('observations') != []
            or state.get('active_processes') or state.get('gpu_processes')):
        raise ValueError('Source amendment requires the empty R0 reference-import integrity block')


def _unstarted(root):
    """A new amendment is impossible once any G20 training artifacts exist."""
    for directory in (Path(root) / 'work_dir').glob('G20*'):
        for path in directory.rglob('*'):
            if (path.is_file() and (path.name in {'training_start_manifest.json', 'training_status.json'}
                                   or path.suffix in {'.safetensors', '.pt', '.pth'})):
                raise ValueError('Source amendment forbidden after G20 training/checkpoint: ' + str(path))


def _readiness(root, identity, previous):
    folder = camp(root, 's1')
    if (previous.get('complete') is not True or previous.get('status') != 'LOCAL_READY'
            or previous.get('server_id') != 's1' or previous.get('device') != 'cuda'
            or previous.get('source_identity') != identity['source_identity']
            or previous.get('sensor_spec_sha256') != identity['sensor_spec_sha256']
            or previous.get('dataset_manifest_sha256') != object_sha(read_json(folder / 'dataset_manifest.json'))):
        raise ValueError('Original local readiness/data provenance does not match registration')


def _immutable_bytes(path, payload):
    """Archive exact historical bytes, including whitespace, with exclusive creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError('Immutable source-amendment archive changed: ' + str(path))


def prepare_amendment(root, server='s1', current_source=None):
    """Publish an immutable receipt after explicit authorization, before unblocking.

    This archives historical readiness/state but does not alter registration,
    preflight, controller status, or any reference/training artifact.
    """
    _scope(server)
    folder = camp(root, server)
    current = source_identity(root) if current_source is None else current_source
    with locked(folder / '.runner.lock'), locked(folder / '.registration.lock'):
        registration = read_json(folder / 'registration.json')
        version = _version(root, server)
        path = amendment_path(root, server)
        if path.exists():
            validate_amendment(root, server, registration, current)
            return read_json(path)
        receipt = _binding(root, server, registration, current, version=version)
        _blocked_snapshot(read_json(folder / 'status.json'))
        _unstarted(root)
        readiness_identity = dict(registration['identity'])
        if version == 2:
            readiness_identity['source_identity'] = receipt['previous_consumer_source_identity']
        _readiness(root, readiness_identity, read_json(folder / 'preflight.json'))
        archives = {}
        for name in ARCHIVE_NAMES:
            original = folder / name
            if original.is_file():
                archive = path.parent / 'archive' / name
                _immutable_bytes(archive, original.read_bytes())
                archives[name] = dict(path='archive/' + name, sha256=sha256(archive))
        receipt.update(archives=archives, dataset_manifest_sha256=sha256(folder / 'dataset_manifest.json'),
                       issued_at_utc=utcnow(), unstarted_at_publication=True)
        immutable_json(path, receipt)
        validate_amendment(root, server, registration, current)
        return receipt


def validate_amendment(root, server, registration, current_source):
    """Return False for no exception; invalid existing evidence always fails closed."""
    return _validate_version(root, server, registration, current_source,
                             version=_version(root, server))


def _validate_version(root, server, registration, current_source, *, version):
    path = amendment_path(root, server, version=version)
    if not path.is_file():
        return False
    _scope(server)
    receipt = read_json(path)
    expected = _binding(root, server, registration, current_source, version=version)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Source amendment receipt/source/authorization changed')
    archives = receipt.get('archives', {})
    if (not {'status.json', 'preflight.json'}.issubset(archives)
            or not set(archives).issubset(ARCHIVE_NAMES)
            or receipt.get('unstarted_at_publication') is not True):
        raise ValueError('Source amendment historical readiness/status archive incomplete')
    for name, row in archives.items():
        archive = path.parent / 'archive' / name
        if (row.get('path') != 'archive/' + name or not archive.is_file()
                or sha256(archive) != row.get('sha256')):
            raise ValueError('Source amendment historical archive bytes changed')
    _blocked_snapshot(read_json(path.parent / 'archive/status.json'))
    readiness_identity = dict(registration['identity'])
    if version == 2:
        readiness_identity['source_identity'] = receipt['previous_consumer_source_identity']
    _readiness(root, readiness_identity, read_json(path.parent / 'archive/preflight.json'))
    if sha256(camp(root, server) / 'dataset_manifest.json') != receipt.get('dataset_manifest_sha256'):
        raise ValueError('Source amendment original dataset manifest changed')
    # Live state may progress only after publication. Its archived predecessor
    # remains pinned, so normal training does not invalidate authorized replay.
    return True


def archive_preflight_for_amendment(root, server, previous, current_source):
    """Verify the exact prior readiness archive before running new-source checks."""
    folder = camp(root, server)
    registration = read_json(folder / 'registration.json')
    if not validate_amendment(root, server, registration, current_source):
        raise ValueError('Existing local preflight source/sensor binding changed')
    archive = amendment_path(root, server).parent / 'archive/preflight.json'
    if (read_json(archive) != previous
            or sha256(folder / 'preflight.json') != sha256(archive)):
        raise ValueError('Only the exact archived original preflight may be regenerated')
    _unstarted(root)
    state = read_json(folder / 'status.json')
    if state.get('runs') != {} or state.get('blocks') != {} or state.get('active_processes'):
        raise ValueError('Amended readiness must be regenerated before any G20 admission')
    return True
