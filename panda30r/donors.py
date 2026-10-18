"""Read-only, server-local A1/R1 early/final donor inventory.

Only a named completed local H50 attempt is admitted. An existing artifact with
bad identity, tensors, checksum, source or evaluation fails closed: it is never
silently replaced by a fresh donor. Runtime still replays the full40 evaluation
and performs an independent strict architecture load before using this pair.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DONOR24 = 'PANDA_WV3_DONOR_S13_24H_20261016_v1'
PANDAS4 = 'PANDA_S4_DONOR_STAGE2_40H_20261017_v1'
MASK2 = 'DETACHED_BICUBIC4_SUPPORT_PER_SAMPLE_MEAN_V1'
SEED = 261016101


class DonorInventoryError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise DonorInventoryError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), 'Missing or linked donor metadata: ' + str(path))
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DonorInventoryError('Unreadable donor metadata: ' + str(path)) from exc
    _require(isinstance(result, dict), 'Donor metadata must be an object: ' + str(path))
    return result


def source_run(server):
    if server in ('s1', 's3'):
        return ('donor24', DONOR24,
                f'D24_WV3_{server.upper()}_A1_a2_eps0.003_lrA1e-05_R3_M2_H50_R1_S{SEED}')
    if server == 's4':
        return 'pandas4', PANDAS4, f'P4_WV3_S4_D_A1_R1_S{SEED}'
    raise DonorInventoryError('Only local s1/s3/s4 WV3 donors are registered')


def _expected_train_manifest(root, cfg):
    """Reuse the exact registered manifest formula; runtime verifies H5 bytes."""
    from panda_method.edge_data import EdgeDataSources
    path = Path(root) / 'ablr2/sensor_sources.json'
    registry = _json(path)
    registry_sha = _sha(path)
    _require(cfg['baseline_resolved'].get('source_registry_sha256') == registry_sha,
             'Source registry SHA differs from local registered data')
    # No H5 reads/GPU allocation are needed to audit the saved source binding.
    sources = EdgeDataSources.__new__(EdgeDataSources)
    sources.sensor, sources.catalog_sha256 = 'WV3', registry_sha
    sources.entry, sources.lms_overrides = registry['sensors']['WV3'], {}
    return sources.manifest_sha('train')


def _config(directory, server, root):
    cfg = _json(directory / 'config.json')
    _, campaign, run_id = source_run(server)
    expected = {'campaign_id': campaign, 'run_id': run_id, 'server': server,
                'dataset': 'WV3', 'phase': 'donor', 'replica': 'R1', 'seed': SEED,
                'case': 'D_A1' if server == 's4' else 'A1',
                'planned_horizon': 50000, 'loss_mask_policy': MASK2,
                'lambda_abs': 2., 'lambda_epsilon': .003, 'lr_A': 1e-5,
                'lr_U': 1e-4}
    _require(all(cfg.get(key) == value for key, value in expected.items()),
             'Donor configuration is not the exact local A1/R1/H50 recipe')
    _require(type(cfg.get('attempt')) is int and cfg['attempt'] > 0 and
             directory.name == f"attempt_{cfg['attempt']:02d}", 'Donor attempt mismatch')
    baseline = cfg.get('baseline_resolved', {})
    _require(baseline.get('aligner_architecture') == {'name': 'DirectPANAligner', 'view_margin': 8}
             and baseline.get('teacher_architecture') == {'bands': 8, 'width': 112, 'depth': [1, 2, 3]}
             and baseline.get('teacher_input_layout') == 'P0'
             and baseline.get('teacher_base') == 'BICUBIC_MS', 'Unexpected donor architecture/input layout')
    state = _json(directory / 'run_state.json')
    _require(state.get('status') == 'COMPLETED' and state.get('step') == 50000
             and all(state.get(key) == cfg[key] for key in ('run_id', 'attempt', 'phase')),
             'Donor run did not complete its registered H50')
    return cfg, _expected_train_manifest(root, cfg)


def _rank(row):
    h, e, step = row.get('HQNR'), row.get('ERGAS'), row.get('completed_update')
    _require(isinstance(h, (int, float)) and not isinstance(h, bool) and math.isfinite(h) and 0 <= h <= 1
             and isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(e) and e >= 0
             and type(step) is int and 0 < step <= 50000, 'Invalid donor selection metrics/update')
    return (1, -e, h, -step) if h >= .959 else (0, h, -e, -step)


def _selected(registry, cfg):
    saved = registry.get('saved')
    _require(isinstance(saved, list) and saved, 'Saved donor candidate registry missing')
    selected = (registry.get('best_selected') or registry.get('best_gate_ergas') or registry.get('best_hqnr'))
    _require(isinstance(selected, dict), 'No HQNR-gated donor selection pointer')
    _require(selected in saved and _rank(selected) == max(_rank(row) for row in saved),
             'Donor selection pointer differs from HQNR-gated minimum-ERGAS ranking')
    _require(selected['completed_update'] < 50000, 'No distinct early donor checkpoint retained')
    for row in saved:
        _require(all(row.get(key) is True for key in
                     ('checkpoint_verified', 'metrics_complete', 'metrics_finite', 'same_state_verified')),
                 'Saved donor candidate lacks verified qualification')
        _require(all(row.get(key) == cfg[key] for key in ('run_id', 'attempt', 'seed')),
                 'Saved donor candidate belongs to a different run/attempt/seed')
    return selected


def _tensor_hash(stage):
    from panda_method.calibration import canonical_sha256, state_dict_sha256
    import torch
    _require(isinstance(stage, dict), 'Checkpoint stage state missing')
    roles = {}
    for prefix, role in (('aligner.', 'A'), ('teacher.', 'T')):
        tensors = {key[len(prefix):]: value for key, value in stage.items() if key.startswith(prefix)}
        _require(tensors and all(isinstance(x, torch.Tensor) and bool(torch.isfinite(x).all())
                                for x in tensors.values()), 'Missing or nonfinite A/T tensor state')
        roles[role] = state_dict_sha256(tensors)
    _require(all(key == '_extra_state' or key.startswith(('aligner.', 'teacher.')) for key in stage),
             'Checkpoint is not a single native A/T donor')
    return canonical_sha256(roles), roles


def _candidate(directory, cfg, train_manifest, step, path, proof=None):
    import torch
    from panda40.stage1 import _check_evaluation
    _require(path.is_file() and not path.is_symlink() and path.parent == directory / 'models',
             'Missing, linked or foreign donor checkpoint: ' + str(path))
    evaluation_path = directory / f'evaluation_{step:07d}.json'
    evaluation = _json(evaluation_path)
    identity = {key: cfg[key] for key in ('run_id', 'attempt', 'seed', 'dataset', 'server', 'phase', 'campaign_id')}
    identity['completed_update'] = step
    _check_evaluation(evaluation, identity)
    _require(evaluation.get('full_method_eligible') is True and
             evaluation.get('metric_function_injection') is False and
             evaluation.get('measurement_domain') == 'OFFICIAL_FULL20_NATIVE' and
             evaluation.get('official_reference') ==
             'original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21',
             'Candidate must come from the official full-method evaluator')
    for split in ('rr', 'fr'):
        _require({row['scene_id'] for row in evaluation[split]['per_scene']} ==
                 {f'WV3:{split}:{index}' for index in range(20)},
                 'Donor evaluation does not contain the exact registered WV3 scene IDs')
    before, sha = path.stat(), _sha(path)
    expected_sha = None if proof is None else proof.get('file_sha256')
    if proof is not None:
        _require(isinstance(expected_sha, str) and re.fullmatch('[0-9a-f]{64}', expected_sha)
                 and sha == expected_sha, 'Donor file differs from its saved SHA proof')
    try:
        # These are only the exact named, local campaign checkpoints, not downloads.
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as exc:
        raise DonorInventoryError('Unreadable local donor checkpoint: ' + str(path)) from exc
    _require(before == path.stat() and sha == _sha(path), 'Donor file changed during inventory')
    _require(payload.get('schema') == 'DONOR30_CHECKPOINT_V1' and payload.get('config') == cfg,
             'Donor payload schema/config differs from local run provenance')
    trainer = payload.get('trainer', {})
    _require(trainer.get('updates') == step, 'Donor optimizer update differs from evaluated step')
    stage = trainer.get('stage', {})
    actual_hash, roles = _tensor_hash(stage)
    expected_hash = evaluation['state_before']['model_state_sha256']
    _require(actual_hash == expected_hash == payload.get('state_hash'),
             'Donor A/T tensors differ from the official evaluated state')
    payload_identity = payload.get('identity', {})
    _require(all(payload_identity.get(key) == value for key, value in
                 {key: identity[key] for key in ('run_id', 'attempt', 'seed', 'completed_update')}.items())
             and payload_identity.get('model_state_sha256') == actual_hash,
             'Donor payload identity differs from evaluation')
    sources = stage.get('_extra_state', {}).get('sources', {})
    _require(sources == {'sensor': 'WV3', 'rr_manifest_sha256': train_manifest},
             'Donor training source binding differs from registered WV3 RR')
    if payload.get('evaluation') is not None:
        _check_evaluation(payload['evaluation'], {**identity, 'model_state_sha256': actual_hash})
        _require(all(payload['evaluation'][split] == evaluation[split] for split in ('rr', 'fr')),
                 'Embedded donor evaluation differs from recorded full40 evidence')
    if proof is not None:
        proved_hash = proof.get('state_hash', proof.get('model_state_sha256'))
        _require(proved_hash == actual_hash and proof.get('completed_update') == step,
                 'Saved donor checksum proof belongs to another state/update')
        for key, split in (('HQNR', 'fr'), ('ERGAS', 'rr'), ('SCC', 'rr')):
            if key in proof:
                _require(proof[key] == evaluation[split]['metrics'][key], 'Saved donor scalar differs: ' + key)
    metrics = {'HQNR': evaluation['fr']['metrics']['HQNR'],
               'ERGAS': evaluation['rr']['metrics']['ERGAS'], 'SCC': evaluation['rr']['metrics']['SCC']}
    _rank({**metrics, 'completed_update': step})
    return {**metrics, 'completed_update': step, 'checkpoint_path': str(path),
            'file_sha256': sha, 'checkpoint_sha256': sha, 'state_hash': actual_hash,
            'model_state_sha256': actual_hash, 'source_config': deepcopy(cfg),
            'evaluation_path': str(evaluation_path), 'source_server': cfg['server'],
            'source': 'EXACT_COMPLETED_LOCAL_A1_R1_EARLY_FINAL_PAIR',
            'run_id': cfg['run_id'], 'attempt': cfg['attempt'], 'seed': cfg['seed'],
            'source_train_manifest_sha256': train_manifest, 'role_state_sha256': roles,
            'checkpoint_verified': True, 'same_state_verified': True,
            'metrics_complete': True, 'metrics_finite': True,
            'sha_provenance': ('SAVED_SHA_AND_OFFICIAL_TENSOR_STATE_VERIFIED' if proof is not None
                               else 'COMPUTED_AFTER_OFFICIAL_TENSOR_STATE_VERIFIED'),
            'reloaded_full40_verified': False,
            'requires_runtime_full40_replay': True}


def validate_pair(directory, server, *, root=ROOT):
    """Audit one explicit local completed attempt without mutating it."""
    directory, root = Path(directory).absolute(), Path(root).resolve(strict=True)
    package, campaign, run_id = source_run(server)
    expected_parent = root / 'work_dir' / package / campaign / server / 'WV3/runs' / run_id
    _require(directory.parent == expected_parent and directory.resolve(strict=True) == directory,
             'Donor attempt must belong to the exact local server/run, without symlink traversal')
    _require(not (directory / 'models').is_symlink(), 'Donor models directory may not be a symlink')
    cfg, manifest = _config(directory, server, root)
    registry = _json(directory / 'models/selection.json')
    early_row = _selected(registry, cfg)
    early = _candidate(directory, cfg, manifest, early_row['completed_update'],
                       directory / 'models' / f"step_{early_row['completed_update']:07d}.pt", early_row)
    late_row = next((row for row in registry['saved'] if row['completed_update'] == 50000), None)
    if late_row is not None:
        late_path, proof = directory / 'models/step_0050000.pt', late_row
    elif (directory / 'models/final_0050000.pt').exists():
        # pandas4 final A/T payload has no SHA sidecar. Its exact tensors must
        # independently match the official 50K evaluation before pinning SHA.
        late_path, proof = directory / 'models/final_0050000.pt', None
    else:
        proof = _json(directory / 'models/latest_resume.json')
        raw = proof.get('checkpoint_path')
        _require(isinstance(raw, str) and re.fullmatch(r'resume_step_0050000_[0-9]+\.pt', Path(raw).name),
                 'Local latest resume does not name the final 50K A/T checkpoint')
        late_path = directory / 'models' / Path(raw).name
    late = _candidate(directory, cfg, manifest, 50000, late_path, proof)
    return {'early': early, 'late': late}


def discover_pair(server, *, root=ROOT):
    """Return None only when no completed named local H50 run is available."""
    root = Path(root).resolve(strict=True)
    package, campaign, run_id = source_run(server)
    run = root / 'work_dir' / package / campaign / server / 'WV3/runs' / run_id
    if not run.exists():
        return None
    _require(run.resolve(strict=True) == run, 'Local donor run may not traverse a symlink')
    completed = []
    for directory in sorted(run.glob('attempt_*')):
        _require(directory.is_dir() and re.fullmatch(r'attempt_[0-9]+', directory.name),
                 'Invalid named donor attempt path')
        state = _json(directory / 'run_state.json')
        if state.get('status') == 'COMPLETED':
            completed.append(directory)
    if not completed:
        return None
    # Prefer latest completed attempt, never best cross-attempt metrics.
    chosen = max(completed, key=lambda path: int(path.name.split('_')[1]))
    try:
        return validate_pair(chosen, server, root=root)
    except (KeyError, TypeError, OSError, ValueError) as exc:
        if isinstance(exc, DonorInventoryError):
            raise
        raise DonorInventoryError('Existing local donor pair failed validation: ' + str(chosen)) from exc
