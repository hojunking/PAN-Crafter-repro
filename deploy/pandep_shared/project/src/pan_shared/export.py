"""Candidate-only inference bundles; no optimizer, teacher, data or metric runtime."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from .common import CAMPAIGN, atomic_json, canonical_sha, file_sha, timestamp
from .evaluate import assert_checkpoint_matches_model, preserved_inference
from .frontend import LP_RECIPE, SENSOR_PROFILES, make_lpan, normalize_dn

INFERENCE_FILES = ('__init__.py', 'model.py', 'blocks.py', 'frontend.py', 'common.py')


def export_candidate(model, checkpoint_identity, run, work_root, selector_scope, *, device='cpu',
                     provenance=None, project_root=None):
    """Export C00/R01 EXACT150K or its first-stage validation-selected whole model.

    Caller loads selected weight bytes first. Neither architecture flags nor a
    passed tensor hash can stand in for comparing the actual stored tensors.
    """
    if (run['case_id'], run['repeat'], run['mode']) != ('C00', 1, 'SHARED'):
        raise ValueError('Only C00 R01 shared first-stage deployment candidates are authorized')
    if set(run['sensors']) != set(SENSOR_PROFILES):
        raise ValueError('Candidate must contain all three sensor branches')
    if run.get('width') != 104 or run.get('depth') != [1, 2, 2] or run.get('train_fraction') != 1.:
        raise ValueError('C00 export must preserve W104/D122/full-data architecture')
    step = checkpoint_identity['completed_step']
    if selector_scope == 'EXACT':
        if step != 150000:
            raise ValueError('Initial EXACT export requires matched global150K')
    elif selector_scope != 'BEST_JOINT_VAL_TO_B0003' or not 0 < step <= 150000:
        raise ValueError('Initial BEST export must be validation-only through block3')
    required = ('source_sha256', 'config_sha256', 'data_sha256', 'subset_sha256', 'protocol_sha256')
    provenance = dict(provenance or {})
    if any(not isinstance(provenance.get(k), str) or len(provenance[k]) != 64 for k in required):
        raise ValueError('Complete source/config/data/subset/protocol identities required')
    source = Path(checkpoint_identity['model_path']).resolve()
    actual_sha = file_sha(source)
    if checkpoint_identity['checkpoint_sha256'] != actual_sha:
        raise ValueError('BLOCKED_IDENTITY: actual model bytes SHA mismatch')
    assert_checkpoint_matches_model(model, source)
    project = Path(project_root or Path(__file__).resolve().parents[2])
    destination_root = Path(work_root).resolve() / 'exports'
    destination_root.mkdir(parents=True, exist_ok=True)
    release_id = f"C00_R01_{selector_scope}_{actual_sha[:16]}"
    destination = destination_root / release_id
    if destination.exists():
        manifest = json.loads((destination/'release_manifest.json').read_text())
        if (manifest['checkpoint_sha256'] != actual_sha or manifest['selector_scope'] != selector_scope
                or manifest['provenance'] != provenance):
            raise ValueError('Existing immutable candidate has different provenance')
        for line in (destination/'SHA256SUMS.txt').read_text().splitlines():
            sha, relative = line.split('  ', 1)
            if file_sha(destination/relative) != sha:
                raise ValueError('Existing candidate bundle is corrupt')
        return destination
    staging = Path(tempfile.mkdtemp(prefix='.'+release_id+'.', dir=destination_root))
    # Failures leave an inspectable scratch bundle; never delete historical candidate material.
    shutil.copy2(source, staging/'model.safetensors')
    architecture = model.architecture_manifest()
    if (architecture['width'] != run['width'] or architecture['depth'] != run['depth']
            or architecture['sensors'] != run['sensors']):
        raise ValueError('BLOCKED_IDENTITY: model architecture differs from exported run config')
    atomic_json(staging/'model_config.json', {'width': run['width'], 'depth': run['depth'],
                'sensors': run['sensors'], 'seed': run['seed'], 'architecture': architecture})
    atomic_json(staging/'sensor_profiles.json', SENSOR_PROFILES)
    atomic_json(staging/'preprocessing.json', {'normalization': '2*DN/max_dn-1',
                'lp': LP_RECIPE, 'inputs': ['PAN', 'LPAN_UP', 'PAN_MINUS_LPAN_UP', 'MS_UP'],
                'resize': 'torch bicubic align_corners=False scale_factor4',
                'output': 'MS_UP+residual_once; unclipped normalized model result',
                'tile_inference_validated': False, 'finite_zero_is_valid': True})
    code = staging/'source_snapshot'/'pan_shared'; code.mkdir(parents=True)
    for name in INFERENCE_FILES:
        shutil.copy2(project/'src'/'pan_shared'/name, code/name)
    licenses = staging/'LICENSES'; licenses.mkdir()
    shutil.copy2(project/'vendor_reference'/'LICENSES'/'PAN-Crafter-LICENSE.txt', licenses/'PAN-Crafter-LICENSE.txt')
    (staging/'reference_inputs').mkdir(); (staging/'reference_outputs').mkdir()
    refs = {}
    with preserved_inference(model):
        for index, sensor in enumerate(SENSOR_PROFILES):
            rng = np.random.default_rng(271001 + index)
            profile = SENSOR_PROFILES[sensor]
            pan = rng.uniform(0, profile['max_dn'], (1, 1, 64, 64)).astype(np.float32)
            ms = rng.uniform(0, profile['max_dn'], (1, profile['bands'], 16, 16)).astype(np.float32)
            native_lp = make_lpan(pan)
            arrays = dict(pan=normalize_dn(pan, sensor), ms=normalize_dn(ms, sensor),
                          lpan=normalize_dn(native_lp, sensor))
            inputs = staging/'reference_inputs'/f'{sensor}.npz'
            np.savez(inputs, **arrays)
            result = model(sensor, *(torch.from_numpy(arrays[k]).to(device) for k in ('pan', 'ms', 'lpan')))
            if not torch.isfinite(result).all():
                raise FloatingPointError('Nonfinite reference export inference')
            output = staging/'reference_outputs'/f'{sensor}.npz'
            np.savez(output, output=result.cpu().numpy())
            refs[sensor] = {'inputs_sha256': file_sha(inputs), 'outputs_sha256': file_sha(output),
                            'fixture': 'synthetic seeded DN; not benchmark/private training data',
                            'shape': list(result.shape)}
    manifest = {'schema': 'PANDEP_CANDIDATE_EXPORT_v1', 'campaign_id': CAMPAIGN,
                'release_id': release_id, 'status': 'candidate', 'run_id': run['run_id'],
                'checkpoint_sha256': actual_sha, 'tensor_state_sha256': checkpoint_identity['tensor_state_sha256'],
                'completed_step': step, 'selector_scope': selector_scope, 'provenance': provenance,
                'reference_fixtures': refs, 'architecture': architecture, 'created_at': timestamp(),
                'public_release_authorized': False, 'commercial_release_authorized': False,
                'license_notice': 'PAN-Crafter license includes non-commercial research/educational limitation; permissions must be separately confirmed',
                'excluded': ['optimizer', 'teacher', 'benchmark/private datasets', 'metrics', 'GPL DLPan evaluation utilities'],
                'tile_inference_validated': False, 'laptop_performance_measured': False}
    atomic_json(staging/'release_manifest.json', manifest)
    (staging/'README.md').write_text(
        '# Deployment PoC candidate\n\nNot a production or public/commercial release approval.\n'
        'The model is one shared trunk and three sensor-specific stems/heads.\n'
        'Use model_config.json, sensor_profiles.json, and preprocessing.json together.\n'
        'Load model.safetensors into SharedPLHUNet; select WV3/GF2/QB explicitly.\n'
        'References are synthetic and normalized; model returns unclipped normalized output.\n'
        'Large-raster tiling, global phase/halo correctness and laptop latency remain unvalidated.\n'
        'See LICENSES before any redistribution. No training data/optimizer is included.\n', encoding='utf-8')
    files = sorted(p for p in staging.rglob('*') if p.is_file())
    (staging/'SHA256SUMS.txt').write_text(''.join(file_sha(p)+'  '+str(p.relative_to(staging))+'\n' for p in files), encoding='utf-8')
    for p in list(files)+[staging/'SHA256SUMS.txt']:
        with p.open('rb') as handle:
            os.fsync(handle.fileno())
    os.rename(staging, destination)
    return destination
