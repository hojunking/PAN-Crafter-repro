"""Same-state RR20/FR20 bridge; the existing metric functions are not changed."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
import time

import numpy as np
import torch

from directshift48.evaluation import expected_ids, metric_functions_for, metric_values
from panda_method.calibration import canonical_sha256, state_dict_sha256
from ta2.evaluation import evaluator_identity, JQM_VARIANT
from .spec import Stage2Spec
from .training import rng_state, restore_rng, SELECTION_RULE
from .selection import SelectionPolicy


def model_identity(trainer, sources, evaluator_pin):
    trainer.assert_contract()
    if sources.sensor != trainer.stage.spec.sensor or sources.manifest_sha('train') != \
            trainer.stage.donor_binding['source_manifest_sha256']:
        raise ValueError('Evaluation sensor/train-source differs from the bound Stage 2')
    return {'method_version': trainer.stage.spec.method_version, 'completed_update': trainer.updates,
            'student_state_sha256': state_dict_sha256(trainer.stage.student),
            'aligner_state_sha256': state_dict_sha256(trainer.stage.aligner),
            'training_contract_sha256': canonical_sha256(trainer.contract()),
            'checkpoint_selection_sha256': canonical_sha256(trainer.selection_policy.manifest()),
            'eval_protocol_sha256': canonical_sha256(evaluator_pin),
            'rr_source_sha256': sources.manifest_sha('rr'), 'fr_source_sha256': sources.manifest_sha('fr')}


@contextmanager
def paused(trainer):
    # Reuse only the pure optimizer/tree hash, not the old method's A/T-specific
    # evaluation identity or loss-dependent qualification rules.
    from donor24.evaluation import _tree_hash
    trainer.assert_contract()
    stage = trainer.stage
    modes = [(m, m.training) for m in stage.modules()]
    gradients = [(p, p.requires_grad, None if p.grad is None else p.grad.detach().clone())
                 for p in stage.parameters()]
    state, update, rng = _tree_hash(stage.state_dict()), trainer.updates, rng_state()
    optimizer = _tree_hash(trainer.optimizer.state_dict())
    try:
        stage.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        for parameter, requires_grad, grad in gradients:
            parameter.requires_grad_(requires_grad)
            parameter.grad = grad
        restore_rng(rng)
        if (trainer.stage is not stage or trainer.updates != update or
                _tree_hash(stage.state_dict()) != state or _tree_hash(trainer.optimizer.state_dict()) != optimizer):
            trainer.poisoned = True
            raise RuntimeError('Evaluation changed model, optimizer or update; result invalid')


def validate_result(record):
    if record.get('schema') != 'PANDA_S2_CONDQ_OFFICIAL_EVAL_V1':
        raise ValueError('New Stage 2 evaluation schema required')
    states = [record.get(k) for k in ('state_before', 'rr_state', 'fr_state', 'state_after')]
    if not isinstance(states[0], dict) or not all(s == states[0] for s in states[1:]):
        raise ValueError('Evaluation did not use the same Student/Aligner state')
    pipeline = record.get('pipeline', {})
    try:
        declared = Stage2Spec(**{key: pipeline[key] for key in
            ('sensor', 'variant', 'seed', 'alpha', 'beta', 'gamma', 'delta')},
            mask_policy=pipeline['loss_mask_policy'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('New Stage 2 evaluation needs an explicit valid loss/mask contract') from exc
    if (record.get('method_version') != declared.method_version
            or states[0].get('method_version') != declared.method_version
            or any(pipeline.get(key) != value for key, value in declared.manifest().items())
            or pipeline.get('mask_policy') != declared.mask_policy):
        raise ValueError('Stage 2 evaluation method/loss/mask schema mismatch')
    sensor = record['sensor']
    selection = SelectionPolicy.from_manifest(record.get('checkpoint_selection'))
    if (sensor not in ('WV3', 'GF2', 'QB') or type(record.get('completed_update')) is not int
            or record['completed_update'] < 0 or record['completed_update'] != states[0].get('completed_update')
            or canonical_sha256(record.get('eval_protocol')) != states[0].get('eval_protocol_sha256')
            or record.get('selection_rule') != SELECTION_RULE
            or states[0].get('checkpoint_selection_sha256') != canonical_sha256(selection.manifest())
            or record.get('pipeline', {}).get('sensor') != sensor):
        raise ValueError('Evaluation top-level sensor/update/protocol/selection metadata mismatch')
    for split in ('rr', 'fr'):
        keys = ('HQNR', 'D_s', 'D_lambda', 'JQM') if split == 'fr' else (
            'ERGAS', 'SCC', 'SAM', 'PSNR', 'SSIM', 'Q8' if sensor == 'WV3' else 'Q4', 'RMSE', 'CC')
        rows = record[split]['per_scene']
        expected = [f'{sensor}:{split}:{i}' for i in range(20)]
        if len(rows) != 20 or [r['scene_id'] for r in rows] != expected:
            raise ValueError('Official RR20/FR20 scene inventory missing/duplicated/out of order')
        for key in keys:
            values = [r['metrics'].get(key) for r in rows]
            avg = record[split]['metrics'].get(key)
            if any(type(v) not in (float, int) or not math.isfinite(v) for v in values + [avg]):
                raise ValueError('Incomplete/nonfinite official metric ' + key)
            if not math.isclose(math.fsum(values)/20, avg, rel_tol=0., abs_tol=1e-12):
                raise ValueError('Scene-wise metric mean mismatch ' + key)
    return True


def official_evaluate(trainer, sources, *, evaluator_pin=None, metric_functions=None,
                      evaluator_revision=None, device=None, on_scene=None, deadline_unix=None):
    """No artifact/upload creation here; original PAN/native LMS remain references."""
    injected = metric_functions is not None
    if injected and evaluator_revision != 'TEST':
        raise ValueError('Metric injection is restricted to synthetic TEST evaluation')
    pin = deepcopy(evaluator_pin if evaluator_pin is not None else trainer.evaluator_pin)
    if pin is None:
        pin = evaluator_identity()
    if trainer.evaluator_pin is not None and trainer.evaluator_pin != pin:
        raise ValueError('Evaluator pin differs from the strict trainer contract')
    functions = metric_functions if injected else metric_functions_for(sources, pin)
    if set(functions) != {'rr', 'fr'}:
        raise ValueError('RR and FR functions both required')
    device = device or next(trainer.stage.student.parameters()).device
    started = time.monotonic()
    record = {'schema': 'PANDA_S2_CONDQ_OFFICIAL_EVAL_V1',
        'method_version': trainer.stage.spec.method_version,
        'sensor': sources.sensor, 'completed_update': trainer.updates, 'rr': {}, 'fr': {},
        'test_aware_selection': True, 'selection_rule': SELECTION_RULE,
        'checkpoint_selection': trainer.selection_policy.manifest(),
        'metric_function_injection': injected, 'jqm_variant': JQM_VARIANT,
        'official_reference': 'original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21',
        'eval_protocol': pin, 'pipeline': deepcopy(trainer.stage.pipeline_manifest()),
        'weights_saved': False}
    with paused(trainer):
        record['state_before'] = model_identity(trainer, sources, pin)
        for split in ('rr', 'fr'):
            rows = []
            for i, sample_id in enumerate(expected_ids(sources, split)):
                if deadline_unix is not None and time.time() >= deadline_unix:
                    raise TimeoutError('Incomplete evaluation: no partial official result accepted')
                batch = sources.batch(split, [i], device=device)
                if tuple(batch.lineage.sample_ids) != (sample_id,):
                    raise ValueError('Official source row/sample identity mismatch')
                side = 256 if split == 'rr' else 512
                if not injected and batch.pan.shape[-2:] != (side, side):
                    raise ValueError('Official native RR/FR grid dimensions differ')
                out = trainer.stage.native_prediction(batch)
                raw = ((out['y'][0].float() + 1) * (sources.peak / 2)).cpu().numpy()
                # Deliberately NOT out['pan_aligned']: evaluator sees original
                # PAN / native LMS / LRMS and its established clipping/crops.
                arrays = {k: v[0] for k, v in sources.raw(split, [i]).items()}
                metrics = functions[split](raw, arrays)
                row = {'scene_id': sample_id, 'metrics': metric_values(metrics, split, sources.bands),
                    'correction_dy_dx': out['u'][0].cpu().tolist(),
                    'raw_prediction_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'raw_min': float(raw.min()), 'raw_max': float(raw.max()),
                    'jqm_weights': np.asarray(metrics.get('jqm_weights', [])).tolist(),
                    'jqm_weight_source': metrics.get('jqm_weight_source')}
                rows.append(row)
                if on_scene is not None:
                    on_scene({'split': split, **row})
            record[split] = {'per_scene': rows,
                'metrics': {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']},
                'std': {k: float(np.std([r['metrics'][k] for r in rows], ddof=1)) for k in rows[0]['metrics']}}
            record[split + '_state'] = model_identity(trainer, sources, pin)
        record['state_after'] = model_identity(trainer, sources, pin)
    if not injected and evaluator_identity() != pin:
        raise ValueError('Official evaluator changed during evaluation')
    validate_result(record)
    record.update(elapsed_seconds=time.monotonic()-started, same_state_verified=True, metrics_complete=True)
    print(f"[Stage2:{trainer.stage.spec.variant}] step={trainer.updates} "
          f"HQNR={record['fr']['metrics']['HQNR']:.6f} "
          f"SCC={record['rr']['metrics']['SCC']:.6f} ERGAS={record['rr']['metrics']['ERGAS']:.6f}", flush=True)
    return record


def checkpoint_rank(record, *, expected_policy=None):
    validate_result(record)
    if record.get('metric_function_injection') or not record.get('same_state_verified'):
        raise ValueError('Synthetic or unverified metrics cannot select an official checkpoint')
    policy = SelectionPolicy.from_manifest(record['checkpoint_selection'])
    if expected_policy is not None and policy != expected_policy:
        raise ValueError('Cannot compare checkpoints under different HQNR targets')
    return policy.rank(record['fr']['metrics']['HQNR'], record['rr']['metrics']['ERGAS'],
                       record['completed_update'])
