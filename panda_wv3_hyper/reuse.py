"""Audited numerical reuse of exactly matched, already completed RGB controls.

Original records keep their original run/config identity.  A new receipt states
why their numerical observations can serve as controls; it never pretends that
previously pruned weights were recovered or that a new inference was performed.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from directshift48.reporting import atomic_json
from ta2.evaluation import evaluator_identity, file_sha
from . import plan, assets

CONTROL_MAP = {
    's1': {'H2_REF': 'M0_OURS', 'H2_A0125': 'M2_A0125', 'H2_G0125': 'M1_G0125'},
    's3': {'H1_REF': 'S1_A5_ALL', 'H1_SPA0': 'S1_A4_REC_EPS', 'H1_EPS0': 'S1_A3_REC_SPA'},
}
COMMON_SCIENCE = ('server', 'stage', 'dataset', 'repeat', 'seed', 'stream_seeds',
    'architecture', 'horizon', 'primary_step', 'lr', 'lr_aligner', 'weight_decay',
    'betas', 'adam_eps', 'batch', 'microbatch', 'warmup', 'min_lr', 'upsampler',
    'epsilon_radius', 'precision', 'augmentation', 'source_registry',
    'source_registry_sha256', 'evaluation_steps', 'validation_steps',
    'inference_tta', 'beta', 'rotation_consistency', 'alignment', 'warp')
STAGE1_SCIENCE = ('lambda_abs', 'lambda_epsilon', 'alignment_target',
    'reference_bands', 'validation_reference')
STAGE2_SCIENCE = ('alpha', 'gamma', 'scale', 'teacher_warp', 'student_input',
    'weight', 'donor_run_id')
_HASH_CACHE = {}
_TRACE_CACHE = {}


def _read(path):
    return json.loads(Path(path).read_text())


def _signature(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sha(path):
    """Avoid rehashing gigabytes at every Sheet poll; mutation invalidates cache."""
    path = Path(path)
    key = (str(path.resolve()), _signature(path))
    value = _HASH_CACHE.get(key)
    if value is None:
        value = file_sha(path)
        if _signature(path) != key[1]:
            raise ValueError('Evidence file changed while hashing: ' + str(path))
        _HASH_CACHE[key] = value
    return value


def _relative(path):
    return str(Path(path).resolve().relative_to(plan.ROOT.resolve()))


def _trace(path, horizon):
    key = (str(Path(path).resolve()), _signature(path), horizon)
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]
    total = 0.
    with Path(path).open() as handle:
        count = 0
        for count, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get('step') != count:
                raise ValueError('Source training trace has missing/reordered steps')
            seconds = float(row['seconds'])
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid source training time')
            total += seconds
    if count != horizon:
        raise ValueError('Source training trace is not complete at its registered horizon')
    result = {'path': _relative(path), 'sha256': _sha(path),
        'updates': count, 'optimizer_wall_seconds': total,
        'scope': 'SUM_OF_RECORDED_TRAIN_UPDATE_SECONDS_EXCLUDES_EVALUATION'}
    _TRACE_CACHE[key] = result
    return result


def source_spec(cfg):
    """Only the six declared controls have an eligible historical counterpart."""
    source_case = CONTROL_MAP.get(cfg['server'], {}).get(cfg['case_id'])
    if source_case is None:
        return None
    if cfg['server'] == 's1':
        from panda_wv3_main_rgb import plan as old_plan, evaluation as old_eval
    else:
        from panda_wv3_ablation_rgb import plan as old_plan, evaluation as old_eval
    return old_plan, old_eval, old_plan.resolve(cfg['server'], source_case, cfg['repeat'])


def _compare_science(cfg, old):
    keys = COMMON_SCIENCE + (STAGE1_SCIENCE if cfg['stage'] == 'stage1' else STAGE2_SCIENCE)
    different = [k for k in keys if k not in cfg or k not in old or cfg[k] != old[k]]
    if different:
        raise ValueError('Historical control science differs: ' + ', '.join(different))
    if cfg.get('weight_averaging', False) or old.get('weight_averaging', False):
        raise ValueError('Weight averaging is not part of this matched control')


def audit_control(cfg, *, pins=None):
    """Return a deterministic receipt; raise on missing, tampered or unlike evidence."""
    plan.validate(cfg)
    source = source_spec(cfg)
    if source is None:
        return None
    old_plan, old_eval, expected_cfg = source
    _compare_science(cfg, expected_cfg)
    root = old_plan.campaign_root(cfg['server'])
    directory = root / 'runs' / expected_cfg['run_id']
    old_cfg = _read(directory / 'config.json')
    if old_cfg != expected_cfg:
        raise ValueError('Stored historical config differs from its original registry')
    state = _read(root / 'campaign_state.json')
    jobs = [r for r in state['runs'] if r['run_id'] == old_cfg['run_id']]
    if (state.get('campaign_id') != old_plan.CAMPAIGN_ID or state.get('server') != cfg['server']
            or len(jobs) != 1 or jobs[0]['status'] != 'COMPLETED'
            or jobs[0]['step'] != cfg['horizon']):
        raise ValueError('Historical control is not a completed local run')
    old_pins = _read(root / 'admission.json')
    old_release = _read(old_plan.CONFIG / 'release.json')
    if old_release['source_sha256'] != old_pins['source_release']:
        raise ValueError('Historical admission and release identity differ')
    if old_pins['evaluator'] != evaluator_identity():
        raise ValueError('Historical and current official evaluator differ')
    if pins is not None and any(old_pins[k] != pins[k]
            for k in ('evaluator', 'train_inventory', 'split_manifests')):
        raise ValueError('Historical control and current campaign data/evaluator differ')
    donor_receipt = calibration_meta = None
    if cfg['stage'] == 'stage2':
        _, donor_receipt, donor_pins, calibration_meta = assets.source_evidence(cfg)
        if donor_pins != old_pins:
            raise ValueError('Historical Student and donor admission differ')
    else:
        from panda_wv3_ablation_rgb import selection
        inventory = _read(root / 'validation_inventory.json')
        if inventory.get('inventory_sha256') != plan.digest(
                {k: v for k, v in inventory.items() if k != 'inventory_sha256'}):
            raise ValueError('Historical validation inventory hash differs')
    records = []
    # Requiring the full scheduled curve preserves the representative rule as
    # well as ensuring the primary fixed endpoint cannot be selected post hoc.
    for step in cfg['evaluation_steps']:
        evpath = directory / f'evaluation_{step:07d}.json'
        record = _read(evpath)
        old_eval.verify_record(record, old_cfg)
        if record['step'] != step or record['eval_protocol_id'] != plan.digest(old_pins['evaluator']):
            raise ValueError('Historical evaluation step/protocol differs')
        meta = record['checkpoint']
        checkpoint = directory / 'models' / f'step_{step:07d}.pt'
        if (meta.get('step') != step or meta.get('config_sha256') != old_cfg['config_sha256']
                or meta.get('model_sha256') != record['model_sha256']
                or old_plan.local_path(meta['path']).resolve() != checkpoint.resolve()):
            raise ValueError('Historical checkpoint identity does not match its evaluation')
        if checkpoint.exists():
            if (_read(checkpoint.with_suffix('.json')) != meta
                    or _sha(checkpoint) != meta['file_sha256']):
                raise ValueError('Retained historical checkpoint SHA/sidecar differs')
            status = 'VERIFIED_RETAINED_WEIGHT'
        else:
            sidecar = checkpoint.with_suffix('.json')
            if sidecar.exists() and _read(sidecar) != meta:
                raise ValueError('Pruned historical weight sidecar disagrees with evaluation')
            status = 'METRICS_ONLY_WEIGHT_PRUNED'
        item = {'step': step, 'evaluation_path': _relative(evpath),
            'evaluation_sha256': _sha(evpath), 'checkpoint_path': _relative(checkpoint),
            'checkpoint_sha256': meta['file_sha256'], 'model_sha256': record['model_sha256'],
            'weight_status': status}
        if cfg['stage'] == 'stage2':
            if (record.get('donor_selection') != donor_receipt
                    or record.get('calibration_sha256') != calibration_meta['artifact_sha256']):
                raise ValueError('Historical Student evaluation used another donor/calibration')
        else:
            valpath = directory / f'validation_{step:07d}.json'
            val = _read(valpath)
            if val.get('protocol') != selection.PROTOCOL:
                raise ValueError('Historical validation did not use RGB RRE')
            aggregate = selection.aggregate_rows(val['per_sample'], inventory)
            if any(val.get(k) != v for k, v in aggregate.items()):
                raise ValueError('Historical validation summary differs from population')
            summary = {k: v for k, v in val.items() if k not in ('per_sample', 'protocol')}
            if record.get('validation') != summary:
                raise ValueError('Historical validation and evaluation disagree')
            item.update(validation_path=_relative(valpath), validation_sha256=_sha(valpath))
        records.append(item)
    required = {cfg['primary_step'], cfg['horizon']}
    if not required.issubset({r['step'] for r in records}):
        raise ValueError('Required fixed primary/final endpoint missing')
    trace = _trace(directory / 'train.jsonl', cfg['horizon'])
    receipt = {'schema': 'WV3_HYPER_NUMERICAL_REUSE_V1',
        'new_run_id': cfg['run_id'], 'new_config_sha256': cfg['config_sha256'],
        'server': cfg['server'], 'repeat': cfg['repeat'], 'source_run_id': old_cfg['run_id'],
        'source_run_dir': _relative(directory), 'source_config': old_cfg,
        'source_config_path': _relative(directory / 'config.json'),
        'source_config_sha256': _sha(directory / 'config.json'),
        'source_admission_path': _relative(root / 'admission.json'),
        'source_admission_sha256': _sha(root / 'admission.json'),
        'source_release': old_pins['source_release'],
        'eval_protocol_id': plan.digest(old_pins['evaluator']),
        'records': records, 'training_trace': trace,
        'training_seconds': trace['optimizer_wall_seconds'],
        'training_time_scope': trace['scope'], 'new_training_performed': False,
        'new_inference_performed': False, 'fixed_steps': sorted(required),
        'weight_status': sorted({r['weight_status'] for r in records})}
    if donor_receipt is not None:
        receipt['donor_selection_receipt_sha256'] = donor_receipt['receipt_sha256']
        receipt['calibration_artifact_sha256'] = calibration_meta['artifact_sha256']
    receipt['receipt_sha256'] = plan.digest(receipt)
    return receipt


def try_reuse(cfg, run_dir, *, pins=None):
    """Audit before skipping. Missing controls fall back to fresh same-seed runs."""
    directory = Path(run_dir)
    if source_spec(cfg) is None:
        return None
    target = directory / 'reuse_receipt.json'
    if target.exists():
        return verify_reuse(cfg, directory, pins=pins)
    try:
        receipt = audit_control(cfg, pins=pins)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        atomic_json(directory / 'reuse_rejected.json', {'schema': 'WV3_HYPER_REUSE_REJECTED_V1',
            'run_id': cfg['run_id'], 'config_sha256': cfg['config_sha256'],
            'reason': str(exc), 'action': 'FRESH_SAME_RECIPE_AND_SEED'})
        return None
    assets.original._immutable_json(target, receipt)
    return receipt


def verify_reuse(cfg, run_dir, *, pins=None):
    """Previously accepted reuse must stay immutable; never silently relabel it."""
    saved = _read(Path(run_dir) / 'reuse_receipt.json')
    actual = audit_control(cfg, pins=pins)
    if actual is None or actual != saved:
        raise ValueError('Accepted historical reuse evidence changed')
    return actual


def load_reused_records(cfg, run_dir, *, pins=None):
    """Publisher access: original identity plus explicit verified reuse evidence."""
    receipt = verify_reuse(cfg, run_dir, pins=pins)
    result = []
    for item in receipt['records']:
        record = deepcopy(_read(plan.ROOT / item['evaluation_path']))
        record.update(evidence_path=item['evaluation_path'],
            evidence_sha256=item['evaluation_sha256'],
            reuse={'source_run_id': receipt['source_run_id'],
                'source_config_sha256': receipt['source_config']['config_sha256'],
                'receipt_sha256': receipt['receipt_sha256'], 'weight_status': item['weight_status'],
                'training_seconds': receipt['training_seconds'],
                'training_time_scope': receipt['training_time_scope'],
                'new_inference_performed': False})
        result.append(record)
    return result
