"""User-authorized QB Student futility gate; never a model/checkpoint selection rule.

Only complete official 8K/9K/10K records can trigger the gate. Existing completed
runs stay completed. A stopped run retains its full optimizer/RNG checkpoint and
is explicitly accounted for rather than reported as a full-horizon completion.
"""
import importlib
import math
from pathlib import Path

from directshift48.reporting import atomic_json, utc_now
from .orchestration import ROOT, read, sha, local, authenticate_stop

POLICY = 'USER_20261028_QB_S2_HQNR_8_9_10K_ALL_LT_090'
STEPS = (8000, 9000, 10000)
THRESHOLD = .90
STATUS = 'EARLY_STOPPED_HQNR'


def eligible(cfg):
    return cfg.get('dataset') == 'QB' and cfg.get('phase') == 'stage2'


def decision(directory, cfg):
    """None = missing evidence/inapplicable; False gate = continue normally."""
    if not eligible(cfg):
        return None
    directory = Path(directory)
    records, protocol = [], None
    for step in STEPS:
        path = directory / f'evaluation_{step:07d}.json'
        meta_path = directory / 'models' / f'step_{step:07d}.json'
        checkpoint = meta_path.with_suffix('.pt')
        if not all(p.is_file() for p in (path, meta_path, checkpoint)):
            return None
        if any(p.is_symlink() for p in (path, meta_path, checkpoint)):
            raise ValueError('QB early-stop evidence must be owned regular files')
        ev, meta = read(path), read(meta_path)
        ident, qual, artifact = ev.get('state_before', {}), ev.get('qualification', {}), meta.get('artifact', {})
        expected = {k: cfg[k] for k in ('run_id', 'dataset', 'phase', 'seed', 'attempt')}
        expected['completed_update'] = step
        if (any(ident.get(k) != v for k, v in expected.items())
                or any(ev.get(k) != ident for k in ('state_after', 'rr_state', 'fr_state'))
                or not all(qual.get(k) is True for k in ('same_state_verified', 'metrics_complete', 'metrics_finite'))
                or ev.get('metric_function_injection') is not False
                or ev.get('full_method_eligible') is not True
                or meta.get('evaluation') != ev
                or any(meta.get('config', {}).get(k) != cfg[k] for k in ('run_id', 'dataset', 'phase', 'seed', 'attempt'))
                or artifact.get('completed_update') != step
                or artifact.get('checkpoint_verified') is not True
                or artifact.get('model_state_sha256') != ident.get('model_state_sha256')):
            raise ValueError('QB early-stop official identity/checkpoint binding differs')
        for split in ('rr', 'fr'):
            completeness = qual.get('scene_completeness', {}).get(split, {})
            scenes = ev[split].get('per_scene', [])
            if (completeness != dict(completed=20, expected=20, duplicates=[], missing=[], unexpected=[])
                    or len(scenes) != 20
                    or {s.get('scene_id') for s in scenes} != {f'QB:{split}:{i}' for i in range(20)}):
                raise ValueError('QB early-stop requires all official RR20/FR20 scenes')
        # alignment_protocol_digest includes deployment.completed_update by
        # design. Compare the actual deployment with only that counter removed.
        deployment = dict(ev.get('mature30_deployment', {}))
        if deployment.pop('completed_update', None) != step or not deployment:
            raise ValueError('QB early-stop deployment identity missing')
        current = {k: ident.get(k) for k in ('evaluator_revision', 'inference_policy_digest')}
        current['deployment'] = deployment
        if not all(current.values()) or (protocol is not None and protocol != current):
            raise ValueError('QB early-stop evaluation protocol changed within the window')
        protocol = current
        value = ev['fr']['metrics']['HQNR']
        scene_mean = sum(s['metrics']['HQNR'] for s in ev['fr']['per_scene']) / 20
        if (type(value) not in (float, int) or not math.isfinite(value)
                or not math.isfinite(scene_mean) or abs(value - scene_mean) > 1e-12
                or qual.get('HQNR') != value or sha(checkpoint) != artifact.get('file_sha256')):
            raise ValueError('QB early-stop metric or retained checkpoint failed verification')
        records.append(dict(step=step, HQNR=value, evaluation_sha256=sha(path),
                            checkpoint_sha256=artifact['file_sha256']))
    return dict(policy=POLICY, run_id=cfg['run_id'], threshold=THRESHOLD,
                comparison='strict_less_than_all_three', window=records,
                stop=all(r['HQNR'] < THRESHOLD for r in records))


def verify(row, cfg, root=ROOT):
    directory = local(row['directory'], root)
    record = read(directory / 'early_stop.json')
    evidence = decision(directory, cfg)
    if (row.get('status') != STATUS or not evidence or not evidence['stop']
            or record.get('decision') != evidence or record.get('policy') != POLICY
            or not STEPS[-1] <= row['step'] < cfg['planned_horizon']
            or record.get('stop') != authenticate_stop(row, cfg, root)):
        raise ValueError('QB early-stop accounting lacks exact decision and safe checkpoint proof')
    return record


def finish_stopped(camp, cfg):
    """Adopt an already safely stopped, unfinished run without loading GPU state."""
    row = next((r for r in camp.state['runs'] if r['run_id'] == cfg['run_id']), None)
    if not row or row['status'] == 'COMPLETED':
        return False
    if row['status'] == STATUS:
        record = verify(row, cfg)
        camp.state.setdefault('early_stopped_runs', {})[cfg['run_id']] = record
        camp.persist()
        emit_status(camp, cfg, row, record)
        return True
    if row['status'] != 'STOPPED' or not eligible(cfg) or not STEPS[-1] <= row['step'] < cfg['planned_horizon']:
        return False
    directory = local(row['directory'])
    evidence = decision(directory, cfg)
    if not evidence or not evidence['stop']:
        return False
    stop = authenticate_stop(row, cfg)
    record = dict(policy=POLICY, decision=evidence, stop=stop, at=utc_now(),
                  planned_horizon=cfg['planned_horizon'], full_horizon_completed=False,
                  status_before=row.copy())
    path = directory / 'early_stop.json'
    # Receipt first permits crash recovery without overwriting the original reason.
    if path.exists():
        saved = read(path)
        if saved['decision'] != evidence or saved['stop'] != stop:
            raise ValueError('Existing QB early-stop receipt differs')
        record = saved
    else:
        atomic_json(path, record)
    row.update(status=STATUS, early_stop_policy=POLICY, early_stopped_at=record['at'])
    atomic_json(directory / 'run_state.json', row)
    camp.state.setdefault('early_stopped_runs', {})[cfg['run_id']] = record
    camp.state['status'] = 'RUNNING'
    camp.persist()
    verify(row, cfg)
    emit_status(camp, cfg, row, record)
    return True


def emit_status(camp, cfg, row, record):
    package = type(camp).__module__.split('.')[0]
    reporting = importlib.import_module(package + '.reporting')
    # The resolved configuration includes frozen donor/calibration provenance.
    resolved = read(local(row['directory']) / 'config.json')
    reporting.emit_status(camp.root, resolved, STATUS, step=row['step'],
                          extra={'early_stop': record}, timestamp=record['at'])


def run_case(camp, cfg, budget_error):
    """Wrap the unchanged training loop with its existing checkpoint-safe stop."""
    if finish_stopped(camp, cfg):
        return
    if not eligible(cfg):
        return camp.run_case(cfg)
    class EarlyStop(budget_error):
        pass
    original_check = camp.check
    cached, inspected = [None], [-1]
    def check(**kwargs):
        original_check(**kwargs)
        row = next((r for r in camp.state['runs'] if r['run_id'] == cfg['run_id']), None)
        if row and STEPS[-1] <= row['step'] < cfg['planned_horizon']:
            bucket = row['step'] // 1000
            if cached[0] is None and inspected[0] != bucket:
                cached[0] = decision(local(row['directory']), cfg)
                inspected[0] = bucket
            if cached[0] and cached[0]['stop']:
                raise EarlyStop(POLICY)
    camp.check = check
    try:
        camp.run_case(cfg)
    except EarlyStop:
        if not finish_stopped(camp, cfg):
            raise RuntimeError('QB early-stop failed to preserve a safe terminal checkpoint')
    finally:
        camp.check = original_check
