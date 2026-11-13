"""Explicit policy overlay; never rewrite frozen configs, weights or evaluations."""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
from panda_eq40_zeroq import plan, publication as pub
from directshift48.reporting import INDEX, atomic_json, canonical_json, utc_now

POLICY = 'ZEROQ_EQ40_CORRECTED_PAN_SELECTION_4DP_V2'
REFERENCE = 'CORRECTED_PAN'
LEGACY_POLICY = pub.POLICY
_legacy_make = pub.make_row
_legacy_verify = pub.verify_owned
_legacy_joint = pub.joint_pass
_installed = False


def hqnr(record):
    value = record['fr_corrected_pan']['metrics']['HQNR']
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Missing/nonfinite corrected-PAN HQNR; no original-PAN fallback')
    return value


def joint_pass(record, cfg):
    target = plan.targets(cfg['sensor'])
    return bool(record and hqnr(record) >= target['HQNR_min']
                and record['rr']['metrics']['ERGAS'] <= target['ERGAS_max'])


def choose(records, stage, cfg):
    if stage != 'stage2':
        raise ValueError('Corrected selection applies to ZEROQ Stage2, not historical Stage1')
    by_step = {}
    for record in records:
        if record['step'] in by_step and by_step[record['step']] != record:
            raise ValueError('Conflicting same-step evaluations')
        hqnr(record)
        by_step[record['step']] = record
    if not by_step:
        return []
    records = [by_step[s] for s in sorted(by_step)]
    qualified = [e for e in records if hqnr(e) >= plan.targets(cfg['sensor'])['HQNR_min']]
    rep = (min(qualified, key=lambda e: (e['rr']['metrics']['ERGAS'], -hqnr(e), e['step']))
           if qualified else min(records, key=lambda e: (-hqnr(e), e['rr']['metrics']['ERGAS'], e['step'])))
    selected = {rep['step']: ['HQNR_QUALIFIED_MIN_ERGAS' if qualified else 'BEST_HQNR_FALLBACK']}
    for e in records:
        if joint_pass(e, cfg):
            selected.setdefault(e['step'], []).append('PASS_ALL_STEPS')
    if cfg['primary_step'] in by_step:
        selected.setdefault(cfg['primary_step'], []).append('FIXED40K')
    order = [rep['step']] + sorted(s for s in selected if s != rep['step'])
    return [(by_step[s], '|'.join(selected[s]), f'ROW{i}') for i, s in enumerate(order)]


def make_row(cfg, status, record=None, selector='STATUS', slot='ROW0'):
    row, note_text = _legacy_make(cfg, status, record, selector, slot)
    note = json.loads(note_text)
    note.update(checkpoint_selection_reference='corrected PAN',
        selection_policy_override={'policy': POLICY, 'reference': REFERENCE,
            'original_training_config_preserved': True,
            'previous_selection_reference': 'ORIGINAL_PAN',
            'thresholds_unchanged': True,
            'external_original_PAN_benchmarks_directly_comparable': False})
    note['publication']['metric_reference'] = REFERENCE
    row[INDEX['Experiment']] = cfg['experiment'] + ' [보정 PAN 선정]'
    row[INDEX['Selection@Step']] = '보정 PAN · ' + row[INDEX['Selection@Step']]
    if status == 'EARLY_STOPPED':
        row[INDEX['Selection@Step']] = '10K 조기중단 · ' + row[INDEX['Selection@Step']]
    row[INDEX['Eval_scope']] = 'DUAL_FR_CORRECTED_SELECTION_FULL20' if record else 'STATUS_ONLY'
    row[INDEX['Original description / notes']] = canonical_json(note)
    return row, canonical_json(note)


@contextmanager
def legacy_context():
    names = {'POLICY': LEGACY_POLICY, 'make_row': _legacy_make, 'joint_pass': _legacy_joint}
    saved = {k: getattr(pub, k) for k in names}
    try:
        for k, v in names.items():
            setattr(pub, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(pub, k, v)


def verify_owned(row, value, server):
    note = json.loads(row[INDEX['Original description / notes']])
    policy = note['publication']['policy']
    if policy == LEGACY_POLICY:
        with legacy_context():
            return _legacy_verify(row, value, server)
    if policy != POLICY or note.get('checkpoint_selection_reference') != 'corrected PAN':
        raise ValueError('Unknown selection policy on an owned row')
    return _legacy_verify(row, value, server)


def early_stop_evidence(cfg, directory, step):
    from panda_eq40_zeroq.evaluation import verify_record
    policy = cfg['early_stop']
    if not policy['enabled'] or step != policy['at_step']:
        return None
    records = [json.loads((Path(directory) / f'evaluation_{s:07d}.json').read_text()) for s in policy['steps']]
    for record in records:
        verify_record(record, cfg)
    values = [hqnr(e) for e in records]
    if all(v < policy['HQNR_below'] for v in values):
        return {'policy': {**policy, 'reference': REFERENCE}, 'selection_policy': POLICY,
                'HQNR': values, 'step': step, 'reason': 'QB_CORRECTED_HQNR_8K_9K_10K_ALL_BELOW_090'}
    return None


def adopt_runtime_policy(server):
    """Reconsider an old QB stop using local immutable evidence before any resume."""
    import fcntl
    from panda_eq40_zeroq.runtime import read_state, latest_checkpoint
    root = plan.campaign_root(server)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_state(root, server)
        rechecks = []
        for job, cfg in zip(state['runs'], plan.queue(server)):
            if job['status'] != 'EARLY_STOPPED':
                continue
            if not cfg['early_stop']['enabled'] or job['step'] != cfg['early_stop']['at_step']:
                raise ValueError('Stopped state differs from the registered stopping boundary')
            directory = root / 'runs' / cfg['run_id']
            decision = early_stop_evidence(cfg, directory, cfg['early_stop']['at_step'])
            audit = {'run_id': cfg['run_id'], 'policy': POLICY, 'at': utc_now(),
                     'old_state': deepcopy(job), 'corrected_stop_evidence': decision}
            old_file = directory / 'early_stop.json'
            audit['previous_stop_evidence'] = json.loads(old_file.read_text()) if old_file.exists() else None
            if decision is None:
                resume = latest_checkpoint(directory)
                if resume is None or json.loads(resume.with_suffix('.json').read_text())['step'] != job['step']:
                    raise ValueError('Reopening requires the complete checkpoint at the stopped update')
                job.update(status='PAUSED', previous_status='EARLY_STOPPED', updated_at=utc_now(),
                           policy_override=POLICY)
                audit['action'] = 'RESUME_SAME_OPTIMIZER_STREAM_RNG'
            else:
                audit['action'] = 'KEEP_STOPPED_CORRECTED_RULE_ALSO_FAILS'
            atomic_json(directory / 'corrected_hqnr_stop_review.json', audit)
            rechecks.append(audit)
        if any(r['status'] == 'PAUSED' for r in state['runs']):
            state['status'] = 'PAUSED'
        state['selection_policy_override'] = POLICY
        atomic_json(root / 'campaign_state.json', state)
        atomic_json(root / 'corrected_hqnr_policy.json', {'policy': POLICY, 'reference': REFERENCE,
            'at': utc_now(), 'rechecks': rechecks, 'thresholds': plan.targets(plan.sensor_for(server)),
            'unchanged': ['model', 'loss', 'optimizer', 'LR_schedule', 'RNG', 'donor', 'raw_metric_definitions'],
            'scope': 'ZEROQ Stage2 selection, goal annotations, QB early stop; historical donor receipts preserved'})
        return rechecks


def install():
    global _installed
    if _installed:
        return
    from panda_fr_columns import publication as fr
    from panda_eq40_zeroq import runtime, evaluation
    fr.install()
    pub.POLICY, pub.choose, pub.joint_pass = POLICY, choose, joint_pass
    pub.make_row, pub.verify_owned = make_row, verify_owned
    runtime.early_stop_evidence = early_stop_evidence
    old_extras = fr.extras
    def extras(value):
        result = old_extras(value)
        if value['policy'] == POLICY:
            for cells in result.values():
                try:
                    note = json.loads(cells[0]['note'])
                except json.JSONDecodeError:
                    continue
                note.update(official_selection_reference=REFERENCE, selection_policy=POLICY,
                            original_PAN_benchmark_comparable=False)
                cells[0]['note'] = canonical_json(note)
        return result
    fr.extras = extras
    old_evaluate = evaluation.evaluate
    def evaluate(*args, **kwargs):
        record = old_evaluate(*args, **kwargs)
        record.update(selection_policy=POLICY, selection_reference=REFERENCE,
                      selection_joint_pass=joint_pass(record, args[0].cfg))
        return record
    evaluation.evaluate = evaluate
    _installed = True
