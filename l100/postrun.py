"""Horizon-aware, evaluation-only LOCAL-T candidate debt and selectors.

An exact endpoint can be reported before the other 49 candidates. This does
not claim a complete grid and does not gate local calibration on performance.
"""
from pathlib import Path

import yaml

from g20.postrun import profile_model, validate_record
from l100.common import (ROOT, atomic_json, check_deadline, load_checkpoint_model,
                          object_sha, read_json, sha256, source_identity, utcnow,
                          apply_runtime_policy)
from l100.evaluation import FRMetrics, evaluate_model
from l100.plan import CAMPAIGN_ID, case_from_config, grid_steps, sensor_spec

SELECTIONS = (('raw_max', 'RAW_MAX'), ('target', 'TARGET'),
              ('exact_final', 'EXACT_FINAL'), ('rr_val_selected', 'RR_VAL_SELECTED'),
              ('e_min_diag', 'E_MIN_DIAG'))


def selections(updates):
    grid_steps(updates)
    return SELECTIONS + ((('mid50_of100', 'MID50_OF100'),) if updates == 100000 else ())


def select_records(records, updates, sensor='GF2'):
    spec = sensor_spec(sensor)
    expected = grid_steps(updates)
    by_step = {r['update']: r for r in records}
    if (len(by_step) != len(records) or sorted(by_step) != list(expected)
            or any(type(step) is not int for step in by_step)):
        raise ValueError('Selection requires every unique horizon-specific fixed-grid candidate')
    for row in records:
        validate_record(row, spec)
    eligible = [r for r in records if r['fr']['hqnr'] > spec.hqnr_threshold]
    target = min(eligible, key=lambda r: (r['rr']['ergas'], -r['rr']['scc'],
                 -r['rr']['psnr'], r['update'])) if eligible else None
    result = dict(raw_max=max(records, key=lambda r: (r['fr']['hqnr'], -r['update'])),
        target=target, exact_final=by_step[updates],
        rr_val_selected=min(records, key=lambda r: (r['val_ergas'], r['update'])),
        e_min_diag=min(records, key=lambda r: (r['rr']['ergas'], r['update'])),
        n_eligible=len(eligible), n_evaluated=len(records),
        target_status='official' if eligible else 'no_eligible',
        joint_pass=bool(target and target['rr']['ergas'] < spec.ergas_goal),
        strong_joint_pass=bool(target and target['rr']['ergas'] < spec.ergas_strong_goal))
    if updates == 100000:
        result['mid50_of100'] = by_step[50000]
    return result


def report_selection(label, record, context, *, official_complete=True, **extra):
    if label not in {name for _, name in selections(context['horizon_updates'])}:
        raise ValueError('Selection label is not registered for this horizon')
    extra.update(test_aware=label in ('RAW_MAX', 'TARGET', 'E_MIN_DIAG'),
                 primary=label in ('EXACT_FINAL', 'RR_VAL_SELECTED'), independent_test=False)
    if record is None:
        if label != 'TARGET':
            raise ValueError('Only an ineligible TARGET may have no checkpoint')
        return dict(context, selection_id=label, official_complete=official_complete,
                    selection=None, **extra)
    return dict(context, selection_id=label, official_complete=official_complete,
        step=record['update'], checkpoint_sha256=record['checkpoint_identity']['model_sha256'],
        checkpoint_identity=record['checkpoint_identity'], rr=record['rr'], fr=record['fr'],
        val_ergas=record['val_ergas'], eval_mode='A_ON', precision='fp32',
        fresh_horizon_endpoint=label == 'EXACT_FINAL',
        is_fresh50k=label == 'EXACT_FINAL' and context['horizon_updates'] == 50000, **extra)


def validate_grid(run, cfg, grid, root=ROOT):
    from l100.training import validate_config
    case = validate_config(cfg)
    expected = list(grid_steps(case.updates))
    if (case.run_id != run or grid.get('run_id') != run or grid.get('campaign_id') != CAMPAIGN_ID
            or cfg['l100']['sensor'] != 'GF2' or cfg['l100']['num_bands'] != 4
            or grid.get('horizon_updates') != case.updates or grid.get('expected_steps') != expected
            or grid.get('config_sha256') != object_sha(cfg)):
        raise ValueError('Official grid/config/campaign/horizon identity mismatch')
    data = read_json(Path(root) / cfg['l100']['dataset_manifest'])
    if (object_sha(data) != grid.get('data_sha256') or data.get('sensor') != 'GF2'
            or data.get('num_bands') != 4 or data.get('max_pixel') != 1023):
        raise ValueError('Official data manifest mismatch')
    records = grid['records']
    steps = [r['update'] for r in records]
    if (len(set(steps)) != len(steps) or any(type(s) is not int or s not in expected for s in steps)
            or grid.get('n_evaluated') != len(records)
            or grid.get('complete') is not (sorted(steps) == expected)):
        raise ValueError('Diagnostic/duplicate steps or false completeness in official grid')
    wd = Path(root) / 'work_dir' / run
    for record in records:
        validate_record(record, 'GF2')
        folder = wd / 'candidates' / str(record['update'])
        identity = read_json(folder / 'identity.json')
        if (identity != record['checkpoint_identity'] or identity.get('model_sha256') != sha256(folder / 'model.safetensors')
                or any(identity.get(k) != grid.get(k) for k in
                       ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256'))):
            raise ValueError('Candidate bytes/config/data/reference/evaluator provenance mismatch')
    return data


def verify_fullstate(run, cfg, grid, root=ROOT, *, include_midpoint=True):
    """Verify exact N (and 100K midpoint) state without evaluating or optimizing."""
    import torch
    from safetensors.torch import load_file
    from fh12.training import BatchStream
    from g20.model import state_hash
    from qg40.exposure import validate_counts
    from l100.training import cosine_factor, aligner_factor
    case = case_from_config(cfg)
    data = read_json(Path(root) / cfg['l100']['dataset_manifest'])
    steps = (50000, case.updates) if include_midpoint and case.updates == 100000 else (case.updates,)
    evidence = {}
    for step in steps:
        folder = Path(root) / 'work_dir' / run / 'candidates' / str(step)
        identity = read_json(folder / 'identity.json')
        state_path, model_path = folder / 'training_state.pt', folder / 'model.safetensors'
        expected = dict(update=step, full_state=True,
                        **{k: grid[k] for k in ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256')})
        if (any(identity.get(k) != v for k, v in expected.items())
                or identity.get('role') != case.role or identity.get('sensor') != 'GF2'
                or identity.get('num_bands') != 4 or identity.get('input_layout') != case.input_layout
                or identity.get('model_sha256') != sha256(model_path)
                or identity.get('training_state_sha256') != sha256(state_path)):
            raise ValueError('Exact/midpoint full-state checkpoint identity or bytes differ')
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if (any(state.get(k) != v for k, v in expected.items()) or state.get('precision') != 'fp32'
                or state.get('model_sha256') != identity['model_sha256']
                or state.get('scheduler', {}).get('last_epoch') != step
                or state_hash(state['model_state']) != identity.get('state_hash')):
            raise ValueError('Exact/midpoint full-state model/scheduler/provenance differs')
        tensors = load_file(str(model_path), device='cpu')
        if (state_hash(tensors) != identity['state_hash']
                or not all(bool(torch.isfinite(value).all()) for value in tensors.values())):
            raise ValueError('Exact/midpoint model weights differ from finite full-state')
        counts = validate_counts(state.get('exposure_counts'), data['splits']['train']['count'], step, cfg['batch_size'])
        stream = BatchStream(len(counts), cfg['batch_size'], case.seed)
        stream.load_state_dict(state['sampler'])
        if stream.epoch * stream.count + stream.cursor != step:
            raise ValueError('Exact/midpoint sampler is not at the declared completed update')
        factor = cosine_factor(step, cfg['num_warmup'], case.updates)
        expected_lr = [cfg['learning_rate'] * factor,
                       cfg['l100']['aligner_lr'] * factor * aligner_factor(step, case.profile)]
        groups = state.get('optimizer', {}).get('param_groups', [])
        lr = state['scheduler'].get('_last_lr')
        import math
        if ([g.get('name') for g in groups] != ['U', 'A'] or not isinstance(lr, list) or len(lr) != 2
                or any(not math.isclose(float(g['lr']), want, rel_tol=1e-10, abs_tol=1e-14)
                       or not math.isclose(float(got), want, rel_tol=1e-10, abs_tol=1e-14)
                       for g, got, want in zip(groups, lr, expected_lr))):
            raise ValueError('Exact/midpoint optimizer LR is not the declared fresh-horizon scheduler')
        evidence[str(step)] = dict(model_sha256=identity['model_sha256'],
                                  training_state_sha256=identity['training_state_sha256'])
        del state, tensors
    return evidence


def update_best_metadata(wd, grid):
    """Retain legacy HQNR/SCC/ERGAS best metadata separately from plan primaries."""
    if not grid['records']:
        return
    best = min(grid['records'], key=lambda r: (-r['fr']['hqnr'], -r['rr']['scc'],
                                              r['rr']['ergas'], r['update']))
    atomic_json(Path(wd) / 'best_hqnr_meta.json', dict(step=best['update'],
        candidate_path=str(Path(wd) / 'candidates' / str(best['update'])),
        checkpoint_sha256=best['checkpoint_identity']['model_sha256'],
        hqnr=best['fr']['hqnr'], scc=best['rr']['scc'], ergas=best['rr']['ergas'],
        selector='HQNR_MAX_SCC_MAX_ERGAS_MIN_STEP_MIN', test_aware=True,
        partial_grid=len(grid['records']) < 50))


def _context(case, cfg, grid):
    return dict(campaign_id=CAMPAIGN_ID, run_id=case.run_id, sensor='GF2', role=case.role,
        server_id=case.server_id, profile=case.profile, horizon_updates=case.updates,
        teacher_reference_id=case.reference_id, teacher_owner=case.teacher_owner,
        teacher_seed=case.teacher_seed, teacher_updates=case.teacher_updates,
        pair_baseline_run_id=case.pair_baseline_run_id, config_sha256=object_sha(cfg),
        data_sha256=grid['data_sha256'], source_identity=grid['source_identity'],
        reference_sha256=grid['reference_sha256'], normal_same_step_A_U=True)


def _open(run, root, deadline):
    from l100.training import runtime_context, validate_config
    root, wd = Path(root), Path(root) / 'work_dir' / run
    apply_runtime_policy(root)
    cfg = yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text())
    case = validate_config(cfg)
    if case.run_id != run:
        raise ValueError('Resolved case and run differ')
    window, _ = runtime_context(cfg, root, deadline, resume=True)
    deadline = window['deadline_utc']
    check_deadline(deadline)
    training = read_json(wd / 'meta/training_status.json')
    actual = training.get('actual_updates')
    if (type(actual) is not int or not 0 <= actual <= case.updates
            or training.get('horizon_updates', case.updates) != case.updates):
        raise ValueError('Training update count is outside its registered horizon')
    start = read_json(wd / 'meta/training_start_manifest.json')
    release = source_identity(root)
    data = read_json(root / cfg['l100']['dataset_manifest'])
    if (start.get('run_id') != run or start.get('campaign_id') != CAMPAIGN_ID
            or start.get('horizon_updates') != case.updates
            or start.get('source_identity') != release or start.get('config_sha256') != object_sha(cfg)
            or start.get('data_sha256') != object_sha(data)):
        raise ValueError('Run release/data/start identity changed')
    path = wd / 'official/raw_grid.json'
    grid = read_json(path) if path.is_file() else dict(campaign_id=CAMPAIGN_ID, run_id=run,
        sensor='GF2', horizon_updates=case.updates, expected_steps=list(grid_steps(case.updates)),
        records=[], n_evaluated=0, complete=False,
        **{k: start[k] for k in ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256')})
    validate_grid(run, cfg, grid, root)
    if grid['source_identity'] != release or any(r['update'] > actual for r in grid['records']):
        raise ValueError('Evaluation grid is ahead of training or from another release')
    return root, wd, cfg, case, training, grid, data, deadline


def _repair_saved_grid(run, cfg, grid, data, device, deadline, root, actual_updates, *, only_steps=None):
    """Repay saved evaluation debt; never invent or optimize a checkpoint."""
    from g20.data import build_dataset
    case = case_from_config(cfg)
    expected = grid_steps(case.updates)
    steps = expected if only_steps is None else tuple(only_steps)
    if any(s not in expected for s in steps):
        raise ValueError('Diagnostic checkpoints may not enter official selectors')
    wd = Path(root) / 'work_dir' / run
    datasets = engine = None
    existing = {r['update'] for r in grid['records']}
    try:
        for step in steps:
            if step > actual_updates or step in existing:
                continue
            check_deadline(deadline)
            folder = wd / 'candidates' / str(step)
            if not (folder / 'identity.json').is_file() or not (folder / 'model.safetensors').is_file():
                continue
            identity = read_json(folder / 'identity.json')
            if identity.get('update') != step or any(identity.get(k) != grid.get(k) for k in
                    ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256')):
                raise ValueError('Saved evaluation-debt candidate identity differs from run')
            if datasets is None:
                datasets = {s: build_dataset(data, s, root=root) for s in ('rr', 'fr', 'val')}
                engine = FRMetrics(datasets['fr'])
            model, identity = load_checkpoint_model(cfg, folder, device, expected_source=grid['source_identity'])
            try:
                metrics = evaluate_model(model, datasets, device, engine=engine, deadline=deadline)
            finally:
                del model
            record = dict(update=step, checkpoint_identity=identity, **metrics)
            validate_record(record, 'GF2')
            grid['records'].append(record)
            grid['records'].sort(key=lambda r: r['update'])
            grid['n_evaluated'] = len(grid['records'])
            grid['complete'] = sorted(r['update'] for r in grid['records']) == list(expected)
            atomic_json(wd / 'official' / f'candidate_{step}.json', record)
            atomic_json(wd / 'official/raw_grid.json', grid)
            print(f'[L100 official {step}/{case.updates}] HQNR={record["fr"]["hqnr"]:.8f} '
                  f'SCC={record["rr"]["scc"]:.8f} ERGAS={record["rr"]["ergas"]:.8f}', flush=True)
    except TimeoutError:
        pass
    grid['complete'] = sorted(r['update'] for r in grid['records']) == list(expected)
    atomic_json(wd / 'official/raw_grid.json', grid)
    pending = [s for s in expected if s not in {r['update'] for r in grid['records']}]
    atomic_json(wd / 'official/evaluation_debt.json', dict(
        **{k: v for k, v in grid.items() if k not in ('records', 'complete', 'n_evaluated')},
        pending_steps=pending, n_pending=len(pending), official_complete=not pending,
        teacher_transition_requires_full_grid=False))
    update_best_metadata(wd, grid)
    return grid


def _status(case, cfg, grid, training, *, endpoint=False):
    seen = {r['update'] for r in grid['records']}
    pending = [s for s in grid_steps(case.updates) if s <= training['actual_updates'] and s not in seen]
    return dict(status='OFFICIAL_EVAL_PENDING', official_complete=False, sheet_uploaded=False,
        endpoint_complete=endpoint, actual_updates=training['actual_updates'], horizon_updates=case.updates,
        n_evaluated=len(seen), pending_steps=pending, n_pending=len(pending),
        config_sha256=object_sha(cfg), source_identity=grid['source_identity'],
        data_sha256=grid['data_sha256'], primary_selection_available=endpoint)


def process_endpoint(run, device='cuda', deadline=None, root=ROOT):
    """Evaluate only exact N, allowing local calibration without the 50-FR gate."""
    root, wd, cfg, case, training, grid, data, deadline = _open(run, root, deadline)
    if training['actual_updates'] != case.updates or not training.get('training_complete'):
        raise ValueError('Endpoint report requires the completed registered horizon')
    verify_fullstate(run, cfg, grid, root, include_midpoint=False)
    grid = _repair_saved_grid(run, cfg, grid, data, device, deadline, root, case.updates,
                              only_steps=(case.updates,))
    endpoint = next((r for r in grid['records'] if r['update'] == case.updates), None)
    status = _status(case, cfg, grid, training, endpoint=endpoint is not None)
    if endpoint is not None:
        report = report_selection('EXACT_FINAL', endpoint, _context(case, cfg, grid),
            official_complete=False, endpoint_complete=True, n_evaluated=len(grid['records']),
            grid_complete=grid['complete'], status='ENDPOINT_COMPLETE_OFFICIAL_GRID_PENDING')
        atomic_json(wd / 'official/exact_final_endpoint.json', report)
    previous = wd / 'official/postrun_status.json'
    if not previous.exists() or not read_json(previous).get('official_complete'):
        atomic_json(previous, status)
    return status


def process_partial(run, device='cuda', deadline=None, root=ROOT):
    root, wd, cfg, case, training, grid, data, deadline = _open(run, root, deadline)
    if training['actual_updates'] >= case.updates or training.get('training_complete'):
        raise ValueError('Partial evaluation requires honestly incomplete training')
    grid = _repair_saved_grid(run, cfg, grid, data, device, deadline, root, training['actual_updates'])
    status = _status(case, cfg, grid, training)
    status.update(status='PARTIAL_TIME_LIMIT', partial_evaluation_complete=not status['pending_steps'])
    atomic_json(wd / 'official/postrun_status.json', status)
    return status


def process(run, device='cuda', deadline=None, upload=False, upload_only=False, root=ROOT):
    root = Path(root)
    path = root / 'work_dir' / run / 'official/postrun_status.json'
    if upload_only:
        status = read_json(path)
        if not status.get('official_complete'):
            raise ValueError('Cannot upload incomplete official results')
        return _upload(run, status, path, root) if upload else 0
    root, wd, cfg, case, training, grid, data, deadline = _open(run, root, deadline)
    if training['actual_updates'] != case.updates or not training.get('training_complete'):
        raise ValueError('Official postrun requires the completed registered horizon')
    verify_fullstate(run, cfg, grid, root)
    if not grid['complete']:
        grid = _repair_saved_grid(run, cfg, grid, data, device, deadline, root, case.updates)
    if not grid['complete']:
        atomic_json(path, _status(case, cfg, grid, training,
            endpoint=any(r['update'] == case.updates for r in grid['records'])))
        return 0
    selected = select_records(grid['records'], case.updates)
    update_best_metadata(wd, grid)
    context, spec = _context(case, cfg, grid), sensor_spec()
    for key, label in selections(case.updates):
        extra = dict(n_evaluated=50)
        if key == 'target':
            extra.update(n_eligible=selected['n_eligible'], target_status=selected['target_status'],
                hqnr_threshold=spec.hqnr_threshold, threshold_comparison='>', ergas_goal=spec.ergas_goal,
                ergas_strong_goal=spec.ergas_strong_goal, selector_order=['ergas', '-scc', '-psnr', 'step'],
                joint_pass=selected['joint_pass'], strong_joint_pass=selected['strong_joint_pass'])
        if key == 'e_min_diag':
            extra.update(n_candidates=50, development_oracle=True)
        filename = 'target_selection' if key == 'target' else key
        atomic_json(wd / 'official' / f'{filename}.json', report_selection(label, selected[key], context, **extra))
    cost_path = wd / 'official/profile.json'
    if cost_path.is_file():
        cost = read_json(cost_path)
        if (cost.get('config_sha256') != object_sha(cfg) or cost.get('source_identity') != grid['source_identity']
                or cost.get('num_bands') != 4 or cost.get('sensor') != 'GF2'):
            raise ValueError('Measured C4 profile provenance differs')
    else:
        from g20.data import build_dataset
        model, _ = load_checkpoint_model(cfg, wd / 'candidates' / str(case.updates), device,
                                         expected_source=grid['source_identity'])
        try:
            cost = profile_model(model, build_dataset(data, 'rr', root=root), device, deadline)
        finally:
            del model
        cost.update(config_sha256=object_sha(cfg), source_identity=grid['source_identity'])
        atomic_json(cost_path, cost)
    status = dict(status='OFFICIAL_EVAL_COMPLETE', official_complete=True, sheet_uploaded=False,
        endpoint_complete=True, config_sha256=object_sha(cfg), source_identity=grid['source_identity'],
        data_sha256=grid['data_sha256'], actual_updates=case.updates, horizon_updates=case.updates,
        n_evaluated=50, completed_at_utc=utcnow(), target_status=selected['target_status'])
    atomic_json(path, status)
    return _upload(run, status, path, root) if upload else 0


def _upload(run, status, path, root):
    from l100.upload import upload_run
    try:
        upload_run(run, root, activated=True)
    except Exception as exc:
        status.update(sheet_uploaded=False, upload_error=f'{type(exc).__name__}: {exc}')
        atomic_json(path, status)
        return 2
    status.update(sheet_uploaded=True)
    status.pop('upload_error', None)
    atomic_json(path, status)
    return 0
