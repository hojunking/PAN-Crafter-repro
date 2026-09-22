"""Same-checkpoint official reporting for FT20K/60K and fresh100K budgets."""
from pathlib import Path
import math
import time

from safetensors.torch import load_file
import torch

from gfp40.common import (ROOT, atomic_json, check_deadline, object_sha, read_json,
    read_config, resolved_path, sha256, source_identity, utcnow)
from gfp40.plan import CAMPAIGN_ID, grid_steps, validate_config
from gfp40.evaluation import evaluate_model, FRMetrics
from g20.model import build_model, state_hash
from g20.postrun import validate_record

SELECTIONS = ('EXACT_FINAL', 'RR_VAL_SELECTED', 'RAW_AUX')


def summarize_grid(records, updates, context=None):
    """Pure selection; t=0 and off-grid curve20K/40K are never candidates."""
    expected = list(grid_steps(updates))
    by_step = {r['update']: r for r in records}
    if (len(by_step) != len(records) or sorted(by_step) != expected
            or any(type(r['update']) is not int for r in records)):
        raise ValueError('Every unique fixed-grid candidate is required for selection')
    for record in records:
        validate_record(record, 'GF2')
    chosen = dict(EXACT_FINAL=by_step[updates],
        RR_VAL_SELECTED=min(records, key=lambda r: (r['val_ergas'], r['update'])),
        RAW_AUX=min(records, key=lambda r: (-r['fr']['hqnr'], -r['rr']['scc'],
                                           r['rr']['ergas'], r['update'])))
    selections = {label: dict(row, selection_id=label, primary=label == 'EXACT_FINAL',
        test_aware=True, independent_test=False, candidate_count=len(expected),
        selector=('fixed_exact_endpoint' if label == 'EXACT_FINAL' else
                  'val_ERGAS_min_then_earlier_step' if label == 'RR_VAL_SELECTED' else
                  'HQNR_max_SCC_max_ERGAS_min_step_min')) for label, row in chosen.items()}
    return dict(context or {}, schema='GFP40_OFFICIAL_SUMMARY_v1', campaign_id=CAMPAIGN_ID,
        complete=True, official_complete=True, n_candidates=len(expected), expected_steps=expected,
        selections=selections, test_aware_development=True,
        candidate_budgets_comparable_only_within_horizon=True,
        endpoint_joint_goal=bool(chosen['EXACT_FINAL']['fr']['hqnr'] > .964
                                  and chosen['EXACT_FINAL']['rr']['ergas'] < .552),
        endpoint_strong_goal=bool(chosen['EXACT_FINAL']['fr']['hqnr'] > .964
                                  and chosen['EXACT_FINAL']['rr']['ergas'] < .522))


def load_candidate(cfg, directory, device='cpu', *, expected_source=None):
    case = validate_config(cfg, require_bound=True)
    directory = Path(directory)
    identity = read_json(directory / 'identity.json')
    expected = dict(config_sha256=object_sha(cfg), role='S', sensor='GF2', num_bands=4,
                    input_layout='PLH')
    if expected_source is not None:
        expected['source_identity'] = expected_source
    if (any(identity.get(k) != v for k, v in expected.items())
            or identity.get('model_sha256') != sha256(directory / 'model.safetensors')):
        raise ValueError('Candidate config/model/source identity differs')
    tensors = load_file(str(directory / 'model.safetensors'), device='cpu')
    if state_hash(tensors) != identity.get('state_hash') or not all(torch.isfinite(t).all().item() for t in tensors.values()):
        raise ValueError('Candidate tensor hash/finite check failed')
    model, _ = build_model('PLH', 104, [1, 2, 2], case.seed, role='S', num_bands=4,
        teacher_aligner_state={k[len('aligner.'):]: v for k, v in tensors.items() if k.startswith('aligner.')})
    model.load_state_dict(tensors, strict=True)
    return model.to(device).eval(), identity


def _verify_grid(cfg, grid, data, wd, release):
    case = validate_config(cfg, require_bound=True)
    expected = list(grid_steps(case.updates))
    if (grid.get('run_id') != case.run_id or grid.get('config_sha256') != object_sha(cfg)
            or grid.get('data_sha256') != object_sha(data) or grid.get('source_identity') != release
            or grid.get('campaign_id') != CAMPAIGN_ID
            or grid.get('expected_steps') != expected):
        raise ValueError('Official grid has wrong campaign/config/data/source/grid identity')
    seen = set()
    for record in grid['records']:
        step = record['update']
        if type(step) is not int or step not in expected or step in seen:
            raise ValueError('Duplicate/diagnostic/parent checkpoint in official candidates')
        seen.add(step)
        validate_record(record, 'GF2')
        folder = wd / 'candidates' / str(step)
        identity = read_json(folder / 'identity.json')
        if (identity != record['checkpoint_identity'] or sha256(folder / 'model.safetensors') != identity.get('model_sha256')
                or any(identity.get(k) != grid.get(k) for k in ('config_sha256', 'data_sha256',
                    'source_identity', 'reference_sha256'))):
            raise ValueError('Official metric/checkpoint bytes/provenance mismatch')
    return seen


def verify_endpoint_progress(cfg, state, data):
    from gfp40.stream import PairedBatchStream
    from gfp40.plan import family_for
    from qg40.exposure import validate_counts
    case = validate_config(cfg, require_bound=True)
    stream = PairedBatchStream(data['splits']['train']['count'], cfg['batch_size'], case.stream_seed,
                              family=cfg['gfp40']['family'])
    stream.load_state_dict(state['sampler'])
    if stream.completed_updates != case.updates or state['update'] != case.updates:
        raise ValueError('Endpoint sampler differs from exact completed optimizer updates')
    validate_counts(state.get('exposure_counts'), stream.n, case.updates, cfg['batch_size'])
    family = 'G025' if cfg['gfp40']['family'] == 'NATIVE' else cfg['gfp40']['family']
    gammas = family_for(family)['gammas']
    totals = state.get('gamma_counts', {})
    count = case.updates * cfg['batch_size']
    if set(totals) != {'drawn', 'effective'}:
        raise ValueError('Endpoint drawn/effective gamma exposure is missing')
    for name, values in totals.items():
        if (not torch.is_tensor(values) or values.dtype != torch.int64
                or values.shape != (len(gammas),) or bool((values < 0).any())
                or int(values.sum()) != count):
            raise ValueError('Endpoint gamma exposure differs from actual consumed updates: ' + name)
    if cfg['gfp40']['profile'] == 'MIX':
        if not torch.equal(totals['drawn'], totals['effective']):
            raise ValueError('MIX drawn and effective gamma counts differ')
    elif int(totals['effective'][gammas.index(1.)]) != count:
        raise ValueError('Native/control arm consumed nonnative effective gamma')
    groups = state.get('optimizer', {}).get('param_groups', [])
    if ([g.get('name') for g in groups] != ['U', 'A']
            or any(g.get('lr') != 0. for g in groups)
            or state.get('scheduler', {}).get('_last_lr') != [0., 0.]):
        raise ValueError('Endpoint U/A cosine optimizer does not end at zero')


def process(config_path, root=ROOT, device='cuda', deadline=None):
    """Repay saved evaluation debt only; never start/restart a training update."""
    from g20.data import build_dataset
    cfg = read_config(config_path)
    case = validate_config(cfg, require_bound=True)
    wd = resolved_path(cfg['work_dir'], root)
    cost_path = wd / 'official/postrun_costs.json'
    measured = read_json(cost_path) if cost_path.is_file() else dict(
        config_sha256=object_sha(cfg), evaluation_seconds=0., io_seconds=0.)
    if (measured.get('config_sha256') != object_sha(cfg) or any(not isinstance(measured.get(k), (int, float))
            or not math.isfinite(measured[k]) or measured[k] < 0 for k in ('evaluation_seconds', 'io_seconds'))):
        raise ValueError('Saved postrun measured-cost identity differs')
    def write(path, value):
        before = time.monotonic()
        atomic_json(path, value)
        measured['io_seconds'] += time.monotonic() - before
        atomic_json(cost_path, measured)
    training = read_json(wd / 'meta/training_status.json')
    actual = training.get('actual_updates')
    if type(actual) is not int or not 0 <= actual <= case.updates:
        raise ValueError('Invalid actual optimizer update count')
    data = read_json(resolved_path(cfg['gfp40']['dataset_manifest'], root))
    release = source_identity(root)
    start = read_json(wd / 'meta/training_start_manifest.json')
    if any(start.get(k) != v for k, v in dict(config_sha256=object_sha(cfg),
            source_identity=release, data_sha256=object_sha(data)).items()):
        raise ValueError('Run start provenance differs from postrun execution')
    grid_path = wd / 'official/raw_grid.json'
    grid = read_json(grid_path) if grid_path.is_file() else dict(campaign_id=CAMPAIGN_ID,
        run_id=case.run_id, horizon_updates=case.updates, expected_steps=list(grid_steps(case.updates)),
        records=[], **{k: start[k] for k in ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256')})
    seen = _verify_grid(cfg, grid, data, wd, release)
    if any(s > actual for s in seen):
        raise ValueError('Official grid is ahead of optimizer state')
    datasets = engine = None
    try:
        for step in grid_steps(case.updates):
            if step in seen or step > actual:
                continue
            check_deadline(deadline)
            folder = wd / 'candidates' / str(step)
            if not (folder / 'identity.json').is_file():
                continue
            before = time.monotonic()
            try:
                if datasets is None:
                    datasets = {key: build_dataset(data, key, root=root) for key in ('rr', 'fr', 'val')}
                    engine = FRMetrics(datasets['fr'])
                model, identity = load_candidate(cfg, folder, device, expected_source=release)
                if any(identity.get(k) != grid.get(k) for k in ('config_sha256', 'data_sha256',
                        'source_identity', 'reference_sha256')):
                    raise ValueError('Evaluation-debt checkpoint reference differs')
                try:
                    record = dict(update=step, checkpoint_identity=identity,
                        **evaluate_model(model, datasets, device, engine=engine, deadline=deadline))
                finally:
                    del model
            finally:
                measured['evaluation_seconds'] += time.monotonic() - before
                atomic_json(cost_path, measured)
            validate_record(record, 'GF2')
            grid['records'].append(record)
            grid['records'].sort(key=lambda r: r['update'])
            seen.add(step)
            write(wd / 'official' / f'candidate_{step}.json', record)
            write(grid_path, grid)
    except TimeoutError:
        pass
    pending = [s for s in grid_steps(case.updates) if s not in seen]
    complete = (actual == case.updates and training.get('training_complete') is True and not pending)
    grid.update(complete=complete, n_evaluated=len(seen))
    write(grid_path, grid)
    status = dict(status=('OFFICIAL_EVAL_COMPLETE' if complete else
                         'OFFICIAL_EVAL_PENDING' if actual == case.updates else training.get('status', 'PARTIAL')),
        official_complete=complete, actual_updates=actual, horizon_updates=case.updates,
        pending_steps=pending, n_evaluated=len(seen), sheet_uploaded=False)
    write(wd / 'official/evaluation_debt.json', status)
    if not complete:
        write(wd / 'official/postrun_status.json', status)
        return status
    endpoint = wd / 'candidates' / str(case.updates)
    endpoint_identity = read_json(endpoint / 'identity.json')
    from gfp40.assets import _finite_state, _fullstate
    full = _fullstate(endpoint / 'training_state.pt', endpoint_identity,
                      _finite_state(endpoint / 'model.safetensors', endpoint_identity))
    verify_endpoint_progress(cfg, full, data)
    parent = read_json(resolved_path(cfg['gfp40']['parent_manifest'], root)) if case.is_ft else None
    parent_compute = (parent or {}).get('parent_compute', {})
    reference_path = wd / 'meta/training_reference.json'
    actual_reference = read_json(reference_path) if reference_path.is_file() else {}
    if actual_reference and object_sha(actual_reference) != grid['reference_sha256']:
        raise ValueError('Saved actual training reference differs from candidate reference')
    teacher_compute = actual_reference.get('teacher_compute', {})
    costs = {key: training.get(key, 0.) for key in ('training_seconds', 'evaluation_seconds', 'io_seconds', 'diagnostic_seconds')}
    costs['evaluation_seconds'] += measured['evaluation_seconds']
    costs['io_seconds'] += measured['io_seconds']
    costs['postrun_evaluation_seconds'] = measured['evaluation_seconds']
    costs['postrun_io_seconds'] = measured['io_seconds']
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in costs.values()):
        raise ValueError('Nonfinite/missing measured run cost')
    context = dict(run_id=case.run_id, case_id=case.case_id, server=case.server, stage=case.stage,
        block_id=case.block_id, profile=cfg['gfp40']['profile'], config_sha256=object_sha(cfg),
        family=cfg['gfp40'].get('family'),
        execution_policy=cfg['gfp40']['execution_policy'],
        execution_policy_sha256=cfg['gfp40']['execution_policy_sha256'],
        registry_sha256=cfg['gfp40']['registry_sha256'],
        confirmation_family=cfg['gfp40']['execution_policy']['confirmation_family'],
        source_identity=release, data_sha256=object_sha(data),
        reference_sha256=grid['reference_sha256'], reference_key=case.reference_key,
        local_updates=case.updates, parent_step=case.parent_step or 0,
        lifetime_updates=case.lifetime_student_updates, parent_run_id=case.parent_run_id,
        parent_model_sha256=(parent or {}).get('parent_model_sha256'),
        init_mode=case.init_mode, optimizer_reset=case.is_ft,
        fresh_seed=case.model_seed, stream_seed=case.stream_seed,
        parent_compute=parent_compute, teacher_compute=teacher_compute, costs=costs,
        teacher_training_hours=(teacher_compute.get('training_seconds') or 0.) / 3600,
        actual_updates=full['update'], scheduler_end=full['scheduler']['last_epoch'],
        source_archive_sha256=cfg['gfp40'].get('source_archive_sha256'),
        calibration_reference_sha256=object_sha(actual_reference) if actual_reference else None,
        training_hours=costs['training_seconds'] / 3600,
        evaluation_hours=costs['evaluation_seconds'] / 3600,
        io_hours=costs['io_seconds'] / 3600,
        parent_training_hours=(parent_compute.get('training_seconds') or 0.) / 3600,
        started_at_utc=start.get('started_at_utc'),
        completed_at_utc=training.get('training_completed_at_utc') or endpoint_identity.get('saved_at_utc'),
        shared_trunk_cost_counted_once=bool(parent and parent.get('shared_trunk_cost_counted_once')))
    context['lineage_training_hours'] = context['training_hours'] + context['parent_training_hours']
    prior_path = wd / 'official/summary.json'
    prior = read_json(prior_path) if prior_path.is_file() else {}
    if prior and (prior.get('config_sha256') != object_sha(cfg)
                  or prior.get('run_id') != case.run_id):
        raise ValueError('Previously published official result belongs to a different run/config')
    numeric_evidence = object_sha([{key: row[key] for key in
        ('update', 'checkpoint_identity', 'rr', 'fr', 'val_ergas')}
        for row in sorted(grid['records'], key=lambda row: row['update'])])
    if prior and prior.get('official_evidence_sha256') != numeric_evidence:
        raise ValueError('Previously published numerical result changed; cannot preserve earlier lock eligibility')
    context['official_evidence_sha256'] = numeric_evidence
    # This is result availability, not optimizer finish time. A delayed lock
    # worker must not use metrics first computed after its registered cutoff.
    context['official_completed_at_utc'] = prior.get('official_completed_at_utc') or utcnow()
    summary = summarize_grid(grid['records'], case.updates, context)
    for label, record in summary['selections'].items():
        write(wd / 'official' / (label.lower() + '.json'), dict(context, **record))
    write(wd / 'official/summary.json', summary)
    write(wd / 'official/postrun_status.json', status)
    return summary
