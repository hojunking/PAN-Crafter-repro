"""Fixed-grid official evaluation, provenance validation, and append-only results."""
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

from fh12.common import ROOT, atomic_json, object_sha, read_json, sha256, utcnow, check_deadline
from ablr2.evaluation import FRMetrics, evaluate_model, rr_keys, FR_KEYS, validate_signed_ds

GRID_STEPS = tuple(range(1010, 49491, 1010)) + (50000,)
SELECTIONS = (('rr_val_selected', 'RR_VAL_SELECTED'), ('exact50k', 'EXACT50K'),
              ('raw_max', 'RAW_MAX'), ('e_min_diag', 'E_MIN_DIAG50'))


def _case(cfg):
    from ablr2.plan import validate_config
    return validate_config(cfg)


def _workdir(cfg, root):
    case = _case(cfg)
    expected = Path(root) / 'work_dir/ablr2' / case.sensor / case.server_id / 'runs' / case.run_id
    actual = Path(root) / cfg['work_dir']
    if actual.resolve() != expected.resolve():
        raise ValueError('ABLR2 run must remain inside its sensor/server namespace')
    return expected


def find_config(run, root=ROOT):
    from ablr2.common import read_config
    from ablr2.plan import LANES
    paths = [Path(root) / 'work_dir/ablr2' / sensor / server / 'runs' / run / 'meta/config.resolved.yaml'
             for server, sensor in LANES.items()]
    paths = [p for p in paths if p.is_file()]
    if len(paths) != 1 or Path(run).name != run:
        raise ValueError('Expected one local ABLR2 resolved run, without path aliases')
    cfg = read_config(paths[0])
    if _case(cfg).run_id != run or _workdir(cfg, root) / 'meta/config.resolved.yaml' != paths[0]:
        raise ValueError('Resolved run identity mismatch')
    return cfg


def validate_record(record, spec):
    from ablr2.evaluation import _spec
    spec = _spec(spec)
    rr, fr = record['rr'], record['fr']
    if not rr.get('official_complete') or ('q4' if spec.num_bands == 8 else 'q8') in rr:
        raise ValueError('Genuine sensor Q8/Q4 required; proxy or wrong-band metric forbidden')
    if not all(math.isfinite(float(rr[k])) for k in rr_keys(spec)):
        raise ValueError('Missing/nonfinite official RR metrics')
    if not all(math.isfinite(float(fr[k])) for k in FR_KEYS) or not math.isfinite(float(record['val_ergas'])):
        raise ValueError('Missing/nonfinite official FR or validation metrics')
    if (rr.get('n_scenes') != 20 or rr.get('crop') != '20:-21' or rr.get('q_block') != 32
            or fr.get('n_scenes') != 20 or fr.get('reference') != 'native_PAN'
            or fr.get('support') != 'full512' or fr.get('masking') is not False
            or fr.get('aggregation') != 'mean_per_scene_HQNR' or fr.get('hqnr_variant') != 'raw-original'):
        raise ValueError('Official support/reference or aggregation changed')
    if any(x.get('sensor') != spec.sensor or x.get('num_bands') != spec.num_bands or x.get('max_dn') != spec.max_dn for x in (rr, fr)):
        raise ValueError('Metrics sensor/bands/DN identity differs')
    if type(record['update']) is not int or record['checkpoint_identity']['update'] != record['update']:
        raise ValueError('Metrics and checkpoint completed update differ')
    if fr.get('jqm') is not None and (not math.isfinite(float(fr['jqm'])) or not fr.get('jqm_variant')):
        raise ValueError('JQM must be finite with an explicit variant')
    validate_signed_ds(fr)
    if fr['signed_ds']['operand_shape'] != [20, spec.num_bands]:
        raise ValueError('Signed Ds must retain every actual sensor band')
    rows = rr.get('per_scene', [])
    if len(rows) != 20 or any(not np.isclose(rr[k], np.mean([r[k] for r in rows]), rtol=0, atol=1e-12) for k in rr_keys(spec)):
        raise ValueError('RR must aggregate all twenty per-scene metrics')


def select_records(records, sensor):
    steps = [r['update'] for r in records]
    if any(type(step) is not int for step in steps) or len(set(steps)) != 50 or sorted(steps) != list(GRID_STEPS):
        raise ValueError('Selection requires every unique fixed-grid candidate')
    for row in records:
        validate_record(row, sensor)
    return dict(rr_val_selected=min(records, key=lambda r: (r['val_ergas'], r['update'])),
                exact50k=next(r for r in records if r['update'] == 50000),
                raw_max=min(records, key=lambda r: (-r['fr']['hqnr'], -r['rr']['scc'], r['rr']['ergas'], r['update'])),
                e_min_diag=min(records, key=lambda r: (r['rr']['ergas'], r['update'])))


def empty_grid(cfg, start):
    case = _case(cfg)
    return dict(campaign_id=start['campaign_id'], run_id=case.run_id, sensor=case.sensor,
                expected_steps=list(GRID_STEPS), records=[], n_evaluated=0, complete=False,
                **{k: start[k] for k in ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256')})


def validate_grid(run, cfg, grid, root=ROOT):
    case, wd = _case(cfg), _workdir(cfg, root)
    if (case.run_id != run or grid.get('run_id') != run or grid.get('campaign_id') != case.campaign_id
            or grid.get('sensor') != case.sensor or grid.get('config_sha256') != object_sha(cfg)
            or grid.get('expected_steps') != list(GRID_STEPS)):
        raise ValueError('Grid config/campaign/candidate identity differs')
    data = read_json(Path(root) / cfg['ablr2']['dataset_manifest'])
    if (object_sha(data) != grid.get('data_sha256') or data.get('sensor') != case.sensor
            or data.get('num_bands') != case.num_bands or data.get('max_pixel') != case.max_dn):
        raise ValueError('Grid sensor-bound data manifest changed')
    records = grid['records']
    steps = [r['update'] for r in records]
    if (any(type(x) is not int or x not in GRID_STEPS for x in steps) or len(set(steps)) != len(steps)
            or grid.get('n_evaluated') != len(records) or grid.get('complete') is not (sorted(steps) == list(GRID_STEPS))):
        raise ValueError('False grid completeness or duplicate/diagnostic candidates')
    for record in records:
        validate_record(record, case.sensor)
        folder = wd / 'candidates' / str(record['update'])
        identity = read_json(folder / 'identity.json')
        if (identity != record['checkpoint_identity'] or identity.get('model_sha256') != sha256(folder / 'model.safetensors')
                or any(identity.get(k) != grid.get(k) for k in ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256'))):
            raise ValueError('Grid checkpoint bytes/provenance mismatch')
    return data


def evaluate_candidate(cfg, folder, datasets, device, engine=None, deadline=None, root=ROOT, model=None):
    from ablr2.common import load_checkpoint_model
    folder, case = Path(folder), _case(cfg)
    identity = read_json(folder / 'identity.json')
    if identity.get('update') not in GRID_STEPS or identity.get('config_sha256') != object_sha(cfg):
        raise ValueError('Unregistered official candidate/config')
    if identity.get('model_sha256') != sha256(folder / 'model.safetensors'):
        raise ValueError('Candidate bytes differ before evaluation')
    saved_path = _workdir(cfg, root) / 'official' / f'candidate_{identity["update"]}.json'
    if saved_path.is_file():
        saved = read_json(saved_path)
        validate_record(saved, case.sensor)
        if saved['checkpoint_identity'] != identity:
            raise ValueError('Previously recorded candidate has different provenance')
        return saved  # Recover a candidate-JSON/grid-write crash without replacing an observation.
    if model is None:
        model, checked = load_checkpoint_model(cfg, folder, device, expected_source=identity['source_identity'])
        if checked != identity:
            raise ValueError('Candidate identity changed during load')
    else:
        from fh12.model import state_hash
        if identity.get('state_hash') != state_hash(model.state_dict()):
            raise ValueError('In-memory evaluation model differs from saved candidate')
    result = dict(update=identity['update'], checkpoint_identity=identity,
                  **evaluate_model(model, datasets, device, engine, deadline))
    result['eval_mode'] = getattr(model, 'aligner_mode', 'UNKNOWN')
    validate_record(result, case.sensor)
    return result


def record_candidate(cfg, record, root=ROOT):
    from ablr2.common import immutable_json
    wd = _workdir(cfg, root)
    start = read_json(wd / 'meta/training_start_manifest.json')
    path = wd / 'official/raw_grid.json'
    grid = read_json(path) if path.exists() else empty_grid(cfg, start)
    validate_grid(_case(cfg).run_id, cfg, grid, root)
    validate_record(record, _case(cfg).sensor)
    old = next((r for r in grid['records'] if r['update'] == record['update']), None)
    if old is not None and old != record:
        raise ValueError('An existing candidate observation cannot be replaced')
    immutable_json(wd / 'official' / f'candidate_{record["update"]}.json', record)
    if old is None:
        grid['records'].append(record)
    grid['records'].sort(key=lambda r: r['update'])
    grid.update(n_evaluated=len(grid['records']), complete=[r['update'] for r in grid['records']] == list(GRID_STEPS))
    validate_grid(_case(cfg).run_id, cfg, grid, root)
    atomic_json(path, grid)
    best = min(grid['records'], key=lambda r: (-r['fr']['hqnr'], -r['rr']['scc'], r['rr']['ergas'], r['update']))
    atomic_json(wd / 'best_hqnr_meta.json', dict(step=best['update'], hqnr=best['fr']['hqnr'], scc=best['rr']['scc'],
        ergas=best['rr']['ergas'], checkpoint_sha256=best['checkpoint_identity']['model_sha256'],
        selector='HQNR_MAX_SCC_MAX_ERGAS_MIN_STEP_MIN', test_aware=True, primary=False, partial_grid=not grid['complete']))
    return grid


def verify_fullstate(cfg, root=ROOT):
    from fh12.model import state_hash
    from safetensors.torch import load_file
    folder = _workdir(cfg, root) / 'candidates/50000'
    identity = read_json(folder / 'identity.json')
    if (identity.get('update') != 50000 or identity.get('full_state') is not True
            or identity.get('config_sha256') != object_sha(cfg)
            or identity.get('model_sha256') != sha256(folder / 'model.safetensors')
            or identity.get('training_state_sha256') != sha256(folder / 'training_state.pt')):
        raise ValueError('Exact50K full-state identity or bytes differ')
    state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=False)
    if (state.get('update') != 50000 or state.get('scheduler', {}).get('last_epoch') != 50000
            or state.get('config_sha256') != object_sha(cfg) or state_hash(state['model_state']) != identity.get('state_hash')
            or state_hash(load_file(str(folder / 'model.safetensors'))) != identity.get('state_hash')):
        raise ValueError('Exact50K scheduler/model state differs')
    for key in ('data_sha256', 'source_identity', 'reference_sha256'):
        if state.get(key) != identity.get(key):
            raise ValueError('Exact50K full-state provenance differs: ' + key)
    return dict(model_sha256=identity['model_sha256'], training_state_sha256=identity['training_state_sha256'])


def profile_model(model, dataset, device, deadline=None):
    from thop import profile
    check_deadline(deadline)
    _gt, _lms, ms, lp, pan, _meta = dataset.base(0)
    args = tuple(x.unsqueeze(0).to(device) for x in (pan, ms, lp))
    was_training = model.training
    model.eval()
    scratch = copy.deepcopy(model)
    try:
        with torch.no_grad():
            macs, _ = profile(scratch, inputs=args, verbose=False)
        del scratch
        h, w = pan.shape[-2:]
        bands = dataset.spec.num_bands
        mode = getattr(model, 'aligner_mode', 'UNKNOWN')
        frontend_flops = 32 * (bands + 1 + (0 if mode == 'IDENTITY' else 2)) * h * w + (bands + 1) * h * w
        dev = torch.device(device)
        with torch.no_grad():
            for _ in range(3):
                model(*args)
            if dev.type == 'cuda':
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            started = time.perf_counter()
            for _ in range(10):
                check_deadline(deadline)
                model(*args)
            if dev.type == 'cuda':
                torch.cuda.synchronize(dev)
        return dict(sensor=dataset.spec.sensor, num_bands=bands, precision='fp32', device=str(dev),
                    params_m=sum(p.numel() for p in model.parameters()) / 1e6,
                    trainable_params_m=sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
                    aligner_params_m=sum(p.numel() for p in model.aligner.parameters()) / 1e6,
                    alignment_mode=mode, flops_g=(macs + frontend_flops / 2) / 1e9,
                    infer_ms=(time.perf_counter() - started) * 100,
                    mem_mb=torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == 'cuda' else None,
                    flops_is_estimate=True, flops_convention='THOP MAC + explicit sensor frontend MAC-equivalent; functional norm/grid arithmetic may be omitted',
                    scope='deployed Student/Teacher own A + frequency frontend + U only; Teacher train/calibration cost reported separately',
                    masked_parameter_note='Stored stem shape is matched; masked channels do not imply identical active capacity',
                    profile_shape=[1, 1, h, w], LP_cache_generation='offline; excluded from inference')
    finally:
        model.train(was_training)


def selection_report(cfg, grid, key, record):
    case, label = _case(cfg), dict(SELECTIONS)[key]
    return dict(campaign_id=grid['campaign_id'], run_id=case.run_id, sensor=case.sensor, server_id=case.server_id,
                case_id=case.case_id, role=case.role, phase=case.phase, wave=case.wave, sweep=case.sweep,
                recipe_id=case.recipe_id, recipe_revision=case.recipe_revision, teacher_seed=case.teacher_seed,
                student_seed=case.student_seed, teacher_reference_id=case.reference_id,
                selection_id=label, official_complete=True, n_evaluated=50, horizon_updates=50000,
                primary=key == 'rr_val_selected', secondary_primary=key == 'exact50k',
                test_aware=key in ('raw_max', 'e_min_diag'), development_mode='TEST_AWARE_DEV', independent_test=False,
                step=record['update'], checkpoint_sha256=record['checkpoint_identity']['model_sha256'],
                precision='fp32', eval_mode=record.get('eval_mode'),
                **{k: grid[k] for k in ('config_sha256', 'data_sha256', 'source_identity', 'reference_sha256')},
                **{k: record[k] for k in ('rr', 'fr', 'val_ergas', 'checkpoint_identity')})


def process(run, device='cuda', deadline=None, upload=False, upload_only=False, root=ROOT):
    from ablr2.common import source_identity, load_checkpoint_model, immutable_json, apply_runtime_policy, assert_compatible_source
    from ablr2.data import build_dataset
    apply_runtime_policy(root)
    cfg = find_config(run, root)
    case, wd = _case(cfg), _workdir(cfg, root)
    status_path = wd / 'official/postrun_status.json'
    if not upload_only:
        training = read_json(wd / 'meta/training_status.json')
        if training.get('actual_updates') != 50000 or training.get('training_complete') is not True:
            raise ValueError('Official postrun requires completed fresh50K')
        start = read_json(wd / 'meta/training_start_manifest.json')
        assert_compatible_source(start.get('source_identity'),source_identity(root),root,case.server_id)
        if start.get('config_sha256') != object_sha(cfg):
            raise ValueError('Postrun source/runtime/config differs from training')
        path = wd / 'official/raw_grid.json'
        grid = read_json(path) if path.exists() else empty_grid(cfg, start)
        data = validate_grid(run, cfg, grid, root)
        verify_fullstate(cfg, root)
        datasets = engine = None
        for step in GRID_STEPS:
            if any(r['update'] == step for r in grid['records']):
                continue
            check_deadline(deadline)
            if datasets is None:
                datasets = {s: build_dataset(data, s, root=root) for s in ('rr', 'fr', 'val')}
                engine = FRMetrics(datasets['fr'])
            record = evaluate_candidate(cfg, wd / 'candidates' / str(step), datasets, device, engine, deadline, root)
            grid = record_candidate(cfg, record, root)
            print(f'[ABLR2 official {step}] HQNR={record["fr"]["hqnr"]:.8f} SCC={record["rr"]["scc"]:.8f} ERGAS={record["rr"]["ergas"]:.8f}', flush=True)
        selected = select_records(grid['records'], case.sensor)
        for key, _label in SELECTIONS:
            immutable_json(wd / 'official' / f'{key}.json', selection_report(cfg, grid, key, selected[key]))
        cost_path = wd / 'official/profile.json'
        if not cost_path.exists():
            model, _identity = load_checkpoint_model(cfg, wd / 'candidates/50000', device, expected_source=grid['source_identity'])
            cost = profile_model(model, build_dataset(data, 'rr', root=root), device, deadline)
            cost.update(config_sha256=object_sha(cfg), source_identity=grid['source_identity'])
            immutable_json(cost_path, cost)
        status = dict(status='OFFICIAL_EVAL_COMPLETE', official_complete=True, n_evaluated=50,
                      actual_updates=50000, sheet_uploaded=False, completed_at_utc=utcnow(),
                      config_sha256=object_sha(cfg), source_identity=grid['source_identity'], data_sha256=grid['data_sha256'])
        if not status_path.exists():
            atomic_json(status_path, status)
    status = read_json(status_path)
    if not status.get('official_complete'):
        raise ValueError('Incomplete official postrun cannot be uploaded')
    if upload:
        from ablr2.upload import upload_run
        try:
            upload_run(run, root, activated=True)
        except Exception as exc:
            status.update(sheet_uploaded=False, upload_error=f'{type(exc).__name__}: {exc}')
            atomic_json(status_path, status)
            return 2
        status.update(sheet_uploaded=True)
        status.pop('upload_error', None)
        atomic_json(status_path, status)
    return 0
