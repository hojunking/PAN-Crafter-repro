"""Fixed-grid, same-checkpoint sensor selectors and measured C4 profiling."""
import copy
import math
from pathlib import Path
import time

import torch
import yaml

from qg40.common import (ROOT, atomic_json, check_deadline, load_checkpoint_model,
                        object_sha, read_json, sha256, source_identity, utcnow)
from qg40.plan import CAMPAIGN_ID, GRID_STEPS, case_for, sensor_spec
from qg40.evaluation import RR_KEYS, FR_KEYS

SELECTIONS = (('raw_max', 'RAW_MAX'), ('target', 'TARGET'), ('exact50k', 'EXACT50K'),
              ('rr_val_selected', 'RR_VAL_SELECTED'), ('e_min_diag', 'E_MIN_DIAG50'))


def validate_record(record, spec):
    spec = sensor_spec(spec) if isinstance(spec, str) else spec
    rr, fr = record['rr'], record['fr']
    if 'q8' in rr or not rr.get('official_complete'):
        raise ValueError('Official C4 Q4 required; Q8/proxy fallback is forbidden')
    if not all(math.isfinite(float(rr[k])) for k in RR_KEYS):
        raise ValueError('Missing/nonfinite official RR metric')
    if not all(math.isfinite(float(fr[k])) for k in FR_KEYS) or not math.isfinite(float(record['val_ergas'])):
        raise ValueError('Incomplete/nonfinite FR or validation metric')
    if (rr.get('n_scenes') != 20 or rr.get('crop') != '20:-21' or rr.get('q_block') != 32
            or fr.get('n_scenes') != 20 or fr.get('reference') != 'native_PAN'
            or fr.get('support') != 'full512' or fr.get('masking') is not False
            or fr.get('aggregation') != 'mean_per_scene_HQNR'
            or fr.get('hqnr_variant') != 'raw-original'):
        raise ValueError('Official native RR/FR support or aggregation changed')
    if any(r.get('sensor') != spec.sensor or r.get('num_bands') != 4 or r.get('max_dn') != spec.max_dn for r in (rr, fr)):
        raise ValueError('Mixed sensor/band/DN metrics')
    if record['checkpoint_identity']['update'] != record['update']:
        raise ValueError('Metrics and A/U checkpoint step differ')
    if fr.get('jqm') is not None and (not math.isfinite(float(fr['jqm'])) or not fr.get('jqm_variant')):
        raise ValueError('JQM must be finite and explicitly variant-labelled')


def select_records(records, sensor, expected_steps=GRID_STEPS):
    """Threshold is strict on full-precision raw HQNR; ties use the frozen order."""
    spec = sensor_spec(sensor) if isinstance(sensor, str) else sensor
    by_step = {int(r['update']): r for r in records}
    if len(by_step) != len(records) or sorted(by_step) != list(expected_steps):
        raise ValueError('Selection requires every unique fixed-grid candidate')
    for row in records:
        validate_record(row, spec)
    raw = max(records, key=lambda r: (r['fr']['hqnr'], -r['update']))
    eligible = [r for r in records if r['fr']['hqnr'] > spec.hqnr_threshold]
    target = min(eligible, key=lambda r: (r['rr']['ergas'], -r['rr']['scc'], -r['rr']['psnr'], r['update'])) if eligible else None
    return dict(raw_max=raw, target=target, exact50k=by_step[50000],
                rr_val_selected=min(records, key=lambda r: (r['val_ergas'], r['update'])),
                e_min_diag=min(records, key=lambda r: (r['rr']['ergas'], r['update'])),
                n_eligible=len(eligible), n_evaluated=len(records),
                target_status='official' if eligible else 'no_eligible',
                joint_pass=bool(target and target['rr']['ergas'] < spec.ergas_goal),
                strong_joint_pass=(bool(target and target['rr']['ergas'] < spec.ergas_strong_goal)
                                   if spec.ergas_strong_goal is not None else None))


def report_selection(label, record, context, **extra):
    if record is None:
        return dict(context, selection_id=label, official_complete=True, selection=None, **extra)
    return dict(context, selection_id=label, official_complete=True, step=record['update'],
                checkpoint_sha256=record['checkpoint_identity']['model_sha256'],
                checkpoint_identity=record['checkpoint_identity'], rr=record['rr'], fr=record['fr'],
                val_ergas=record['val_ergas'], eval_mode='A_ON', precision='fp32', **extra)


def profile_model(model, dataset, device, deadline=None):
    """Measure the actual C4 A+frequency frontend+U, retaining legacy MAC scale."""
    from thop import profile
    check_deadline(deadline)
    if dataset.spec.num_bands != 4:
        raise ValueError('QG40 cost must measure the four-band model')
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
        frontend_flops = 32 * (4 + 1 + 2) * h * w + h * w + 4 * h * w
        dev = torch.device(device)
        with torch.no_grad():
            for _ in range(3):
                model(*args)
            if dev.type == 'cuda':
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            start = time.perf_counter()
            for _ in range(10):
                check_deadline(deadline)
                model(*args)
            if dev.type == 'cuda':
                torch.cuda.synchronize(dev)
            milliseconds = (time.perf_counter() - start) * 100
        return dict(params_m=sum(p.numel() for p in model.parameters()) / 1e6,
                    flops_g=(macs + frontend_flops / 2) / 1e9,
                    backbone_aligner_thop_macs_g=macs / 1e9,
                    frontend_mac_equivalent_g=frontend_flops / 2 / 1e9,
                    flops_convention='THOP MAC scale + explicit C4 frontend MAC-equivalent; functional normalization/grid arithmetic may be omitted',
                    flops_is_estimate=True, frontend_included=True, profile_shape=[1, 1, h, w],
                    num_bands=4, sensor=dataset.spec.sensor, infer_ms=milliseconds,
                    mem_mb=torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == 'cuda' else None,
                    device=str(dev), torch=torch.__version__, precision='fp32',
                    scope='run profile: entire C4 A + synchronized PAN/LP/HP frontend + U',
                    LP_cache_generation='offline; excluded from inference')
    finally:
        model.train(was_training)


def validate_grid(run, cfg, grid, root=ROOT):
    from qg40.training import validate_config
    validate_config(cfg)
    case = case_for(run)
    if (grid.get('run_id') != run or grid.get('campaign_id') != CAMPAIGN_ID
            or cfg['qg40']['sensor'] != case.sensor or cfg['qg40']['num_bands'] != 4
            or grid.get('config_sha256') != object_sha(cfg)):
        raise ValueError('Official grid/config/campaign identity mismatch')
    data = read_json(Path(root) / cfg['qg40']['dataset_manifest'])
    if object_sha(data) != grid.get('data_sha256') or data.get('sensor') != case.sensor:
        raise ValueError('Official data manifest mismatch')
    wd = Path(root) / 'work_dir' / run
    for record in grid['records']:
        folder = wd / 'candidates' / str(record['update'])
        identity = read_json(folder / 'identity.json')
        if (identity != record['checkpoint_identity'] or identity['model_sha256'] != sha256(folder / 'model.safetensors')
                or any(identity.get(k) != grid.get(k) for k in ('config_sha256', 'data_sha256', 'source_identity'))):
            raise ValueError('Candidate bytes/config/data/evaluator provenance mismatch')
    return data


def process(run, device='cuda', deadline=None, upload=False, upload_only=False, root=ROOT):
    root = Path(root)
    case = case_for(run)
    wd = root / 'work_dir' / run
    path = wd / 'official/postrun_status.json'
    if upload_only:
        status = read_json(path)
        if not status.get('official_complete'):
            raise ValueError('Cannot upload incomplete official results')
        return _upload(run, status, path, root) if upload else 0
    check_deadline(deadline)
    cfg = yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text())
    from qg40.training import runtime_context
    _window, deadline = runtime_context(cfg, root, deadline, resume=True)
    check_deadline(deadline)
    grid = read_json(wd / 'official/raw_grid.json')
    training = read_json(wd / 'meta/training_status.json')
    if training.get('actual_updates') != 50000 or not training.get('training_complete'):
        raise ValueError('Official postrun requires completed exact50K')
    data = validate_grid(run, cfg, grid, root)
    if not grid.get('complete'):
        # Evaluation debt can be repaid from the saved fixed-grid checkpoints;
        # this path never calls an optimizer or replays training updates.
        if grid['source_identity'] != source_identity(root):
            raise ValueError('Pending evaluation release differs from saved candidates')
        from qg40.data import build_dataset
        from qg40.evaluation import FRMetrics, evaluate_model
        datasets = None
        engine = None
        existing = {r['update'] for r in grid['records']}
        try:
            for step in GRID_STEPS:
                if step in existing:
                    continue
                check_deadline(deadline)
                folder = wd / 'candidates' / str(step)
                if not (folder / 'identity.json').is_file() or not (folder / 'model.safetensors').is_file():
                    continue
                if datasets is None:
                    datasets = {s: build_dataset(data, s, root=root) for s in ('rr', 'fr', 'val')}
                    engine = FRMetrics(datasets['fr'])
                model, identity = load_checkpoint_model(cfg, folder, device, expected_source=grid['source_identity'])
                try:
                    metrics = evaluate_model(model, datasets, device, engine=engine, deadline=deadline)
                finally:
                    del model
                record = dict(update=step, checkpoint_identity=identity, **metrics)
                validate_record(record, case.sensor)
                grid['records'].append(record)
                grid['records'].sort(key=lambda r: r['update'])
                grid['n_evaluated'] = len(grid['records'])
                atomic_json(wd / 'official' / f'candidate_{step}.json', record)
                atomic_json(wd / 'official/raw_grid.json', grid)
        except TimeoutError:
            pass
        grid['complete'] = sorted(r['update'] for r in grid['records']) == list(GRID_STEPS)
        atomic_json(wd / 'official/raw_grid.json', grid)
        if not grid['complete']:
            atomic_json(path, dict(status='OFFICIAL_EVAL_PENDING', official_complete=False, actual_updates=50000,
                                   n_evaluated=len(grid['records']), n_pending=50-len(grid['records']), sheet_uploaded=False))
            return 0
    if grid['source_identity'] != source_identity(root):
        raise ValueError('Current evaluator differs from recorded numerical release')
    selected = select_records(grid['records'], case.sensor)
    spec = sensor_spec(case.sensor)
    context = dict(campaign_id=CAMPAIGN_ID, run_id=run, sensor=case.sensor, role=case.role,
                   profile=case.profile, teacher_alias=case.teacher_alias, config_sha256=object_sha(cfg),
                   data_sha256=grid['data_sha256'], source_identity=grid['source_identity'], normal_same_step_A_U=True)
    for key, label in SELECTIONS:
        extra = dict(n_evaluated=50)
        if key == 'target':
            extra.update(n_eligible=selected['n_eligible'], target_status=selected['target_status'],
                         hqnr_threshold=spec.hqnr_threshold, threshold_comparison='>',
                         ergas_goal=spec.ergas_goal, ergas_strong_goal=spec.ergas_strong_goal,
                         selector_order=['ergas', '-scc', '-psnr', 'step'], joint_pass=selected['joint_pass'],
                         strong_joint_pass=selected['strong_joint_pass'], test_aware=True)
        if key == 'e_min_diag':
            extra.update(n_candidates=50, development_oracle=True, independent_test=False)
        filename = 'target_selection' if key == 'target' else key
        atomic_json(wd / 'official' / f'{filename}.json', report_selection(label, selected[key], context, **extra))
    cost_path = wd / 'official/profile.json'
    if cost_path.exists():
        cost = read_json(cost_path)
        if cost.get('config_sha256') != object_sha(cfg) or cost.get('source_identity') != grid['source_identity'] or cost.get('num_bands') != 4:
            raise ValueError('Measured C4 profile provenance differs')
    else:
        from qg40.data import build_dataset
        dataset = build_dataset(data, 'rr', root=root)
        model, _ = load_checkpoint_model(cfg, wd / 'candidates' / str(selected['raw_max']['update']), device,
                                         expected_source=grid['source_identity'])
        cost = profile_model(model, dataset, device, deadline)
        cost.update(config_sha256=object_sha(cfg), source_identity=grid['source_identity'])
        atomic_json(cost_path, cost)
        del model
    status = dict(status='OFFICIAL_EVAL_COMPLETE', official_complete=True, sheet_uploaded=False,
                  config_sha256=object_sha(cfg), source_identity=grid['source_identity'], actual_updates=50000,
                  n_evaluated=50, completed_at_utc=utcnow(), target_status=selected['target_status'])
    atomic_json(path, status)
    return _upload(run, status, path, root) if upload else 0


def screen_result(run, root=ROOT):
    """A policy-ready row only after revalidating official same-step artifacts."""
    from qg40.upload import row_values
    values = row_values(run, root)
    case = case_for(run)
    eligible = values['Target status'] != 'no_eligible'
    return dict(seed=case.student_seed, target_eligible=eligible,
                target_e=values['Target ERGAS↓'] if eligible else None,
                target_h=values['Target HQNR(raw)↑'] if eligible else None,
                raw_max_h=values['RAW_MAX HQNR(raw)↑'], e50=values['Exact50K ERGAS↓'],
                actual_updates=values['QG40 actual updates'], n_evaluated=values['QG40 n evaluated'],
                official_complete=True, same_checkpoint_verified=True)


def _upload(run, status, path, root):
    try:
        from qg40.upload import upload_run
        receipt = upload_run(run, root=root, activated=True)
        status.update(status='UPLOAD_VERIFIED', sheet_uploaded=True, upload_receipt=receipt)
        status.pop('upload_error', None)
    except Exception as exc:
        status.update(status='OFFICIAL_EVAL_COMPLETE_UPLOAD_PENDING', sheet_uploaded=False,
                      upload_error=f'{type(exc).__name__}: {exc}')
    atomic_json(path, status)
    return 0
