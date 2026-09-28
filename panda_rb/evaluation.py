"""New B01 measurements only: native exact/val and paired exact50K stress.

No legacy selector, historical report, training queue, or Sheet is modified.
Every completed report binds source/checkpoint/data identities and raw files.
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import yaml

from fh12.data import build_dataset
from fh12.evaluation import FRMetrics, infer, native_gt, rr_metrics
from fh12.model import build_model
from reporting_extra.evaluation import (rr_metrics as rr_extra, fr_metrics as fr_extra,
                                         evaluator_identity as _base_evaluator_identity, JQM_VARIANT, _precision)
from panda_rb.common import ROOT, atomic_json, object_sha, read_json, sha256, utcnow, source_identity
from panda_rb.plan import case_for, run_dir, shift_grid, VAL_GRID
from panda_rb.stress import (MODES, RAW_SUBSET, STRESS_KEYS, array_digest, validate_grid,
                             stress_forward, relative_response, geometric_coverage,
                             stress_metrics, warp_convention_test)


def evaluator_identity(root=ROOT):
    """Bind the actually imported external FR interp23tap, not just repo code."""
    from tools.metrics.eval_fr import load_dlpan
    directory = os.environ.get('PANCRAFTER_DLPAN', str(Path(root).parent / 'DLPan-Toolbox'))
    module = load_dlpan(directory)
    path = Path(module.__file__).resolve()
    before = sha256(path)
    probe = np.arange(64, dtype=np.float64).reshape(8, 8, 1)
    result = module.interp23tap(probe, 4)
    if result.shape != (32, 32, 1) or not np.isfinite(result).all():
        raise ValueError('External DLPan interp23tap cannot execute the native FR reference recipe')
    if sha256(path) != before:
        raise ValueError('External DLPan source changed during evaluator validation')
    identity = dict(_base_evaluator_identity(root))
    identity.update(external_wald_sha256=before, external_wald_smoke='finite8to32_ratio4_PASS')
    identity['content_sha256'] = object_sha(dict(files=identity['files'], external_wald_sha256=before))
    return identity


def seal(value):
    return dict(value, payload_sha256=object_sha(value))


def validate_artifacts(report, directory):
    body = {k: v for k, v in report.items() if k != 'payload_sha256'}
    if report.get('payload_sha256') != object_sha(body) or not report.get('complete'):
        raise ValueError('Incomplete or modified B01 evaluation report')
    for name, digest in report.get('file_hashes', {}).items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe relative evaluation artifact path')
        if sha256(Path(directory) / path) != digest:
            raise ValueError('B01 evaluation artifact changed: ' + name)
    return report


def _atomic_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Refusing to replace different B01 evidence: ' + str(path))
        return
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def save_array(path, value):
    value = np.asarray(value, dtype=np.float32)
    stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
    _atomic_bytes(path, stream.getvalue())
    return dict(file_sha256=sha256(path), prediction_digest=array_digest(value),
                dtype='float32', shape=list(value.shape))


def save_csv(path, rows):
    if not rows:
        raise ValueError('Empty evaluation scene table')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def selected_checkpoints(manifest):
    """The only new selections: exact50K and validation ERGAS / earliest tie."""
    if not manifest.get('complete') or manifest.get('actual_updates') != 50000:
        raise ValueError('B01 native evaluation requires completed 50000 updates')
    records = manifest.get('validation_records', [])
    if [r['update'] for r in records] != list(VAL_GRID):
        raise ValueError('Incomplete or altered 50-point validation grid')
    if not all(np.isfinite(r['val_ergas']) for r in records):
        raise ValueError('Nonfinite validation selection record')
    winner = min(records, key=lambda r: (r['val_ergas'], r['update']))
    primary, secondary = manifest['primary'], manifest['secondary']
    for item, name, step, folder in ((primary, 'EXACT_50000', 50000, 'exact50000'),
                                     (secondary, 'RR_VAL_ERGAS_MIN', winner['update'], 'val_selected')):
        if (item.get('selection_id') != name or item.get('update') != step
                or item.get('directory') != folder or not item.get('checkpoint_sha256')):
            raise ValueError('Selection differs from predeclared exact/validation rule')
    same = primary['checkpoint_sha256'] == secondary['checkpoint_sha256']
    if bool(secondary.get('alias_of') == 'EXACT_50000') != same:
        raise ValueError('Same checkpoint SHA must be an explicit selection alias')
    return primary, secondary


def _context(run_id, binding_path, root=ROOT):
    from panda_rb.binding import load_binding
    row = case_for(run_id, root); wd = run_dir(run_id, root)
    cfg = yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text())
    if (cfg.get('panda_rb', {}).get('run_id') != run_id or cfg['seed'] != row['seed']
            or cfg['panda_rb'].get('case_id') != row['case_id']):
        raise ValueError('B01 evaluation resolved config belongs to another run')
    teacher, _teacher_cfg, binding, data, _q = load_binding(binding_path, device='cpu')
    del teacher
    manifest_path = wd / 'checkpoints/selection_manifest.json'
    manifest = read_json(manifest_path)
    selected = selected_checkpoints(manifest)
    source = source_identity(root)
    identities = {}
    for item in selected:
        path = wd / 'checkpoints' / item['directory']
        if not path.resolve().is_relative_to((wd / 'checkpoints').resolve()):
            raise ValueError('Selected checkpoint escapes this new run')
        identity = read_json(path / 'identity.json')
        if (identity.get('model_sha256') != sha256(path / 'model.safetensors')
                or identity['model_sha256'] != item['checkpoint_sha256']
                or identity.get('update') != item['update']
                or identity.get('config_sha256') != object_sha(cfg)
                or identity.get('source_identity') != source):
            raise ValueError('Selected B01 checkpoint bytes/config/source/update mismatch')
        expected_binding = object_sha(binding)
        if identity.get('binding_sha256') != expected_binding:
            raise ValueError('B01 checkpoint was trained with a different frozen F1 binding')
        identities[item['directory']] = identity
    for split in ('rr', 'fr'):
        if data['splits'][split]['count'] != 20:
            raise ValueError('Native B01 needs all 20 RR and FR scenes')
    context = dict(run_id=run_id, case=row, config_sha256=object_sha(cfg),
                   binding_sha256=expected_binding, data_sha256=object_sha(data),
                   binding_common_sha256=binding.get('common_sha256'),
                   data_content_identity={s: {k: v for k, v in info.items() if k in
                       ('sha256', 'lpan_sha256', 'sample_order_sha256', 'count', 'shapes')}
                       for s, info in data['splits'].items()},
                   selection_manifest_sha256=sha256(manifest_path), source_identity=source,
                   evaluator_identity=evaluator_identity(root))
    return dict(wd=wd, cfg=cfg, data=data, selected=selected, identities=identities, context=context)


def _model(ctx, selection, device):
    from safetensors.torch import load_file
    path = ctx['wd'] / 'checkpoints' / selection['directory']
    state = load_file(str(path / 'model.safetensors'))
    aligner = {k[8:]: v for k, v in state.items() if k.startswith('aligner.')}
    cfg = ctx['cfg']; ma = cfg['model_args']
    model, _ = build_model(cfg['panda_rb']['input_layout'], ma['hidden_size'], ma['depth'], cfg['seed'],
                           role='S', teacher_aligner_state=aligner)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def _raw_checks(pred, split):
    expected = (20, 8, 256, 256) if split == 'rr' else (20, 8, 512, 512)
    if pred.shape != expected or pred.dtype != np.float32:
        raise ValueError(f'Invalid {split} raw shape/dtype: {pred.shape}/{pred.dtype}')
    if not np.isfinite(pred).all() or pred.min() < 0 or pred.max() > 2047:
        raise ValueError('Raw prediction is not finite DN0..2047')


def _native_selection(ctx, selection, directory, datasets, device, engine):
    model = _model(ctx, selection, device)
    per_scene, raw = [], {}
    try:
        with _precision(ctx['context']['source_identity'], device):
            rr_pred, rr_c = infer(model, datasets['rr'], device, batch_size=1)
            fr_pred, fr_c = infer(model, datasets['fr'], device, batch_size=1)
        _raw_checks(rr_pred, 'rr'); _raw_checks(fr_pred, 'fr')
        truth = native_gt(datasets['rr'])
        rr = rr_metrics(rr_pred, truth); extra_rr = rr_extra(rr_pred, truth)
        fr = engine(fr_pred)
        extra_fr = fr_extra(fr_pred, datasets['fr'].arrays['ms'], datasets['fr'].arrays['pan'])
        for split, folder, pred, correction, normal, extra in (
                ('rr', 'reduced', rr_pred, rr_c, rr, extra_rr),
                ('fr', 'full', fr_pred, fr_c, fr, extra_fr)):
            raw[split] = []
            for i in range(20):
                path = directory / 'Ours_raw/WV3' / folder / 'pred' / f'testimg{i}.npy'
                metadata = save_array(path, pred[i])
                metadata.update(scene_index=i, h5_row_index=i, path=str(path.relative_to(ctx['wd'])))
                raw[split].append(metadata)
                metrics = {k: v for k, v in {**normal['per_scene'][i], **extra['per_scene'][i]}.items()
                           if isinstance(v, (float, int))}
                per_scene.append(dict(split=split, scene_index=i, source_h5_row=i, **metrics,
                                      correction_dy=float(correction[i, 0]), correction_dx=float(correction[i, 1]),
                                      prediction_digest=metadata['prediction_digest']))
        # Same source raw H5 holds native GT/PAN/MS; do not duplicate gigabytes.
        references = {s: {k: ctx['data']['splits'][s][k] for k in
                          ('dataroot', 'sha256', 'lpan_path', 'lpan_sha256', 'sample_order_sha256')}
                      for s in ('rr', 'fr')}
        atomic_json(directory / 'references/manifest.json', references)
        save_csv(directory / 'per_scene.csv', per_scene)
        return dict(selection_id=selection['selection_id'], update=selection['update'],
                    checkpoint_sha256=selection['checkpoint_sha256'], alias_of=None,
                    rr=rr, fr=fr, supplemental_rr=extra_rr, supplemental_fr=extra_fr,
                    raw=raw, n_scenes={'rr': 20, 'fr': 20}, coordinate_frame='native_MS',
                    eval_mode='A_ON', fr_masking=False, jqm_variant=JQM_VARIANT)
    finally:
        del model


def _file_hashes(directory, wd):
    return {str(p.relative_to(wd)): sha256(p) for p in sorted(Path(directory).rglob('*'))
            if p.is_file() and p.name not in ('metrics.json', 'completion.json')}


def evaluate_native(run_id, binding_path, root=ROOT, device='cuda'):
    ctx = _context(run_id, binding_path, root); folder = ctx['wd'] / 'native'; report_path = folder / 'metrics.json'
    if report_path.exists():
        report = validate_artifacts(read_json(report_path), ctx['wd'])
        if report.get('context') != ctx['context']:
            raise ValueError('Cached native measurements use different provenance')
        return report
    started = time.monotonic()
    datasets = {s: build_dataset(ctx['data'], s, root=root) for s in ('rr', 'fr')}
    engine = FRMetrics(datasets['fr']); results = {}
    primary, secondary = ctx['selected']
    results['EXACT_50000'] = _native_selection(ctx, primary, folder, datasets, device, engine)
    if secondary['checkpoint_sha256'] == primary['checkpoint_sha256']:
        results['RR_VAL_ERGAS_MIN'] = dict(results['EXACT_50000'], selection_id='RR_VAL_ERGAS_MIN',
                                            update=secondary['update'], alias_of='EXACT_50000')
    else:
        results['RR_VAL_ERGAS_MIN'] = _native_selection(ctx, secondary, folder / 'rr_val_selected', datasets, device, engine)
    report = seal(dict(schema='PANDA_RB01_NATIVE_v1', complete=True, context=ctx['context'],
                       selections=results, n_independent_students=1, n_unique_checkpoints=1 + int(secondary.get('alias_of') != 'EXACT_50000'),
                       file_hashes=_file_hashes(folder, ctx['wd']), elapsed_seconds=time.monotonic() - started,
                       completed_at_utc=utcnow(), device=str(device), historical_results_unchanged=True))
    atomic_json(report_path, report)
    metrics = results['EXACT_50000']
    print(f"[{run_id}] EXACT_50000 HQNR={metrics['fr']['hqnr']:.6f} SCC={metrics['rr']['scc']:.6f} ERGAS={metrics['rr']['ergas']:.6f}", flush=True)
    return validate_artifacts(report, ctx['wd'])


def _batch(dataset, scene, device):
    gt, _lms, ms, lp, pan, identity = dataset.base(scene)
    if int(identity[0]) != scene:
        raise ValueError('Stress sample no longer follows original H5 row order')
    return np.asarray(dataset.arrays['gt'][scene]), ms[None].to(device), lp[None].to(device), pan[None].to(device)


def _curve(rows, shifts):
    output = []
    for shift in shifts:
        subset = [x for x in rows if x['shift_id'] == shift['id']]
        if len(subset) != 20 or sorted(r['scene_index'] for r in subset) != list(range(20)):
            raise ValueError('Stress curve lost or duplicated a scene')
        item = dict(shift, n_scenes=20, n_failures=sum(r['status'] != 'ok' for r in subset))
        for key in (*STRESS_KEYS, 'relative_response_l1_sum', 'coverage_all_pan_paths'):
            values = [r[key] for r in subset if r.get(key) is not None]
            item[key] = float(np.mean(values)) if len(values) == 20 else None
            item[key + '_scene_std'] = float(np.std(values, ddof=1)) if len(values) == 20 else None
        output.append(item)
    base = output[0]
    for row in output:
        for key in STRESS_KEYS:
            row['delta_from_zero_' + key] = (None if row[key] is None or base[key] is None else row[key] - base[key])
    return output


@torch.no_grad()
def evaluate_stress(run_id, binding_path, mode, root=ROOT, device='cuda'):
    if mode not in MODES:
        raise ValueError('Unregistered stress mode')
    ctx = _context(run_id, binding_path, root); primary = ctx['selected'][0]
    if primary['update'] != 50000:
        raise ValueError('RB02 never evaluates a val-selected nonexact checkpoint')
    native = validate_artifacts(read_json(ctx['wd'] / 'native/metrics.json'), ctx['wd'])
    if native.get('context') != ctx['context']:
        raise ValueError('Stress must follow this exact source native evaluation')
    grid = shift_grid(root); shifts = validate_grid(grid)
    identity = dict(context=ctx['context'], source_selection='EXACT_50000', update=50000,
                     checkpoint_sha256=primary['checkpoint_sha256'], mode=mode, grid_sha256=object_sha(grid),
                     raw_subset=RAW_SUBSET)
    folder = ctx['wd'] / 'stress' / mode; path = folder / 'completion.json'
    if path.exists():
        report = validate_artifacts(read_json(path), ctx['wd'])
        if report.get('identity') != identity:
            raise ValueError('Cached stress curve has different provenance')
        return report
    started = time.monotonic(); dataset = build_dataset(ctx['data'], 'rr', root=root)
    if len(dataset) != 20:
        raise ValueError('RB02 requires all 20 original RR scenes')
    model = _model(ctx, primary, device); rows, inputs = [], []
    extra_raw_count = 0
    convention = warp_convention_test(device)
    try:
        with _precision(ctx['context']['source_identity'], device):
            for scene in range(20):
                truth, ms, lp, pan = _batch(dataset, scene, device)
                base = torch.nn.functional.interpolate(ms.float(), scale_factor=4, mode='bicubic', align_corners=False)
                native_correction = model.predict_delta(pan, base)
                for shift in shifts:
                    cache_path = folder / 'scenes' / f"scene{scene:02d}_{shift['id']}.json"
                    item_identity = dict(identity, scene_index=scene, shift=shift)
                    if cache_path.exists():
                        item = validate_artifacts(read_json(cache_path), ctx['wd'])
                        if item.get('identity') != item_identity:
                            raise ValueError('Stress scene cache identity changed')
                        rows.append(item['row']); inputs.append(item['inputs'])
                        extra_raw_count += int(item['row'].get('additional_failure_raw_saved', False))
                        continue
                    epsilon = pan.new_tensor([[shift['dy'], shift['dx']]])
                    out, estimated, shifted_pan, shifted_lp = stress_forward(model, pan, ms, lp, epsilon, mode)
                    prediction = ((out['y'].float().clamp(-1, 1) + 1.) * 1023.5)[0].cpu().numpy()
                    row = dict(scene_index=scene, shift_id=shift['id'], radius_hr=shift['radius_hr'],
                               epsilon_dy=shift['dy'], epsilon_dx=shift['dx'], status='ok',
                               prediction_digest=array_digest(prediction))
                    finite = bool(torch.isfinite(out['y']).all() and torch.isfinite(estimated).all()
                                  and torch.isfinite(native_correction).all())
                    if finite:
                        if mode == 'A_ON' and shift['id'] == 'D000':
                            native_path = ctx['wd'] / 'native/Ours_raw/WV3/reduced/pred' / f'testimg{scene}.npy'
                            if not native_path.is_file():
                                raise FileNotFoundError('Native exact evaluation must precede RB02')
                            native = np.load(native_path, allow_pickle=False)
                            error = float(np.abs(native - prediction).max())
                            if not np.array_equal(native, prediction):
                                raise ValueError(f'Zero-shift A_ON differs from native inference, max DN error {error}')
                            row['native_zero_max_abs_dn'] = error
                        row.update(stress_metrics(prediction, truth))
                        geometry = geometric_coverage(256, 256, epsilon, out['delta'])
                        row.update({f'coverage_{k}': float(v[0]) for k, v in geometry.items()})
                        row.update(estimated_dy=float(estimated[0, 0]), estimated_dx=float(estimated[0, 1]),
                                   applied_dy=float(out['delta'][0, 0]), applied_dx=float(out['delta'][0, 1]),
                                   native_correction_dy=float(native_correction[0, 0]), native_correction_dx=float(native_correction[0, 1]),
                                   relative_response_l1_sum=float(relative_response(estimated, epsilon, native_correction)[0]))
                        row['invalid_roi_sampling'] = row['coverage_all_pan_paths'] < 1.
                        if not all(np.isfinite(row[k]) for k in STRESS_KEYS):
                            row['status'] = 'nonfinite_metric'
                            for key in STRESS_KEYS:
                                if not np.isfinite(row[key]): row[key] = None
                    else:
                        row.update(status='nonfinite_prediction_or_correction', **{k: None for k in STRESS_KEYS},
                                   relative_response_l1_sum=None, coverage_all_pan_paths=None)
                    files = {}
                    predefined = scene in RAW_SUBSET['scene_indices'] and shift['id'] in RAW_SUBSET['shift_ids']
                    extra_failure = (not predefined and (row['status'] != 'ok' or row.get('invalid_roi_sampling'))
                                     and extra_raw_count < RAW_SUBSET['max_additional_failure_raw_per_curve'])
                    row['additional_failure_raw_saved'] = bool(extra_failure)
                    extra_raw_count += int(bool(extra_failure))
                    if predefined or extra_failure:
                        raw_path = folder / 'raw' / f"scene{scene:02d}_{shift['id']}.npy"
                        save_array(raw_path, prediction)
                        files[str(raw_path.relative_to(ctx['wd']))] = sha256(raw_path)
                    info = dict(scene_index=scene, shift_id=shift['id'],
                                native_pan_digest=array_digest(pan[0].cpu().numpy()),
                                shifted_pan_digest=array_digest(shifted_pan[0].cpu().numpy()),
                                shifted_lp_digest=array_digest(shifted_lp[0].cpu().numpy()),
                                ms_digest=array_digest(ms[0].cpu().numpy()),
                                native_input_reused=shift['id'] == 'D000', lp_regenerated=shift['id'] != 'D000')
                    item = seal(dict(complete=True, identity=item_identity, row=row, inputs=info, file_hashes=files))
                    atomic_json(cache_path, item); rows.append(row); inputs.append(info)
    finally:
        del model
    rows.sort(key=lambda x: (x['shift_id'], x['scene_index']))
    save_csv(folder / 'per_scene.csv', rows)
    atomic_json(folder / 'shift_inputs_manifest.json', dict(identity=identity, conventions=convention, inputs=inputs,
                 input_frame='native MS/GT unchanged; PAN once shifted then frozen LP recipe',
                 geometry='PAN and LP/HP frontend interpolation/filter supports; not U receptive field',
                 edge_error='signed Scharr/32, reflect at fixed192 ROI, mean absolute x/y difference in DN, all ROI centers',
                 stress_roi='32:-32, fixed192; never masked or recropped', outliers_clamped=False))
    summary = _curve(rows, shifts)
    atomic_json(folder / 'curve_summary.json', dict(mode=mode, points=summary,
                 statistical_unit=run_id, n_independent_students=1,
                 interpretation='paired inference correction dependence, not a trained no-align baseline'))
    report = seal(dict(schema='PANDA_RB02_CURVE_v1', complete=True, identity=identity, n_scenes=20,
                       n_shifts=49, n_observations=980, n_independent_students=1,
                       n_numerical_failures=sum(r['status'] != 'ok' for r in rows),
                       n_invalid_geometry=sum(bool(r.get('invalid_roi_sampling')) for r in rows),
                       curve=summary, file_hashes=_file_hashes(folder, ctx['wd']),
                       device=str(device), elapsed_seconds=time.monotonic() - started, completed_at_utc=utcnow()))
    atomic_json(path, report)
    return validate_artifacts(report, ctx['wd'])
