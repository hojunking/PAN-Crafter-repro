"""Independent M12 evidence: native selectors and four fixed-ROI stress modes.

All metrics reuse the frozen FH12/B01 numeric recipes.  M12 never edits B01
artifacts, chooses by HQNR, shrinks diagnostic support, or retrains on API debt.
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
                                        evaluator_identity as base_evaluator_identity, JQM_VARIANT, _precision)
from panda_rb_m12.stress import (MODES, ROIS, RAW_SUBSET, STRESS_KEYS, PROTOCOL_ID,
    array_digest, validate_grid, stress_forward, relative_response,
    geometric_coverage, stress_metrics, warp_convention_test, summarize_curve)
from panda_rb_m12.common import ROOT, atomic_json, object_sha, read_json, sha256, utcnow, source_identity


def evaluator_identity(root=ROOT):
    from tools.metrics.eval_fr import load_dlpan
    module = load_dlpan(os.environ.get('PANCRAFTER_DLPAN', str(Path(root).parent / 'DLPan-Toolbox')))
    path = Path(module.__file__).resolve()
    digest = sha256(path)
    result = module.interp23tap(np.arange(64, dtype=np.float64).reshape(8, 8, 1), 4)
    if result.shape != (32, 32, 1) or not np.isfinite(result).all() or sha256(path) != digest:
        raise ValueError('Frozen external FR interpolation validation failed')
    identity = dict(base_evaluator_identity(root))
    identity.update(external_wald_sha256=digest, external_wald_smoke='finite8to32_ratio4_PASS')
    files = dict(identity['files'])
    for name in ('evaluation.py', 'stress.py', 'reference_probe.py'):
        local = Path(root) / 'panda_rb_m12' / name
        if local.is_file():
            files[str(local.relative_to(root))] = sha256(local)
    identity['files'] = files
    identity['content_sha256'] = object_sha(dict(files=files, external_wald_sha256=digest))
    return identity


def seal(value):
    return dict(value, payload_sha256=object_sha(value))


def validate_artifacts(report, directory):
    body = {k: v for k, v in report.items() if k != 'payload_sha256'}
    if report.get('payload_sha256') != object_sha(body) or not report.get('complete'):
        raise ValueError('Incomplete or modified M12 evaluation report')
    directory = Path(directory).resolve()
    for name, digest in report.get('file_hashes', {}).items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not (directory / path).resolve().is_relative_to(directory):
            raise ValueError('Unsafe M12 relative evaluation artifact path')
        if sha256(directory / path) != digest:
            raise ValueError('M12 evaluation artifact changed: ' + name)
    return report


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Refusing to replace different M12 evidence: ' + str(path))
        return
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError('Concurrent different M12 evidence: ' + str(path))
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
        raise ValueError('Empty M12 evaluation scene table')
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
    writer.writeheader(); writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def selected_checkpoints(manifest):
    from panda_rb_m12.plan import VAL_GRID
    if not manifest.get('complete') or manifest.get('actual_updates') != 50000:
        raise ValueError('M12 native evaluation requires completed 50000 updates')
    records = manifest.get('validation_records', [])
    if [r['update'] for r in records] != list(VAL_GRID) or not all(np.isfinite(r['val_ergas']) for r in records):
        raise ValueError('Incomplete, nonfinite or altered M12 validation grid')
    winner = min(records, key=lambda r: (r['val_ergas'], r['update']))
    primary, secondary = manifest['primary'], manifest['secondary']
    for item, name, step, folder in ((primary, 'EXACT_50000', 50000, 'exact50000'),
                                     (secondary, 'RR_VAL_ERGAS_MIN', winner['update'], 'val_selected')):
        if (item.get('selection_id') != name or item.get('update') != step
                or item.get('directory') != folder or not item.get('checkpoint_sha256')):
            raise ValueError('M12 selector differs from exact/validation ERGAS earliest-tie rule')
    same = primary['checkpoint_sha256'] == secondary['checkpoint_sha256']
    if bool(secondary.get('alias_of') == 'EXACT_50000') != same:
        raise ValueError('Same model bytes must be an explicit selection alias')
    return primary, secondary


def _context(run_id, binding_path, root=ROOT):
    from panda_rb_m12.binding import load_binding
    from panda_rb_m12.common import campaign_dir
    from panda_rb_m12.plan import case_for, run_dir
    row = case_for(run_id, root); wd = run_dir(run_id, root)
    cfg = yaml.safe_load((wd / 'meta/config.resolved.yaml').read_text())
    meta = cfg.get('panda_rb_m12', {})
    if (meta.get('run_id') != run_id or cfg['seed'] != row['seed'] or meta.get('case_id') != row['case_id']):
        raise ValueError('M12 resolved config belongs to another run')
    teacher, _teacher_cfg, binding, data, _q = load_binding(binding_path, device='cpu')
    del teacher
    manifest_path = wd / 'checkpoints/selection_manifest.json'
    manifest = read_json(manifest_path)
    selected = selected_checkpoints(manifest)
    initial = read_json(wd / 'init_manifest.json')
    stream = read_json(wd / 'stream_manifest.json')
    consumed = read_json(wd / 'diagnostics/consumed_weights.json')
    if (manifest.get('initialization_sha256') != object_sha(initial)
            or not manifest.get('consumed_stream_sha256')
            or manifest['consumed_stream_sha256'] != consumed.get('consumed_stream_sha256')
            or manifest['consumed_stream_sha256'] != stream.get('full_stream_sha256')
            or stream.get('updates') != 50000 or stream.get('seed') != row['seed']):
        raise ValueError('Initial U/A or actual committed 50K sample/view stream provenance differs')
    source = source_identity(root); identities = {}; expected_binding = object_sha(binding)
    for item in selected:
        path = wd / 'checkpoints' / item['directory']
        if not path.resolve().is_relative_to((wd / 'checkpoints').resolve()):
            raise ValueError('Selected checkpoint escapes this M12 run')
        identity = read_json(path / 'identity.json')
        if (identity.get('model_sha256') != sha256(path / 'model.safetensors')
                or identity['model_sha256'] != item['checkpoint_sha256']
                or identity.get('update') != item['update'] or identity.get('config_sha256') != object_sha(cfg)
                or identity.get('source_identity') != source or identity.get('binding_sha256') != expected_binding):
            raise ValueError('M12 selected bytes/config/source/update/frozen F1 mismatch')
        identities[item['directory']] = identity
    if any(data['splits'][s]['count'] != 20 for s in ('rr', 'fr')):
        raise ValueError('Native M12 requires all 20 RR and FR scenes')
    weight_manifest = read_json(campaign_dir(root) / 'common/weights' / f"seed_{row['seed']}.json")
    if object_sha(weight_manifest) != identities[selected[0]['directory']].get('weights_sha256'):
        raise ValueError('Raw q/e and geometry map identity differ from the trained checkpoint')
    context = dict(run_id=run_id, case=row, config_sha256=object_sha(cfg),
                   binding_sha256=expected_binding, data_sha256=object_sha(data),
                   binding_common_sha256=binding.get('common_sha256'),
                   teacher_sha256=binding['teacher_checkpoint_sha256'],
                   q_cache_sha256=binding['origin_reference']['q_cache_sha256'],
                   calibration_identity={key: binding[key] for key in ('q_ref', 'tau_R', 'teacher_config_sha256', 'common_sha256')},
                   weight_map_manifest_sha256=object_sha(weight_manifest),
                   raw_map_identity_sha256=object_sha({key: weight_manifest['arrays_sha256'][key] for key in ('q', 'e_bar')}),
                   initialization_sha256=manifest['initialization_sha256'],
                   initial_tensor_hashes=initial['hashes'], consumed_stream_sha256=manifest['consumed_stream_sha256'],
                   data_content_identity={s: {k: v for k, v in info.items() if k in
                       ('sha256', 'lpan_sha256', 'sample_order_sha256', 'count', 'shapes')}
                       for s, info in data['splits'].items()},
                   selection_manifest_sha256=sha256(manifest_path), source_identity=source,
                   evaluator_identity=evaluator_identity(root))
    return dict(wd=wd, cfg=cfg, data=data, selected=selected, identities=identities, context=context)


def _model(ctx, selection, device):
    from safetensors.torch import load_file
    state = load_file(str(ctx['wd'] / 'checkpoints' / selection['directory'] / 'model.safetensors'))
    aligner = {k[8:]: v for k, v in state.items() if k.startswith('aligner.')}
    cfg = ctx['cfg']; ma = cfg['model_args']
    model, _ = build_model('PLH', ma['hidden_size'], ma['depth'], cfg['seed'],
                           role='S', teacher_aligner_state=aligner)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def _raw_checks(pred, split):
    expected = (20, 8, 256, 256) if split == 'rr' else (20, 8, 512, 512)
    if pred.shape != expected or pred.dtype != np.float32:
        raise ValueError(f'Invalid {split} raw shape/dtype: {pred.shape}/{pred.dtype}')
    if not np.isfinite(pred).all() or pred.min() < 0 or pred.max() > 2047:
        raise ValueError('Raw prediction is not finite DN0..2047')


def _scene_record(normal, extra, split, scene, correction, digest):
    # Metrics may already include scene_index: update after merge, never dict(**x, scene_index=x).
    row = {k: v for k, v in {**normal, **extra}.items() if isinstance(v, (float, int))}
    row.update(split=split, scene_index=scene, source_h5_row=scene,
               correction_dy=float(correction[0]), correction_dx=float(correction[1]),
               prediction_digest=digest)
    return row


def _native_selection(ctx, selection, directory, datasets, device, engine):
    model = _model(ctx, selection, device); per_scene, raw = [], {}
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
            for scene in range(20):
                path = directory / 'Ours_raw/WV3' / folder / 'pred' / f'testimg{scene}.npy'
                metadata = save_array(path, pred[scene])
                metadata.update(scene_index=scene, h5_row_index=scene, path=str(path.relative_to(ctx['wd'])))
                raw[split].append(metadata)
                per_scene.append(_scene_record(normal['per_scene'][scene], extra['per_scene'][scene],
                                                split, scene, correction[scene], metadata['prediction_digest']))
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
    ctx = _context(run_id, binding_path, root); folder = ctx['wd'] / 'native'; path = folder / 'metrics.json'
    if path.exists():
        report = validate_artifacts(read_json(path), ctx['wd'])
        if report.get('context') != ctx['context']:
            raise ValueError('Cached M12 native provenance changed')
        return report
    started = time.monotonic()
    datasets = {s: build_dataset(ctx['data'], s, root=root) for s in ('rr', 'fr')}
    engine = FRMetrics(datasets['fr']); primary, secondary = ctx['selected']
    results = {'EXACT_50000': _native_selection(ctx, primary, folder, datasets, device, engine)}
    if secondary['checkpoint_sha256'] == primary['checkpoint_sha256']:
        results['RR_VAL_ERGAS_MIN'] = dict(results['EXACT_50000'], selection_id='RR_VAL_ERGAS_MIN',
                                          update=secondary['update'], alias_of='EXACT_50000')
    else:
        results['RR_VAL_ERGAS_MIN'] = _native_selection(ctx, secondary, folder / 'rr_val_selected', datasets, device, engine)
    report = seal(dict(schema='PANDA_M12_NATIVE_v1', complete=True, context=ctx['context'],
                       selections=results, n_independent_students=1,
                       n_unique_checkpoints=len({p['checkpoint_sha256'] for p in ctx['selected']}),
                       file_hashes=_file_hashes(folder, ctx['wd']), elapsed_seconds=time.monotonic()-started,
                       completed_at_utc=utcnow(), device=str(device), historical_results_unchanged=True))
    atomic_json(path, report)
    metrics = results['EXACT_50000']
    print(f"[{run_id}] EXACT_50000 HQNR={metrics['fr']['hqnr']:.6f} SCC={metrics['rr']['scc']:.6f} ERGAS={metrics['rr']['ergas']:.6f}", flush=True)
    return validate_artifacts(report, ctx['wd'])


def _batch(dataset, scene, device):
    _gt, _lms, ms, lp, pan, identity = dataset.base(scene)
    if int(identity[0]) != scene:
        raise ValueError('Stress sample no longer follows original H5 row order')
    return np.asarray(dataset.arrays['gt'][scene]), ms[None].to(device), lp[None].to(device), pan[None].to(device)


@torch.no_grad()
def evaluate_stress(run_id, binding_path, mode, root=ROOT, device='cuda'):
    from panda_rb_m12.plan import shift_grid
    if mode not in MODES:
        raise ValueError('Unregistered M12 stress mode')
    ctx = _context(run_id, binding_path, root); primary = ctx['selected'][0]
    if ctx['context']['case']['case_id'] not in ('QFULL', 'QMEAN', 'QSHUF', 'QESUR', 'QEDGE', 'QALIGN'):
        raise ValueError('M12 stress is restricted to STEP1 models')
    native = validate_artifacts(read_json(ctx['wd'] / 'native/metrics.json'), ctx['wd'])
    if native.get('context') != ctx['context']:
        raise ValueError('Stress must follow the same exact source native evaluation')
    grid = shift_grid(root); shifts = validate_grid(grid)
    identity = dict(context=ctx['context'], source_selection='EXACT_50000', update=50000,
                    checkpoint_sha256=primary['checkpoint_sha256'], mode=mode, grid_sha256=object_sha(grid),
                    raw_subset=RAW_SUBSET, protocol_id=PROTOCOL_ID, roi_primary='fixed160', roi_aux='fixed192')
    folder = ctx['wd'] / 'stress' / mode; path = folder / 'completion.json'
    if path.exists():
        report = validate_artifacts(read_json(path), ctx['wd'])
        if report.get('identity') != identity:
            raise ValueError('Cached M12 stress provenance changed')
        return report
    started = time.monotonic(); dataset = build_dataset(ctx['data'], 'rr', root=root)
    if len(dataset) != 20:
        raise ValueError('M12 stress requires every original RR20 scene')
    model = _model(ctx, primary, device); rows = {roi: [] for roi in ROIS}; inputs = []
    extra_raw_count = 0; convention = warp_convention_test(device)
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
                            raise ValueError('M12 stress point provenance changed')
                        for roi in ROIS: rows[roi].append(item['rows'][roi])
                        inputs.append(item['inputs']); extra_raw_count += int(item['additional_failure_raw_saved'])
                        continue
                    epsilon = pan.new_tensor([[shift['dy'], shift['dx']]])
                    out, estimated, shifted_pan, shifted_lp = stress_forward(model, pan, ms, lp, epsilon, mode, native_correction)
                    prediction = ((out['y'].float().clamp(-1, 1)+1.)*1023.5)[0].cpu().numpy()
                    common = dict(scene_index=scene, shift_id=shift['id'], radius_hr=shift['radius_hr'],
                                  epsilon_dy=shift['dy'], epsilon_dx=shift['dx'], prediction_digest=array_digest(prediction))
                    corrections = dict(estimated_dy=float(estimated[0, 0]), estimated_dx=float(estimated[0, 1]),
                                       applied_dy=float(out['delta'][0, 0]), applied_dx=float(out['delta'][0, 1]),
                                       native_correction_dy=float(native_correction[0, 0]), native_correction_dx=float(native_correction[0, 1]))
                    common.update({key: value if np.isfinite(value) else None for key, value in corrections.items()})
                    common['nonfinite_correction_fields'] = [key for key, value in corrections.items() if not np.isfinite(value)]
                    finite = bool(torch.isfinite(out['y']).all() and torch.isfinite(estimated).all()
                                  and torch.isfinite(native_correction).all() and torch.isfinite(out['delta']).all())
                    if finite:
                        common.update(relative_response_l1_sum=float(relative_response(estimated, epsilon, native_correction)[0]))
                        if mode != 'A_ZERO_INFERENCE_ONLY' and shift['id'] == 'D000':
                            native_path = ctx['wd'] / 'native/Ours_raw/WV3/reduced/pred' / f'testimg{scene}.npy'
                            native_pred = np.load(native_path, allow_pickle=False)
                            error = float(np.abs(native_pred-prediction).max())
                            if not np.array_equal(native_pred, prediction):
                                raise ValueError(f'M12 D000 {mode} differs from exact native output: {error} DN')
                            common['native_zero_max_abs_dn'] = error
                    else:
                        common['relative_response_l1_sum'] = None
                    roi_rows = {}
                    for roi, margin in ROIS.items():
                        row = dict(common, roi=roi, protocol_id=PROTOCOL_ID, status='ok')
                        if finite:
                            row.update(stress_metrics(prediction, truth, roi))
                            geometry = geometric_coverage(256, 256, epsilon, out['delta'], margin=margin)
                            row.update({f'coverage_{k}': float(v[0]) for k, v in geometry.items()})
                            row['invalid_roi_sampling'] = row['coverage_all_pan_paths'] < 1.
                            if not all(np.isfinite(row[k]) for k in STRESS_KEYS):
                                row['status'] = 'nonfinite_metric'
                                for key in STRESS_KEYS:
                                    if not np.isfinite(row[key]): row[key] = None
                            elif row['invalid_roi_sampling']:
                                row['status'] = 'coverage_failure'
                        else:
                            row.update(status='nonfinite_prediction_or_correction', **{k: None for k in STRESS_KEYS},
                                       coverage_all_pan_paths=None, invalid_roi_sampling=True)
                        roi_rows[roi] = row
                    predefined = scene in RAW_SUBSET['scene_indices'] and shift['id'] in RAW_SUBSET['shift_ids']
                    failure = any(row['status'] != 'ok' for row in roi_rows.values())
                    extra_failure = bool(not predefined and failure and extra_raw_count < RAW_SUBSET['max_additional_failure_raw_per_curve'])
                    extra_raw_count += int(extra_failure); files = {}
                    if predefined or extra_failure:
                        raw_path = folder / 'raw' / f"scene{scene:02d}_{shift['id']}.npy"
                        save_array(raw_path, prediction); files[str(raw_path.relative_to(ctx['wd']))] = sha256(raw_path)
                        if not finite:
                            # Clipping can hide +/-inf. Preserve the actual failing normalized
                            # tensor alongside the usual DN artifact within the same bounded subset.
                            failure_path = folder / 'raw_nonfinite_normalized' / raw_path.name
                            save_array(failure_path, out['y'][0].float().cpu().numpy())
                            files[str(failure_path.relative_to(ctx['wd']))] = sha256(failure_path)
                    info = dict(scene_index=scene, shift_id=shift['id'], native_pan_digest=array_digest(pan[0].cpu().numpy()),
                                shifted_pan_digest=array_digest(shifted_pan[0].cpu().numpy()), shifted_lp_digest=array_digest(shifted_lp[0].cpu().numpy()),
                                ms_digest=array_digest(ms[0].cpu().numpy()), native_input_reused=shift['id']=='D000', lp_regenerated=shift['id']!='D000')
                    item = seal(dict(complete=True, identity=item_identity, rows=roi_rows, inputs=info, file_hashes=files,
                                     additional_failure_raw_saved=extra_failure))
                    atomic_json(cache_path, item)
                    for roi in ROIS: rows[roi].append(roi_rows[roi])
                    inputs.append(info)
    finally:
        del model
    for roi in ROIS: rows[roi].sort(key=lambda row: (row['shift_id'], row['scene_index']))
    save_csv(folder / 'per_scene.csv', rows['fixed160'])
    save_csv(folder / 'per_scene_fixed192.csv', rows['fixed192'])
    summaries = {roi: summarize_curve(rows[roi], shifts, roi) for roi in ROIS}
    atomic_json(folder / 'shift_inputs_manifest.json', dict(identity=identity, conventions=convention, inputs=inputs,
        input_frame='native MS/GT unchanged; PAN shifted once then frozen LP regeneration',
        geometry='PAN and LP/HP sampling/filter support; not U receptive field',
        edge_error='signed Scharr/32 reflect at fixed ROI; mean absolute x/y difference DN',
        stress_rois=ROIS, support_mask_applied_to_metrics=False, outliers_clamped=False))
    atomic_json(folder / 'curve_summary.json', dict(summaries=summaries, statistical_unit=run_id,
        n_independent_students=1, interpretation='same Student paired inference; not retrained no-align; inverse is not absolute oracle'))
    primary_rows = rows['fixed160']
    failures = sum(row['status'] != 'ok' for row in primary_rows)
    report = seal(dict(schema='PANDA_M12_CURVE_v1', complete=True, identity=identity,
        n_scenes=20, n_shifts=49, n_observations=980, n_independent_students=1,
        status='PROCESSED_WITH_FLAGS' if any(r['status']!='ok' for roi in ROIS for r in rows[roi]) else 'CLEAN',
        n_failures=failures, n_valid=980-failures, n_expected=980,
        n_numerical_failures=sum(row['status'].startswith('nonfinite') for row in primary_rows),
        n_invalid_geometry=sum(bool(row['invalid_roi_sampling']) for row in primary_rows),
        n_auxiliary_failures=sum(row['status']!='ok' for row in rows['fixed192']),
        curve=summaries['fixed160']['points'], radii=summaries['fixed160']['radii'],
        auxiliary_curve=summaries['fixed192']['points'], auxiliary_radii=summaries['fixed192']['radii'],
        file_hashes=_file_hashes(folder, ctx['wd']), device=str(device),
        elapsed_seconds=time.monotonic()-started, completed_at_utc=utcnow()))
    atomic_json(path, report)
    return validate_artifacts(report, ctx['wd'])
