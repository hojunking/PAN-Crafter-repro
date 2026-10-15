"""Build a verified RR-only pred+GT bundle from the fixed s1 raw exports.

No model forward, GPU access, rendering, training, checkpoint selection, or upload.
Re-evaluate the delivered float32 arrays using the frozen original RR evaluator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
from zoneinfo import ZoneInfo


TARGETS = {
    'WV3': ('FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1',
            38380, '3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50', 8),
    'QB': ('QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1',
           47470, '548d0ed2ccdc968d319e850e1470d7068f9321a9f0cc18e1794807f5cfd3399d', 4),
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def export_sensor(args, package, sensor):
    import h5py
    import numpy as np
    import yaml

    run, step, checkpoint_sha, channels = TARGETS[sensor]
    source = args.source / f'Ours_s1_{sensor}_raw_20260924_031541KST'
    checksums = {}
    for line in (source / 'checksums.sha256').read_text().splitlines():
        value, relative = line.split('  ', 1)
        require(relative not in checksums, 'Duplicate source checksum path')
        checksums[relative] = value

    def verified(relative):
        path = source / relative
        require(path.resolve().is_relative_to(source.resolve()), 'Source path escapes package')
        require(checksums.get(relative) == sha(path), f'Prior export checksum mismatch: {relative}')
        return path

    prior = read(verified('manifest.json'))
    prior_scenes = read(verified('scene_manifest.json'))
    data = read(verified('provenance/dataset_manifest.json'))
    identity = read(verified('provenance/checkpoint_identity.json'))
    selection = read(verified('metrics/original_selection.json'))
    original = read(verified('metrics/original_candidate.json'))
    config_path = verified('provenance/config.resolved.yaml')
    config = yaml.safe_load(config_path.read_text())
    origin_source = read(verified('provenance/source_identity.json'))
    inference_runtime = read(verified('provenance/actual_source_identity.json'))
    live = args.root / 'work_dir' / run
    candidate = live / 'candidates' / str(step)
    require(sha(candidate / 'model.safetensors') == checkpoint_sha, 'Live checkpoint SHA mismatch')
    require(identity == read(candidate / 'identity.json'), 'Live checkpoint identity changed')
    require(selection == read(live / 'official/target_selection.json'), 'Selected TARGET record changed')
    require(original == read(live / 'official' / f'candidate_{step}.json'), 'Original candidate record changed')
    require(selection['checkpoint_identity'] == identity == original['checkpoint_identity'], 'Selection/candidate identity mismatch')
    require(selection['run_id'] == run and selection['step'] == step and selection['selection_id'] == 'TARGET', 'Wrong selection')
    require(selection['checkpoint_sha256'] == identity['model_sha256'] == checkpoint_sha, 'Wrong checkpoint binding')
    require(identity['update'] == original['update'] == step and selection['normal_same_step_A_U'], 'Wrong A/U binding')
    require(prior['run_id'] == run and prior['step'] == step and prior['checkpoint_sha256'] == checkpoint_sha, 'Wrong reused predictions')
    require(prior['axis'] == 'CHW' and prior['unit'] == 'DN' and prior['dtype'] == 'float32' and prior['max_dn'] == 2047, 'Unknown prediction scale/axis')
    require(prior['num_bands'] == channels and prior['selection'] == 'TARGET', 'Prediction band/selection mismatch')
    policy = prior['inference']
    require('A_ON' in policy['mode'] and not any(policy[k] for k in ('gt_in_forward', 'shuffle', 'augment', 'mask', 'crop_or_resize_raw_or_standalone_png')), 'Incompatible prior forward policy')
    require(policy['output'] == '(clamp(y,-1,1)+1)*1023.5; no DN rounding', 'Unknown DN conversion')
    require(object_sha(config) == identity['config_sha256'] == prior['config_sha256'], 'Config identity mismatch')
    require(object_sha(data) == identity['data_sha256'] == prior['dataset_manifest_sha256'], 'Data manifest identity mismatch')
    require(sha(live / 'meta/config.resolved.yaml') == sha(config_path), 'Live config differs')
    require(origin_source == identity['source_identity'] == prior['original_runtime'], 'Source identity mismatch')
    require(inference_runtime == prior['actual_runtime'], 'Prior inference runtime mismatch')

    binding = data['splits']['rr']
    h5_path, lp_path = Path(binding['dataroot']), Path(binding['lpan_path'])
    require(sha(h5_path) == binding['sha256'], 'Native RR H5 SHA mismatch')
    require(sha(lp_path) == binding['lpan_sha256'], 'Original RR LP SHA mismatch')
    prior_rows = [row for row in prior_scenes if row['sensor'] == sensor and row['split'] == 'reduced']
    require(len(prior_rows) == 20 and sorted(row['scene_index_zero_based'] for row in prior_rows) == list(range(20)), 'Invalid original RR scene mapping')
    prior_rows = sorted(prior_rows, key=lambda row: row['scene_index_zero_based'])
    band_order = (dict(source_indices_zero_based=list(range(channels)), labels=data['band_order'], status='SOURCE_MANIFEST_VERIFIED', permutation_applied=False)
                  if 'band_order' in data else
                  dict(source_indices_zero_based=list(range(channels)), labels=None, status='SOURCE_BAND_NAMES_UNCONFIRMED_INDICES_PRESERVED', permutation_applied=False))
    provenance = package / 'provenance' / sensor
    provenance.mkdir(parents=True)
    for relative, name in [('provenance/checkpoint_identity.json', 'checkpoint_identity.json'),
                           ('provenance/config.resolved.yaml', 'config.resolved.yaml'),
                           ('provenance/dataset_manifest.json', 'dataset_manifest.json'),
                           ('provenance/source_identity.json', 'source_identity.json'),
                           ('provenance/actual_source_identity.json', 'prediction_runtime.json'),
                           ('metrics/original_selection.json', 'original_selection.json'),
                           ('metrics/original_candidate.json', 'original_candidate.json'),
                           ('manifest.json', 'source_export_manifest.json'),
                           ('scene_manifest.json', 'source_export_scene_manifest.json')]:
        shutil.copyfile(verified(relative), provenance / name)
    shutil.copyfile(source / 'checksums.sha256', provenance / 'source_export_checksums.sha256')

    rows, checks = [], []
    with h5py.File(h5_path, 'r') as h5:
        require('gt' in h5, f'{sensor}: BLOCKED_MISSING_GT')
        require(h5['gt'].shape == (20, channels, 256, 256), 'Native GT shape/order unknown')
        for index, old in enumerate(prior_rows):
            require(old['h5_index'] == index and old['source_h5'] == str(h5_path)
                    and old['source_h5_sha256'] == binding['sha256'] and old['lpan_h5_sha256'] == binding['lpan_sha256'], 'Prediction/GT scene pairing mismatch')
            expected_pred = f'Ours_raw/{sensor}/reduced/pred/testimg{index}.npy'
            expected_gt = f'references/{sensor}/reduced/gt/testimg{index}.npy'
            require(old['prediction']['path'] == expected_pred and old['references']['gt']['path'] == expected_gt, 'Prior path/index mismatch')
            pred_source = verified(expected_pred)
            old_gt_path = verified(expected_gt)
            require(sha(pred_source) == old['prediction']['sha256'] and sha(old_gt_path) == old['references']['gt']['sha256'], 'Scene record SHA mismatch')
            pred = np.load(pred_source, allow_pickle=False)
            gt_source = np.asarray(h5['gt'][index])
            require(np.array_equal(np.load(old_gt_path, allow_pickle=False), gt_source), 'Prior GT not identical to native RR GT')
            require(pred.dtype == np.float32 and pred.shape == gt_source.shape == (channels, 256, 256), 'Prediction shape/dtype mismatch')
            require(np.isfinite(pred).all() and np.isfinite(gt_source).all() and pred.min() >= 0 and pred.max() <= 2047, 'Invalid/nonfinite raw values')
            gt = gt_source.astype(np.float32)  # Required dtype only; no clipping/normalization/warp.
            require(np.isfinite(gt).all(), 'GT float32 conversion overflow')
            pred_path, gt_path = package / expected_pred, package / expected_gt
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pred_source, pred_path)  # Exact prediction bytes, no new inference.
            with gt_path.open('xb') as stream:
                np.save(stream, gt, allow_pickle=False)
            require(sha(pred_path) == sha(pred_source), 'Prediction copy is not byte-identical')
            records = {}
            for label, path, array in [('pred', pred_path, pred), ('gt', gt_path, gt)]:
                saved = np.load(path, allow_pickle=False)
                require(saved.dtype == np.float32 and saved.shape == (channels, 256, 256)
                        and np.isfinite(saved).all() and np.array_equal(saved, array), 'Saved raw roundtrip failed')
                records[label] = dict(path=str(path.relative_to(package)), sha256=sha(path), bytes=path.stat().st_size,
                                      shape=list(saved.shape), dtype='float32', axis_order='CHW', units='DN',
                                      minimum=float(saved.min()), maximum=float(saved.max()))
            cast_error = float(np.max(np.abs(gt.astype(np.float64) - gt_source.astype(np.float64))))
            rows.append(dict(sensor=sensor, export_index=index, **records,
                             source=dict(file=str(h5_path), sha256=binding['sha256'], gt_key='gt', original_index=index,
                                         pred_source_file=str(pred_source), source_gt_dtype=str(gt_source.dtype)),
                             band_order=band_order, pairing='SOURCE_H5_SHA_AND_ORIGINAL_INDEX_VERIFIED',
                             cross_method_png_validation='NOT_CHECKED_BY_SCOPE'))
            checks.append(dict(sensor=sensor, index=index, pred_byte_identical=True, gt_source_cast_exact=True,
                               gt_cast_max_abs_error_dn=cast_error, gt_cast_lossless=cast_error == 0,
                               shape_dtype_finite_roundtrip_passed=True, scene_pairing_passed=True))
    write(provenance / 'rr_reuse_binding.json', dict(source_package=str(source),
          source_manifest_sha256=sha(source / 'manifest.json'), source_scene_manifest_sha256=sha(source / 'scene_manifest.json'),
          source_checksums_sha256=sha(source / 'checksums.sha256'), rr_scenes_verified=20))

    # Evaluator availability does not block already verified raw delivery.
    metric_report = dict(metric_parity='NOT_RECOMPUTED', original=original['rr'], saved_arrays=None)
    try:
        if sensor == 'WV3':
            from fh20r1 import common
            from fh12 import evaluation
        else:
            from qg40 import common, evaluation
        require(Path(evaluation.__file__).resolve().is_relative_to(args.runtime), 'Wrong evaluator checkout')
        current_runtime = common.source_identity(args.runtime)
        numerical_keys = ('files', 'content_sha256', 'torch', 'numpy', 'scipy', 'skimage', 'cuda', 'cudnn',
                          'tf32_matmul', 'tf32_cudnn', 'numeric_method_revision')
        require(all(origin_source.get(k) == current_runtime.get(k) for k in numerical_keys), 'RR evaluator code/runtime mismatch')
        p = np.stack([np.load(package / row['pred']['path'], allow_pickle=False) for row in rows])
        g = np.stack([np.load(package / row['gt']['path'], allow_pickle=False) for row in rows])
        if sensor == 'WV3':
            reevaluated = evaluation.rr_metrics(p, g)
        else:
            from dataclasses import replace
            from qg40.plan import sensor_spec
            spec = replace(sensor_spec(sensor), band_order=tuple(data['band_order']))
            reevaluated = evaluation.rr_metrics(p, g, spec)
        keys = ['ergas', 'sam', 'psnr', 'ssim', 'scc', 'q8' if channels == 8 else 'q4']
        comparison = {key: dict(original=original['rr'][key], saved_arrays=reevaluated[key],
                               difference=reevaluated[key] - original['rr'][key]) for key in keys}
        metric_report.update(metric_parity='PARITY_REVIEW_REQUIRED', saved_arrays=reevaluated, comparison=comparison,
                             evaluator_runtime=current_runtime, evaluator_path=str(evaluation.__file__),
                             reason='Reused CPU predictions vs original CUDA results; no predeclared numerical tolerance. Raw validity is separate from parity.')
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        metric_report['reason'] = f'{type(error).__name__}: {error}'
    metric_report.update(prediction_device=prior['actual_device'], metric_device='cpu', crop='20:-21', q_block=32,
                         saved_array_shape=[20, channels, 256, 256], compared_from_saved_float32_pred_and_gt=True
                         if metric_report['saved_arrays'] is not None else False)
    print(f'{sensor}: reused pred20 + native GT20 verified; metrics {metric_report["metric_parity"]}', flush=True)
    record = dict(sensor=sensor, origin_server='s1', run_id=run, selection='TARGET', selected_step=step,
                  checkpoint_sha256=checkpoint_sha, checkpoint_identity=identity, source_identity=origin_source,
                  resolved_config_sha256=identity['config_sha256'], raw_dataset_sha256=binding['sha256'],
                  LP_input_identity=dict(path=str(lp_path), sha256=binding['lpan_sha256'], recipe=data['recipe']),
                  reuse_or_inference='reused', source_prediction_path_or_new_export=str(source / 'Ours_raw' / sensor / 'reduced/pred'),
                  scene_count=20, axis_order='CHW', dtype='float32', units='DN', max_dn=2047,
                  band_order=band_order, spatial_shape=[256, 256], num_bands=channels,
                  normalization_and_clamp_policy=policy, gt_conversion='native GT.astype(float32); no clipping or other transformation',
                  execution_device='cpu: file validation/metric recomputation only; no new inference',
                  prediction_execution_device=prior['actual_device'], runtime_version=inference_runtime,
                  prediction_generated_at=prior['started_utc'], exported_at=utcnow(), raw_export_status='COMPLETE',
                  rendering_status='SKIPPED_BY_USER', cross_method_png_validation='NOT_CHECKED_BY_SCOPE',
                  delivery_status='READY_LOCAL_NOT_DELIVERED', metric_parity=metric_report['metric_parity'])
    return record, rows, checks, metric_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'runtime', 'source', 'delivery'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('root', 'runtime', 'source', 'delivery'):
        setattr(args, name, getattr(args, name).resolve())
    require(args.delivery.is_relative_to(args.root / 'results_log'), 'Destination must be in results_log')
    sys.path.insert(0, str(args.runtime))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    require(not torch.cuda.is_available(), 'GPU must not be exposed to this packaging job')
    timestamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%SKST')
    package = args.delivery / f'Ours_s1_WV3_QB_RR_raw_gt_{timestamp}'
    package.mkdir(parents=True, exist_ok=False)
    sensors, scenes, checks, metrics = {}, [], [], {}
    for sensor in TARGETS:
        record, rows, tests, metric_report = export_sensor(args, package, sensor)
        sensors[sensor] = record
        scenes.extend(rows)
        checks.extend(tests)
        metrics[sensor] = metric_report
    actual_npys = {str(p.relative_to(package)) for p in package.rglob('*.npy')}
    expected_npys = {row[k]['path'] for row in scenes for k in ('pred', 'gt')}
    require(actual_npys == expected_npys and len(actual_npys) == 80, 'Mandatory NPY inventory mismatch')
    require(not list(package.rglob('*.png')), 'Rendering is out of scope')
    write(package / 'scene_manifest.json', scenes)
    write(package / 'validation/raw_array_checks.json', dict(status='PASS', required_npy_count=80, actual_npy_count=80,
          prediction_files=40, gt_files=40, image_rendering_performed=False, per_scene=checks))
    write(package / 'validation/rr_metric_comparison.json', metrics)
    write(package / 'manifest.json', dict(task_id='PANDA_S1_WV3_QB_RR_RAW_GT_v2', exported_at=utcnow(),
          raw_export_status='COMPLETE', rendering_status='SKIPPED_BY_USER', cross_method_png_validation='NOT_CHECKED_BY_SCOPE',
          delivery_status='READY_LOCAL_NOT_DELIVERED', sensors=sensors, npy_count=80, training_changed=False,
          new_inference_performed=False, fr_included=False, external_upload_performed=False))
    shutil.copyfile(Path(__file__), package / 'provenance/ours_rr_raw_gt_export.py')
    with (package / 'README.md').open('x') as stream:
        stream.write('# WV3 + QB RR raw prediction / GT\n\n'
                     'Raw export COMPLETE. Local ZIP ready; not externally delivered.\n\n'
                     '- WV3 TARGET38380, QB TARGET47470: verified reused predictions, no new inference.\n'
                     '- Ours_raw/<sensor>/reduced/pred/testimg0..19.npy: 40 predictions.\n'
                     '- references/<sensor>/reduced/gt/testimg0..19.npy: 40 matched native GT arrays.\n'
                     '- All arrays float32 CHW DN, maxDN2047; full256x256, all8/4bands.\n'
                     '- GT comes from original RR H5, converted only to float32; no clipping/registration.\n'
                     '- Scene mapping uses original H5 SHA + gt key + index; no cross-method PNG matching.\n'
                     '- No PNG, rendering, calibration, FR inference, training, selection, or upload.\n'
                     '- Original CPU predictions differ from historical CUDA metrics; see validation/rr_metric_comparison.json.\n'
                     '- WV3 band names are not asserted: unchanged source indices0..7; QB labels come from source manifest.\n'
                     '- checksums.sha256 covers package files; sibling ZIP receipt gives archive SHA/CRC and size.\n')
    files = sorted(p for p in package.rglob('*') if p.is_file())
    checksum = package / 'checksums.sha256'
    with checksum.open('x') as stream:
        for path in files:
            stream.write(f'{sha(path)}  {path.relative_to(package).as_posix()}\n')
    files.append(checksum)
    archive = package.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for path in files:
            output.write(path, str(path.relative_to(package.parent)))
    with zipfile.ZipFile(archive) as output:
        require(output.testzip() is None, 'ZIP CRC failed')
        npy_entries = {str(Path(n).relative_to(package.name)) for n in output.namelist() if n.endswith('.npy')}
        require(npy_entries == expected_npys, 'ZIP does not contain required 80 raw arrays')
        for path in files:
            with output.open(str(path.relative_to(package.parent))) as member:
                h = hashlib.sha256()
                for block in iter(lambda: member.read(8 * 1024 * 1024), b''):
                    h.update(block)
            require(h.hexdigest() == sha(path), f'ZIP member differs: {path.name}')
    receipt = dict(package=str(package), zip=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
                   zip_crc_verified=True, zip_members_sha_verified=True, actual_npy_count=80, png_count=0,
                   raw_export_status='COMPLETE', delivery_status='READY_LOCAL_NOT_DELIVERED', completed_at=utcnow())
    write(args.delivery / 'DELIVERY_RECEIPT.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
