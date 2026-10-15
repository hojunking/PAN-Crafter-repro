"""Package the verified, fixed s1 WV3/QB FR predictions without new inference.

The previous RR export module supplies only stdlib file/hash helpers and fixed
checkpoint IDs. No RR dataset, model forward, GT extraction, or renderer runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import sys
import zipfile
from zoneinfo import ZoneInfo

from ours_rr_raw_gt_export import TARGETS, object_sha, read, require, sha, utcnow, write


TASK_ID = 'PANDA_S1_WV3_QB_FR_RAW_SUPPLEMENT_v1'
NUMERICAL_KEYS = ('files', 'content_sha256', 'torch', 'numpy', 'scipy', 'skimage',
                  'cuda', 'cudnn', 'tf32_matmul', 'tf32_cudnn', 'numeric_method_revision')


def export_sensor(args, package, sensor):
    import h5py
    import numpy as np
    import yaml

    run, step, checkpoint_sha, channels = TARGETS[sensor]
    source = args.source / f'Ours_s1_{sensor}_raw_20260924_031541KST'
    checksums = {}
    for line in (source / 'checksums.sha256').read_text().splitlines():
        value, relative = line.split('  ', 1)
        require(relative not in checksums, 'Duplicate prior checksum path')
        checksums[relative] = value

    def verified(relative):
        path = source / relative
        require(path.resolve().is_relative_to(source), 'Prior path escapes package')
        require(checksums.get(relative) == sha(path), f'Prior checksum mismatch: {relative}')
        return path

    prior = read(verified('manifest.json'))
    prior_scenes = read(verified('scene_manifest.json'))
    identity = read(verified('provenance/checkpoint_identity.json'))
    selection = read(verified('metrics/original_selection.json'))
    original = read(verified('metrics/original_candidate.json'))
    config_path = verified('provenance/config.resolved.yaml')
    config = yaml.safe_load(config_path.read_text())
    data_path = verified('provenance/dataset_manifest.json')
    data = read(data_path)
    source_identity = read(verified('provenance/source_identity.json'))
    inference_runtime = read(verified('provenance/actual_source_identity.json'))
    live = args.root / 'work_dir' / run
    candidate = live / 'candidates' / str(step)
    require(sha(candidate / 'model.safetensors') == checkpoint_sha, 'Live checkpoint SHA mismatch')
    require(identity == read(candidate / 'identity.json') == selection['checkpoint_identity'] == original['checkpoint_identity'], 'Identity binding mismatch')
    require(selection == read(live / 'official/target_selection.json'), 'Live selected TARGET differs')
    require(original == read(live / 'official' / f'candidate_{step}.json'), 'Original candidate differs')
    require(selection['run_id'] == run and selection['selection_id'] == 'TARGET' and selection['step'] == step, 'Wrong selection')
    require(selection['checkpoint_sha256'] == identity['model_sha256'] == checkpoint_sha, 'Wrong checkpoint')
    require(identity['update'] == original['update'] == step and selection['normal_same_step_A_U'], 'Wrong step or A/U combination')
    require(sha(live / 'meta/config.resolved.yaml') == sha(config_path), 'Live config differs')
    require(object_sha(config) == identity['config_sha256'] == prior['config_sha256'], 'Config hash mismatch')
    require(object_sha(data) == identity['data_sha256'] == prior['dataset_manifest_sha256'], 'Dataset identity mismatch')
    require(source_identity == identity['source_identity'] == prior['original_runtime'], 'Source provenance mismatch')
    require(inference_runtime == prior['actual_runtime'], 'Inference runtime mismatch')
    require(all(source_identity[k] == inference_runtime[k] for k in NUMERICAL_KEYS), 'Prior inference numerical code/runtime mismatch')
    for relative, expected_sha in source_identity['files'].items():
        path = (args.root.parent / 'DLPan-Toolbox/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py'
                if relative == 'external/DLPan/wald_utilities.py' else args.runtime / relative)
        require(sha(path) == expected_sha, f'Frozen source differs: {relative}')
    require(prior['sensor'] == sensor and prior['run_id'] == run and prior['step'] == step
            and prior['checkpoint_sha256'] == checkpoint_sha and prior['selection'] == 'TARGET', 'Wrong prior model output')
    require(prior['axis'] == 'CHW' and prior['unit'] == 'DN' and prior['dtype'] == 'float32'
            and prior['max_dn'] == 2047 and prior['num_bands'] == channels, 'Prior output convention mismatch')
    policy = prior['inference']
    require('A_ON' in policy['mode'] and 'FP32' in policy['mode'] and policy['batch_size'] == 1, 'Wrong forward policy')
    require(not any(policy[k] for k in ('gt_in_forward', 'shuffle', 'augment', 'mask', 'crop_or_resize_raw_or_standalone_png')), 'Invalid prior postprocessing')
    require(policy['output'] == '(clamp(y,-1,1)+1)*1023.5; no DN rounding', 'Unknown prediction normalization')
    field = config['fh20r1' if sensor == 'WV3' else 'qg40']
    bound_manifest = args.root / field['dataset_manifest']
    require(read(bound_manifest) == data, 'Resolved config manifest binding mismatch')
    binding = data['splits']['fr']
    raw_path, lp_path = Path(binding['dataroot']), Path(binding['lpan_path'])
    require((args.root / config['test_full_feeder_args']['dataroot']).resolve() == raw_path.resolve(), 'Resolved config FR path mismatch')
    require(binding['count'] == 20 and sha(raw_path) == binding['sha256'], 'Native FR identity mismatch')
    require(sha(lp_path) == binding['lpan_sha256'], 'Original LP identity mismatch')
    with h5py.File(raw_path, 'r') as raw, h5py.File(lp_path, 'r') as lp:
        require('gt' not in raw, 'Unexpected GT in native FR source')
        for key, expected in [('pan', (20, 1, 512, 512)), ('ms', (20, channels, 128, 128)), ('lms', (20, channels, 512, 512))]:
            require(raw[key].shape == expected, f'Original FR {key} shape mismatch')
        require(lp['lpan'].shape == (20, 1, 128, 128), 'Original FR LP shape mismatch')
    old_rows = [row for row in prior_scenes if row['sensor'] == sensor and row['split'] == 'full']
    require(len(old_rows) == 20 and sorted(row['scene_index_zero_based'] for row in old_rows) == list(range(20)), 'Invalid FR index mapping')
    old_rows.sort(key=lambda row: row['scene_index_zero_based'])
    band_order = dict(source_indices_zero_based=list(range(channels)), labels=data.get('band_order'),
                      status='SOURCE_MANIFEST_VERIFIED' if 'band_order' in data else 'SOURCE_NAMES_UNCONFIRMED_INDICES_PRESERVED',
                      permutation_applied=False)
    rows, checks = [], []
    for index, old in enumerate(old_rows):
        relative = f'Ours_raw/{sensor}/full/pred/testimg{index}.npy'
        require(old['prediction']['path'] == relative and old['h5_index'] == index, 'Prediction/H5 index mismatch')
        require(old['source_h5'] == str(raw_path) and old['source_h5_sha256'] == binding['sha256']
                and old['lpan_h5_sha256'] == binding['lpan_sha256'], 'Prediction input provenance mismatch')
        origin = verified(relative)
        require(sha(origin) == old['prediction']['sha256'], 'Prior scene prediction SHA mismatch')
        original_array = np.load(origin, allow_pickle=False)
        require(original_array.dtype == np.float32 and original_array.shape == (channels, 512, 512)
                and np.isfinite(original_array).all() and original_array.min() >= 0 and original_array.max() <= 2047,
                'Invalid prediction dtype/shape/finite/range')
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), 'Refusing to overwrite an existing prediction')
        shutil.copyfile(origin, destination)
        saved = np.load(destination, allow_pickle=False)
        require(sha(destination) == sha(origin) and np.array_equal(saved, original_array), 'Prediction copy changed')
        record = dict(path=relative, shape=list(saved.shape), dtype='float32', axis_order='CHW', units='DN',
                      minimum=float(saved.min()), maximum=float(saved.max()), sha256=sha(destination), bytes=destination.stat().st_size)
        rows.append(dict(sensor=sensor, export_index=index, filename=f'testimg{index}.npy', prediction=record,
                         source_fr_h5=str(raw_path), source_fr_h5_sha256=binding['sha256'], source_h5_index=index,
                         source_input_keys=['pan', 'ms', 'lms'], source_lp_path=str(lp_path), source_lp_sha256=binding['lpan_sha256'],
                         source_prediction_path=str(origin), band_order=band_order,
                         scene_order='FR_H5_ROW_INDEX_UNCHANGED', cross_method_order='USER_CONFIRMED_PREVIEW_ORDER',
                         external_png_order_revalidated=False))
        checks.append(dict(sensor=sensor, index=index, shape_dtype_finite_range_passed=True,
                           source_index_matches_export_index=True, prediction_byte_identical=True, numeric_roundtrip_exact=True))
    provenance = package / 'provenance' / sensor
    provenance.mkdir(parents=True)
    for relative, name in [('provenance/checkpoint_identity.json', 'checkpoint_identity.json'),
                           ('metrics/original_selection.json', 'original_selection.json'),
                           ('metrics/original_candidate.json', 'original_candidate.json'),
                           ('provenance/config.resolved.yaml', 'config.resolved.yaml'),
                           ('provenance/dataset_manifest.json', 'dataset_manifest.json'),
                           ('provenance/source_identity.json', 'source_identity.json'),
                           ('provenance/actual_source_identity.json', 'prediction_runtime.json'),
                           ('manifest.json', 'prior_export_manifest.json'),
                           ('scene_manifest.json', 'prior_scene_manifest.json')]:
        shutil.copyfile(verified(relative), provenance / name)
    shutil.copyfile(source / 'checksums.sha256', provenance / 'prior_checksums.sha256')
    write(provenance / 'reuse_binding.json', dict(source_package=str(source), source_manifest_sha256=sha(source / 'manifest.json'),
          source_scene_manifest_sha256=sha(source / 'scene_manifest.json'), prior_checksums_sha256=sha(source / 'checksums.sha256'),
          config_bound_manifest=str(bound_manifest), raw_sha_verified=True, lp_sha_verified=True, source_files_verified=len(source_identity['files'])))

    metrics = dict(metric_parity='NOT_RECOMPUTED', original=original['fr'], saved_arrays=None)
    try:
        if sensor == 'WV3':
            from fh20r1 import common
            from fh12 import evaluation, data as dataset_module
        else:
            from qg40 import common, evaluation, data as dataset_module
        require(Path(evaluation.__file__).resolve().is_relative_to(args.runtime), 'Wrong FR evaluator')
        actual_runtime = common.source_identity(args.runtime)
        require(all(actual_runtime[k] == source_identity[k] for k in NUMERICAL_KEYS), 'FR evaluator runtime mismatch')
        dataset = dataset_module.build_dataset(data, 'fr', root=args.root, augment=False)
        require(len(dataset) == 20 and not dataset.has_gt, 'FR-only native dataset required')
        predictions = np.stack([np.load(package / row['prediction']['path'], allow_pickle=False) for row in rows])
        print(f'{sensor}: saved FR20; re-evaluating HQNR/D_lambda/D_s from delivered NPY', flush=True)
        result = evaluation.FRMetrics(dataset)(predictions)
        comparison = {key: dict(original=original['fr'][key], saved_arrays=result[key], difference=result[key]-original['fr'][key])
                      for key in ('hqnr', 'd_lambda', 'd_s')}
        metrics.update(metric_parity='PARITY_REVIEW_REQUIRED', saved_arrays=result, comparison=comparison,
                       evaluator_runtime=actual_runtime, evaluator_path=str(evaluation.__file__),
                       reason='Reused CPU inference vs original CUDA results; no predeclared numerical tolerance.',
                       evaluated_from_saved_npy=True, reference='native_PAN_and_LMS', support='full512', masking=False,
                       aggregation='mean_per_scene_HQNR')
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        metrics['reason'] = f'{type(error).__name__}: {error}'
    record = dict(sensor=sensor, run_id=run, origin_server='s1', selection='TARGET', selected_step=step,
                  checkpoint_sha256=checkpoint_sha, config_sha256=identity['config_sha256'], dataset_manifest_sha256=identity['data_sha256'],
                  actual_fr_source=str(raw_path), actual_fr_source_sha256=binding['sha256'], lp_input_path=str(lp_path), lp_input_sha256=binding['lpan_sha256'],
                  original_source_identity=source_identity, original_inference_runtime=inference_runtime, band_order=band_order,
                  shape=[channels, 512, 512], axis_order='CHW', dtype='float32', units='DN', max_dn=2047, scene_count=20,
                  reuse_or_inference='reused', source_export=str(source), prediction_execution_device=prior['actual_device'],
                  prediction_generated_at=prior['started_utc'], normalization_and_clamp_policy=policy, generated_at=utcnow(),
                  raw_export_status='FR_RAW_COMPLETE', rendering_status='RENDERING_SKIPPED_BY_SCOPE',
                  cross_method_order='USER_CONFIRMED_PREVIEW_ORDER', external_png_order_revalidated=False,
                  metric_parity=metrics['metric_parity'], delivery_status='READY_LOCAL_NOT_DELIVERED')
    print(f'{sensor}: FR20 verified, {metrics["metric_parity"]}', flush=True)
    return record, rows, checks, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'runtime', 'source', 'delivery'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    for key in ('root', 'runtime', 'source', 'delivery'):
        setattr(args, key, getattr(args, key).resolve())
    require(args.delivery.is_relative_to(args.root / 'results_log'), 'Destination must be in results_log')
    sys.path.insert(0, str(args.runtime))
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    require(not torch.cuda.is_available(), 'This packaging job must not have GPU access')
    timestamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%SKST')
    package = args.delivery / f'Ours_s1_WV3_QB_FR_raw_{timestamp}'
    package.mkdir(parents=True, exist_ok=False)
    sensors, scenes, checks, metrics = {}, [], [], {}
    for sensor in TARGETS:
        record, rows, tests, report = export_sensor(args, package, sensor)
        sensors[sensor] = record
        scenes.extend(rows)
        checks.extend(tests)
        metrics[sensor] = report
    expected = {row['prediction']['path'] for row in scenes}
    inventory = {str(p.relative_to(package)) for p in package.rglob('*.npy')}
    require(expected == inventory and len(expected) == 40, 'FR40 file inventory mismatch')
    require(not list(package.rglob('*.png')) and not (package / 'references').exists(), 'FR prediction-only package required')
    write(package / 'manifest.json', dict(task_id=TASK_ID, generated_at=utcnow(), raw_export_status='FR_RAW_COMPLETE',
          rendering_status='RENDERING_SKIPPED_BY_SCOPE', cross_method_order='USER_CONFIRMED_PREVIEW_ORDER',
          external_png_order_revalidated=False, delivery_status='READY_LOCAL_NOT_DELIVERED', sensors=sensors,
          npy_count=40, execution_device='cpu: copy/validation/metrics only', new_inference_performed=False,
          training_changed=False, rr_included=False, gt_included=False, reference_arrays_included=False,
          external_upload_performed=False, zip_layout='manifest.json and Ours_raw at ZIP root; no enclosing directory'))
    write(package / 'scene_manifest.json', scenes)
    write(package / 'validation/raw_array_checks.json', dict(status='PASS', required_npy_count=40,
          actual_npy_count=40, png_count=0, rr_or_gt_count=0, per_scene=checks))
    write(package / 'validation/fr_metric_comparison.json', metrics)
    for name in ('ours_fr_raw_export.py', 'ours_rr_raw_gt_export.py'):
        shutil.copyfile(Path(__file__).with_name(name), package / 'provenance' / name)
    instruction = args.root / 'research_log/S1_WV3_QB_FR_RAW_EXPORT_SUPPLEMENT_2026-09-25_KR.md'
    shutil.copyfile(instruction, package / 'provenance/FR_EXPORT_INSTRUCTION_KR.md')
    with (package / 'README.md').open('x') as stream:
        stream.write('# WV3 + QB FR raw supplement\n\n'
                     'FR_RAW_COMPLETE / RENDERING_SKIPPED_BY_SCOPE / READY_LOCAL_NOT_DELIVERED\n\n'
                     '- WV3 TARGET38380:20 float32 CHW(8,512,512) predictions.\n'
                     '- QB TARGET47470:20 float32 CHW(4,512,512) predictions.\n'
                     '- Ours_raw/<sensor>/full/pred/testimg0..19.npy at ZIP root.\n'
                     '- DN0..2047, native512x512, all bands, unchanged source order.\n'
                     '- Source FR H5 row index0..19 preserved. USER_CONFIRMED_PREVIEW_ORDER, not externally reverified here.\n'
                     '- Byte-identical reuse of verified CPU predictions; no new inference or postprocessing.\n'
                     '- No RR, GT, PNG, error maps, training, checkpoint reselection, or reference arrays.\n'
                     '- Source PAN/MS/LP inputs were verified; their arrays are omitted from delivery only.\n'
                     '- Original selection metadata is retained, including historical RR fields, without RR arrays/re-evaluation.\n'
                     '- CPU vs original CUDA metric differences are recorded, not certified as bit-identical.\n'
                     '- See validation/fr_metric_comparison.json and raw_array_checks.json.\n'
                     '- WV3 names unconfirmed: original band indices0..7; QB band labels from source manifest.\n'
                     '- Existing training/previous exports unchanged. Local ZIP requires user retrieval or separately authorized transfer.\n')
    files = sorted(path for path in package.rglob('*') if path.is_file())
    checksum = package / 'checksums.sha256'
    with checksum.open('x') as stream:
        for path in files:
            stream.write(f'{sha(path)}  {path.relative_to(package).as_posix()}\n')
    files.append(checksum)
    archive = package.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for path in files:
            output.write(path, path.relative_to(package).as_posix())
    with zipfile.ZipFile(archive) as output:
        require(output.testzip() is None, 'ZIP CRC failed')
        require(set(output.namelist()) == {str(path.relative_to(package)) for path in files}, 'ZIP inventory mismatch')
        require({name for name in output.namelist() if name.endswith('.npy')} == expected, 'ZIP must contain the exact FR40 NPY files')
        for path in files:
            with output.open(path.relative_to(package).as_posix()) as member:
                h = hashlib.sha256()
                for block in iter(lambda: member.read(8 * 1024 * 1024), b''):
                    h.update(block)
            require(h.hexdigest() == sha(path), 'ZIP member differs from verified package')
    receipt = dict(package=str(package), zip=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
                   zip_crc_verified=True, zip_members_sha_verified=True, actual_npy_count=40, png_count=0,
                   raw_export_status='FR_RAW_COMPLETE', delivery_status='READY_LOCAL_NOT_DELIVERED', completed_at=utcnow())
    write(args.delivery / 'DELIVERY_RECEIPT.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
