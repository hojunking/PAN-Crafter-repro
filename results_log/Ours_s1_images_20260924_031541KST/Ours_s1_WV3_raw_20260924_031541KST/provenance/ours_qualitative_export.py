"""Inference-only export of the two explicitly selected s1 paper checkpoints.

Run in the pinned original runtime image, with the original source checkout and
inputs mounted read-only. No training, checkpoint selection, or Sheets writes.
CPU execution is intentionally recorded as different from original CUDA inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile


TARGETS = {
    'WV3': dict(run='FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1',
                step=38380, sha256='3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50',
                manifest='work_dir/_fh12/s1/dataset_manifest.json', bands=8),
    'QB': dict(run='QG40_QB_S_PLH_W104_D122_TB_C3_TS81002_SS82001_BASE_FRESH50_v1',
               step=47470, sha256='548d0ed2ccdc968d319e850e1470d7068f9321a9f0cc18e1794807f5cfd3399d',
               manifest='work_dir/_qg40/s1/dataset_manifest.json', bands=4),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with path.open('x') as stream:
        stream.write(text)


def now():
    return datetime.now(timezone.utc).isoformat()


def file_record(path, root):
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=digest(path))


class OneScene:
    """Preserve the original dataset's conventions while persisting each scene."""
    def __init__(self, dataset, index):
        self.dataset, self.index = dataset, index
        self.has_gt = dataset.has_gt
        if hasattr(dataset, 'spec'):
            self.spec = dataset.spec

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.dataset[self.index]


def archive(package):
    checksum = package / 'checksums.sha256'
    files = sorted(p for p in package.rglob('*') if p.is_file())
    with checksum.open('x') as stream:
        for path in files:
            stream.write(f'{digest(path)}  {path.relative_to(package).as_posix()}\n')
    files.append(checksum)
    destination = package.with_suffix('.zip')
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for path in files:
            output.write(path, str(path.relative_to(package.parent)))
    with zipfile.ZipFile(destination) as output:
        bad = output.testzip()
        if bad:
            raise ValueError(f'ZIP CRC mismatch: {bad}')
        if len(output.namelist()) != len(files):
            raise ValueError('ZIP file count mismatch')
    result = file_record(destination, package.parent)
    result.update(zip_crc_verified=True, file_count=len(files))
    write_json(destination.with_suffix('.zip.receipt.json'), result)
    return result


def export_sensor(args, sensor, renderer):
    import h5py
    import numpy as np
    import torch
    import yaml

    started = now()
    target = TARGETS[sensor]
    run = args.root / 'work_dir' / target['run']
    candidate = run / 'candidates' / str(target['step'])
    identity = json.loads((candidate / 'identity.json').read_text())
    selection = json.loads((run / 'official/target_selection.json').read_text())
    config = yaml.safe_load((run / 'meta/config.resolved.yaml').read_text())
    data = json.loads((args.root / target['manifest']).read_text())
    if digest(candidate / 'model.safetensors') != target['sha256']:
        raise ValueError('Requested checkpoint SHA mismatch')
    if identity['model_sha256'] != target['sha256'] or identity['update'] != target['step']:
        raise ValueError('Checkpoint identity mismatch')
    if object_sha(config) != identity['config_sha256'] or object_sha(data) != identity['data_sha256']:
        raise ValueError('Original config/dataset identity mismatch')

    if sensor == 'WV3':
        from fh20r1 import common
        from fh12 import data as data_module, evaluation
    else:
        from qg40 import common, data as data_module, evaluation
    for module in (common, data_module, evaluation):
        if not Path(module.__file__).resolve().is_relative_to(args.runtime):
            raise ValueError(f'Not using the frozen runtime: {module.__file__}')
    actual_source = common.source_identity(args.runtime)
    source = identity['source_identity']
    keys = ('files', 'content_sha256', 'torch', 'numpy', 'scipy', 'skimage', 'cuda',
            'cudnn', 'tf32_matmul', 'tf32_cudnn', 'numeric_method_revision')
    differences = [key for key in keys if source.get(key) != actual_source.get(key)]
    if differences:
        raise ValueError(f'Frozen numerical code/runtime mismatch: {differences}')
    model, _ = common.load_checkpoint_model(config, candidate, device='cpu', expected_source=source)
    if model.training or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('FP32 eval model required')

    package = args.output / f'Ours_s1_{sensor}_raw_{args.stamp}'
    package.mkdir(parents=True, exist_ok=False)
    provenance = package / 'provenance'
    provenance.mkdir()
    for original, filename in [(candidate / 'identity.json', 'checkpoint_identity.json'),
                               (run / 'meta/config.resolved.yaml', 'config.resolved.yaml'),
                               (args.root / target['manifest'], 'dataset_manifest.json')]:
        shutil.copy2(original, provenance / filename)
    for name in ('ours_qualitative_export.py', 'ours_qualitative_render.py'):
        shutil.copy2(Path(__file__).parent / name, provenance / name)
    write_json(provenance / 'source_identity.json', source)
    write_json(provenance / 'actual_source_identity.json', actual_source)
    (provenance / 'checkpoint.sha256').write_text(f"{target['sha256']}  model.safetensors\n")
    write_json(package / 'metrics/original_selection.json', selection)
    original_metric = json.loads((run / 'official' / f"candidate_{target['step']}.json").read_text())
    write_json(package / 'metrics/original_candidate.json', original_metric)
    extra = run / 'supplemental_metrics' / f"step_{target['step']}.json"
    if extra.exists():
        shutil.copy2(extra, package / 'metrics/original_supplemental.json')

    band_order = data.get('band_order') if sensor == 'QB' else list(renderer.WV3_BANDS)
    scenes, rendering, reevaluation = [], [], {}
    for split, name, side in [('rr', 'reduced', 256), ('fr', 'full', 512)]:
        dataset = data_module.build_dataset(data, split, root=args.root, augment=False)
        if len(dataset) != 20 or dataset.has_gt != (split == 'rr'):
            raise ValueError(f'Incorrect {sensor}/{split} dataset count or GT status')
        pred_dir = package / 'Ours_raw' / sensor / name / 'pred'
        pred_dir.mkdir(parents=True)
        rows = []
        with h5py.File(dataset.raw_h5_path, 'r') as h5, h5py.File(dataset.lp_path, 'r') as lp:
            for index in range(20):
                clock = time.monotonic()
                pred, delta = evaluation.infer(model, OneScene(dataset, index), 'cpu', batch_size=1)
                pred = pred[0]
                if (pred.shape != (target['bands'], side, side) or pred.dtype != np.float32
                        or not np.isfinite(pred).all() or pred.min() < 0 or pred.max() > 2047):
                    raise ValueError('Prediction shape/dtype/range invalid')
                p = pred_dir / f'testimg{index}.npy'
                with p.open('xb') as stream:
                    np.save(stream, pred, allow_pickle=False)
                # Re-load the saved file: metrics and PNGs derive from actual delivered bytes.
                pred = np.load(p, allow_pickle=False)
                raw = {key: np.asarray(h5[key][index]) for key in ('gt', 'lms', 'ms', 'pan') if key in h5}
                raw['lpan'] = np.asarray(lp['lpan'][index])
                refs = {}
                for key, array in raw.items():
                    reference = package / 'references' / sensor / name / key / p.name
                    reference.parent.mkdir(parents=True, exist_ok=True)
                    with reference.open('xb') as stream:
                        np.save(stream, array, allow_pickle=False)
                    refs[key] = dict(file_record(reference, package), dtype=str(array.dtype), shape=list(array.shape))
                rendered = renderer.render_scene(pred, raw, sensor, package / 'Ours' / sensor / name,
                                                 index, band_order, 2047, split)
                scene = dict(sensor=sensor, split=name, scene_index_zero_based=index, h5_index=index,
                             source_h5=data['splits'][split]['dataroot'], source_h5_sha256=data['splits'][split]['sha256'],
                             lpan_h5_sha256=data['splits'][split]['lpan_sha256'],
                             prediction=dict(file_record(p, package), shape=list(pred.shape), dtype='float32',
                                             axis='CHW', unit='DN', minimum=float(pred.min()), maximum=float(pred.max())),
                             references=refs, predicted_delta=np.asarray(delta).tolist(),
                             correspondence='original H5 index verified; external paper scene/ROI not reverified')
                scenes.append(scene)
                rendering.append(dict(split=name, scene_index=index, **rendered))
                rows.append(pred)
                print(f'{sensor} {name} {index + 1}/20 saved raw+PNG ({time.monotonic()-clock:.1f}s)', flush=True)
        # Metrics use the saved full-band prediction files and original evaluator.
        predictions = np.stack(rows)
        print(f'{sensor} {name}: evaluating saved arrays with original evaluator', flush=True)
        if split == 'rr':
            gt = np.stack([np.load(package / 'references' / sensor / name / 'gt' / f'testimg{i}.npy', allow_pickle=False) for i in range(20)])
            rr = (evaluation.rr_metrics(predictions, gt) if sensor == 'WV3'
                  else evaluation.rr_metrics(predictions, gt, dataset.spec))
            reevaluation['rr'] = rr
        else:
            reevaluation['fr'] = evaluation.FRMetrics(dataset)(predictions)
        write_json(package / 'metrics' / f'{split}_re_evaluation.json', reevaluation[split])
        del dataset, predictions, rows

    comparisons = {}
    # Historical summaries keep per-split dictionaries under rr/fr.
    for split in ('rr', 'fr'):
        prior = original_metric.get(split, selection.get(split, {}))
        for key, value in reevaluation[split].items():
            if isinstance(value, (float, int)) and not isinstance(value, bool) and isinstance(prior.get(key), (float, int)):
                comparisons[f'{split}.{key}'] = dict(original=prior[key], exported=float(value),
                                                     difference=float(value-prior[key]))
    if not all(f'{split}.{key}' in comparisons for split, key in [('rr', 'ergas'), ('fr', 'hqnr'), ('fr', 'd_lambda'), ('fr', 'd_s')]):
        raise ValueError('Original metric schema not mapped; raw outputs retained for inspection')
    write_json(package / 'metrics/re_evaluation.json', reevaluation)
    write_json(package / 'metrics/metric_comparison.json', dict(status='PARITY_REVIEW_REQUIRED',
               reason='CPU export vs original CUDA; no predeclared reproducibility tolerance. Differences reported without claiming bit parity.',
               metrics=comparisons))
    write_json(package / 'scene_manifest.json', scenes)
    write_json(package / 'rendering_requirements.json', dict(status='PREVIEW_ONLY', paper_comparison_ready=False,
               raw_status='RAW_READY', canconv_renderer_and_roi='not available; native full-frame previews only', scenes=rendering))
    manifest = dict(schema='OURS_S1_INFERENCE_EXPORT_v1', sensor=sensor, run_id=target['run'],
                    selection='TARGET', step=target['step'], checkpoint_sha256=target['sha256'],
                    config_sha256=identity['config_sha256'], dataset_manifest_sha256=identity['data_sha256'],
                    bands=band_order, band_order_status='QB manifest verified' if sensor=='QB' else 'repository WV3 canonical convention',
                    num_bands=target['bands'], max_dn=2047, dtype='float32', axis='CHW', unit='DN',
                    raw_prediction_count=40, split_counts=dict(reduced=20, full=20),
                    png_count=len(list(package.rglob('*.png'))), execution='fresh inference',
                    original_runtime=source, actual_runtime=actual_source, actual_device='cpu',
                    docker_image=args.image, cpu_threads=torch.get_num_threads(), started_utc=started, completed_utc=now(),
                    inference=dict(mode='FP32 eval no_grad A_ON, whole checkpoint A+frontend+U', gt_in_forward=False,
                                   batch_size=1, shuffle=False, augment=False, mask=False,
                                   input_normalization='DN * (2/2047) - 1', output='(clamp(y,-1,1)+1)*1023.5; no DN rounding',
                                   crop_or_resize_raw_or_standalone_png=False),
                    evaluation=dict(rr='20 scenes, crop20:-21, original frozen evaluator',
                                    fr='20 scenes, native PAN/LMS, full512, no masking, mean per-scene HQNR',
                                    source='saved NPY bytes'),
                    parity_status='PARITY_REVIEW_REQUIRED', raw_status='RAW_READY', png_status='PREVIEW_ONLY',
                    paper_comparison_ready=False, training_changed=False, external_upload_performed=False)
    write_json(package / 'manifest.json', manifest)
    (package / 'README.md').write_text(
        f'# {sensor} actual raw predictions and preview images\n\n'
        f'Run: {target["run"]}; TARGET step {target["step"]}.\n\n'
        'This package contains real inference outputs, not just metadata.\n'
        f'- PNG predictions: `Ours/{sensor}/reduced/output/` and `Ours/{sensor}/full/output/`.\n'
        f'- Full-band float32 DN predictions: `Ours_raw/{sensor}/`, 20 RR + 20 FR.\n'
        '- References include only the corresponding 20+20 test slices, unchanged source values.\n'
        '- Standalone PNGs retain the entire native frame; panels add labels outside it.\n'
        '- RGB uses shared per-scene reference 1–99% per-band stretch, gamma=1.\n'
        '- PNGs are inspection previews; original CANConv renderer/ROI was unavailable.\n'
        '- Frozen code and package versions match the original checkpoint. CPU export differs from original CUDA execution.\n'
        '- See metrics/metric_comparison.json for differences: no predeclared parity tolerance, so review is required.\n'
        '- Original full-band arrays are available for independent rendering/evaluation.\n'
        '- No training, Sheets upload, or checkpoint selection was performed.\n', encoding='utf-8')
    result = archive(package)
    print(json.dumps(dict(sensor=sensor, status='RAW_AND_PREVIEWS_SAVED', archive=result, metrics=comparisons)), flush=True)
    return dict(sensor=sensor, package=str(package), archive=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stamp', required=True)
    parser.add_argument('--sensor', choices=['WV3', 'QB', 'both'], default='both')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    args.root, args.runtime, args.output = (p.resolve() for p in (args.root, args.runtime, args.output))
    if not 1 <= args.threads <= 4 or not args.output.is_relative_to(args.root / 'results_log'):
        raise ValueError('Bounded CPU threads and results_log destination required')
    sys.path.insert(0, str(args.runtime))
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        raise ValueError('This exporter must run without GPU access')
    spec = importlib.util.spec_from_file_location('ours_qualitative_render', Path(__file__).with_name('ours_qualitative_render.py'))
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    args.output.mkdir(parents=True, exist_ok=True)
    sensors = list(TARGETS) if args.sensor == 'both' else [args.sensor]
    results = [export_sensor(args, sensor, renderer) for sensor in sensors]
    write_json(args.output / 'EXPORT_RECEIPTS.json', results)


if __name__ == '__main__':
    main()
