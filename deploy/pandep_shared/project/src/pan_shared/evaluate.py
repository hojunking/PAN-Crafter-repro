"""Isolated sensor-native evaluation, never a source of training/selection policy.

RR is ALL 20 scenes, MATLAB dim_cut=21 (Python 20:-21). FR is native
PAN/original LMS, full512, no shifting/masking. L1 selection is un-clipped
normalized validation, a sample mean within each sensor and then sensor macro.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import atomic_json, canonical_sha, file_sha
from .metrics.eval_rr import ergas, sam
from .metrics.extras import psnr_global, scc_dlpan, ssim_skimage
from .metrics.q2n import q2n

SENSOR = {'WV3': (8, 2047), 'GF2': (4, 1023), 'QB': (4, 2047)}
JQM_VARIANT = 'SRF-substitute (NNLS-normalized), MTF41, phase2, global CMSC, v1=0.5; not SIPSA-equivalent'
PROTOCOL = {
    'id': 'PANDEP_NATIVE_RR20_FR20_v1', 'prediction': 'float32 clip[-1,1] then native DN; no rounding',
    'rr': '20 scenes; dim21=20:-21; Q32; Sobel2D-zero SCC; global PSNR; Gaussian11 SSIM',
    'fr': '20 scenes; native PAN/original LMS; full512; mean per-scene HQNR; no masking',
    'jqm': JQM_VARIANT, 'validation_selection': 'unclipped normalized sample mean L1; macro sensors',
}


def evaluator_sha():
    """Numerical implementation + version, not an arbitrary model-state digest."""
    import scipy
    import skimage
    paths = [Path(__file__)] + sorted((Path(__file__).parent / 'metrics').glob('*.py'))
    return canonical_sha({'files': {str(p.relative_to(Path(__file__).parent)): file_sha(p) for p in paths},
                          'protocol': PROTOCOL, 'numpy': np.__version__, 'scipy': scipy.__version__,
                          'skimage': skimage.__version__, 'torch': torch.__version__})


@contextlib.contextmanager
def preserved_inference(model):
    """Preserve *all* module modes and global RNG streams (including DataLoader)."""
    modes = {module: module.training for module in model.modules()}
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, training in modes.items():
            module.training = training
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


def _output(model, sensor, batch, device):
    out = model(sensor, batch['pan'].to(device), batch['ms'].to(device), batch['lpan'].to(device))
    if not isinstance(out, torch.Tensor) or not torch.isfinite(out).all():
        raise FloatingPointError('Nonfinite or non-tensor model output')
    return out


def validation(model, datasets, device, batch_size=16):
    """Evaluate every sample once. Different sensor/batch sizes never bias L1."""
    if not datasets or any(s not in SENSOR for s in datasets):
        raise ValueError('Validation requires known sensors')
    sensors = {}
    with preserved_inference(model):
        for sensor, dataset in datasets.items():
            if getattr(dataset, 'augment', False):
                raise ValueError('Validation/probes must be unaugmented')
            records, losses = [], []
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
                pred = _output(model, sensor, batch, device)
                gt = batch['gt'].to(device)
                if pred.shape != gt.shape or not torch.isfinite(gt).all():
                    raise ValueError('Incomplete or invalid validation ground truth')
                losses.extend((pred.double() - gt.double()).abs().flatten(1).mean(1).cpu().tolist())
                scale = SENSOR[sensor][1] / 2
                # Diagnostic legacy validation convention: full64 DN, clip prediction only.
                native = ((pred.float().clamp(-1, 1) + 1) * scale).cpu().numpy().transpose(0, 2, 3, 1)
                truth = ((gt.float() + 1) * scale).cpu().numpy().transpose(0, 2, 3, 1)
                for p, g in zip(native.astype(np.float64), truth.astype(np.float64)):
                    records.append({'ergas': ergas(p, g), 'sam': sam(p, g),
                                    'psnr': psnr_global(p, g, SENSOR[sensor][1])})
            if len(losses) != len(dataset) or not losses or not np.isfinite(losses).all():
                raise ValueError('Incomplete/nonfinite validation')
            # Infinite PSNR for exact ground truth is diagnostic null, not invalid L1 selection.
            diagnostics = {k: float(np.mean([r[k] for r in records])) for k in ('ergas', 'sam', 'psnr')}
            sensors[sensor] = dict(mean_l1=float(np.mean(losses)), n_samples=len(losses),
                                   **{k: v if np.isfinite(v) else None for k, v in diagnostics.items()})
    return {'sensors': sensors, 'joint_l1': float(np.mean([s['mean_l1'] for s in sensors.values()])),
            'complete': True, 'selector_protocol': PROTOCOL['validation_selection']}


def _summary(rows, keys):
    if not rows:
        raise ValueError('No metric scenes')
    result = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError('Nonfinite official metrics')
    return dict(result, n_scenes=len(rows), per_scene=rows,
                standard_deviation={k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys})


def rr_metrics(sr, gt, sensor):
    bands, maximum = SENSOR[sensor]
    sr, gt = np.asarray(sr, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if sr.shape != (20, bands, 256, 256) or gt.shape != sr.shape:
        raise ValueError('Official RR requires exactly 20 corresponding sensor-band 256 scenes')
    if not np.isfinite(sr).all() or not np.isfinite(gt).all():
        raise FloatingPointError('Nonfinite RR inputs, including excluded borders')
    rows, qkey = [], 'q' + str(bands)
    for pred, truth in zip(sr.transpose(0, 2, 3, 1)[:, 20:-21, 20:-21],
                           gt.transpose(0, 2, 3, 1)[:, 20:-21, 20:-21]):
        if pred.std() == 0 or truth.std() == 0:
            raise ValueError('Undefined flattened Pearson CC')
        rows.append(dict(ergas=ergas(pred, truth), sam=sam(pred, truth), scc=scc_dlpan(pred, truth),
                         psnr=psnr_global(pred, truth, maximum), ssim=ssim_skimage(pred, truth, maximum),
                         rmse=float(np.sqrt(np.mean((pred-truth)**2))),
                         cc=float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1]),
                         **{qkey: float(q2n(truth, pred, 32, 32)[0])}))
    return dict(_summary(rows, ('ergas', 'sam', 'scc', 'psnr', 'ssim', 'rmse', 'cc', qkey)),
                sensor=sensor, crop='20:-21', q_block=32, official_complete=True)


def fr_metrics(sr, original_lms, native_pan, native_ms, sensor):
    from .metrics.eval_fr import mtf_filter, imresize_matlab, _blockproc_uqi
    from .metrics.interp23 import interp23tap
    from .metrics.jqm import jqm
    bands, maximum = SENSOR[sensor]
    arrays = [np.asarray(a, dtype=np.float64) for a in (sr, original_lms, native_pan, native_ms)]
    shapes = [(20, bands, 512, 512), (20, bands, 512, 512), (20, 1, 512, 512), (20, bands, 128, 128)]
    if any(a.shape != shape for a, shape in zip(arrays, shapes)):
        raise ValueError('FR requires full20 native512 PAN, original LMS and native128 MS')
    if not all(np.isfinite(a).all() for a in arrays):
        raise FloatingPointError('Nonfinite native FR inputs')
    sr, lms, pan, ms = arrays
    rows, signed = [], []
    for fused, original, p, m in zip(sr.transpose(0, 2, 3, 1), lms.transpose(0, 2, 3, 1),
                                    pan[:, 0], ms.transpose(0, 2, 3, 1)):
        filtered = mtf_filter(fused, sensor.lower(), 4)
        dl = float(1 - q2n(original, filtered, 32, 32)[0])
        low_pan = interp23tap(imresize_matlab(p, 1/4)[..., None], 4)[..., 0]
        delta = np.asarray([_blockproc_uqi(fused[..., b], p, 32) -
                            _blockproc_uqi(original[..., b], low_pan, 32) for b in range(bands)])
        ds = float(np.abs(delta).mean())
        jq = jqm(fused, m, p, sensor, ratio=4, R=maximum, lpf='mtf', window=None, v1=.5)
        signed.append(delta.tolist())
        rows.append(dict(d_lambda=dl, d_s=ds, hqnr=float((1-dl)*(1-ds)), jqm=float(jq['JQM'])))
    return dict(_summary(rows, ('hqnr', 'd_s', 'd_lambda', 'jqm')), sensor=sensor,
                reference='native_PAN_and_original_LMS', support='full512', masking=False,
                jqm_variant=JQM_VARIANT, signed_ds=float(np.mean(signed)),
                positive_fraction=float((np.asarray(signed) > 0).mean()),
                signed_ds_operands=signed, official_complete=True)


def infer(model, sensor, dataset, device):
    outputs = []
    with preserved_inference(model):
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            pred = _output(model, sensor, batch, device)
            outputs.append(((pred.float().clamp(-1, 1) + 1) * (SENSOR[sensor][1] / 2)).cpu().numpy())
    if len(outputs) != 20:
        raise ValueError('Benchmark inference must cover exactly 20 scenes')
    return np.concatenate(outputs)


def _raw_path(dataset):
    for attr in ('source_path', 'raw_h5_path', 'path', 'raw_path'):
        value = getattr(dataset, attr, None)
        if value:
            return Path(value)
    raise ValueError('Dataset lacks a verified native source path')


def assert_checkpoint_matches_model(model, checkpoint_path):
    from safetensors.torch import load_file
    saved = load_file(str(checkpoint_path), device='cpu')
    actual = model.state_dict()
    if set(saved) != set(actual) or any(not torch.equal(saved[k], actual[k].detach().cpu()) for k in saved):
        raise ValueError('BLOCKED_IDENTITY: evaluated tensors differ from actual model bytes')


def benchmark_checkpoint(model, checkpoint_path, datasets, device, cache_dir, identity=None):
    """Read-only sources, immutable content cache. Selector aliases reuse actual bytes."""
    import h5py
    assert_checkpoint_matches_model(model, checkpoint_path)
    weight_sha, metric_sha = file_sha(checkpoint_path), evaluator_sha()
    reports = {}
    for sensor, splits in datasets.items():
        if any(getattr(d, 'sensor', sensor) != sensor for d in splits.values()):
            raise ValueError('Benchmark dataset sensor differs from the active model path')
        rr_path, fr_path = _raw_path(splits['rr']), _raw_path(splits['fr'])
        key_data = {'checkpoint_sha256': weight_sha, 'evaluator_sha256': metric_sha,
                    'sensor': sensor, 'rr_sha256': file_sha(rr_path), 'fr_sha256': file_sha(fr_path),
                    'lp_cache_sha256': {split: file_sha(getattr(splits[split], 'lp_path'))
                                       if getattr(splits[split], 'lp_path', None) else None for split in ('rr', 'fr')},
                    'pipeline_sha256': (identity or {}).get('source_sha256') or canonical_sha({
                        name: file_sha(Path(__file__).parent/name) for name in ('model.py', 'blocks.py', 'frontend.py', 'data.py')}),
                    'environment_sha256': (identity or {}).get('environment_sha256'),
                    'device': str(device), 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                    'tf32_cudnn': torch.backends.cudnn.allow_tf32}
        key = canonical_sha(key_data)
        path = Path(cache_dir) / (key + '.json')
        if path.exists():
            cached = json.loads(path.read_text())
            sealed = {k: v for k, v in cached.items() if k != 'payload_sha256'}
            if cached.get('payload_sha256') != canonical_sha(sealed) or cached.get('identity') != key_data:
                raise ValueError('BLOCKED_IDENTITY: corrupt or mismatched evaluation cache')
            if any(cached[scope]['n_scenes'] != 20 or len(cached[scope]['per_scene']) != 20 for scope in ('rr', 'fr')):
                raise ValueError('Incomplete cached official metric')
            reports[sensor] = cached
            continue
        start = time.monotonic()
        rr_sr = infer(model, sensor, splits['rr'], device)
        with h5py.File(rr_path, 'r') as source:
            rr = rr_metrics(rr_sr, source['gt'][:], sensor)
        fr_sr = infer(model, sensor, splits['fr'], device)
        with h5py.File(fr_path, 'r') as source:
            fr = fr_metrics(fr_sr, source['lms'][:], source['pan'][:], source['ms'][:], sensor)
        report = {'schema': 'PANDEP_BENCHMARK_v1', 'identity': key_data, 'rr': rr, 'fr': fr,
                  'protocol': PROTOCOL, 'eval_seconds': time.monotonic() - start}
        report['payload_sha256'] = canonical_sha(report)
        atomic_json(path, report)
        reports[sensor] = report
    return reports


def profile_model(model, sensor, device):
    """Batch1/PAN256, warm20/repeat100; prepared inputs, no I/O or LP generation."""
    device = torch.device(device)
    bands = SENSOR[sensor][0]
    timings = []
    is_cuda = device.type == 'cuda'
    with preserved_inference(model):
        args = (sensor, torch.zeros(1, 1, 256, 256, device=device),
                torch.zeros(1, bands, 64, 64, device=device), torch.zeros(1, 1, 64, 64, device=device))
        for _ in range(20):
            model(*args)
        if is_cuda:
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        for _ in range(100):
            if is_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter(); model(*args)
            if is_cuda:
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    groups = {'total': sum(p.numel() for p in model.parameters())}
    for name in ('trunk', 'stems', 'heads', 'stem', 'head'):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            groups[name] = sum(p.numel() for p in module.parameters())
    if hasattr(model, 'stems') and hasattr(model, 'heads'):
        groups['active'] = groups['trunk'] + sum(p.numel() for p in model.stems[sensor].parameters()) + sum(p.numel() for p in model.heads[sensor].parameters())
    else:
        groups['active'] = groups['total']
    return {'sensor': sensor, 'infer_ms': float(np.median(timings)), 'p95_ms': float(np.percentile(timings, 95)),
            'mem_mb': torch.cuda.max_memory_allocated(device) / 2**20 if is_cuda else None,
            'params_m': groups['total'] / 1e6, 'parameter_counts': groups, 'flops_g': None,
            'flops_status': 'NOT_MEASURED: no unsupported-op or MAC conversion assumptions',
            'scope': 'batch1 PAN256/MS64/LP64 FP32 eval; warm20/repeat100; model only; prepared inputs',
            'device': str(device), 'hardware': torch.cuda.get_device_name(device) if is_cuda else 'CPU',
            'evaluator_sha256': evaluator_sha()}
