"""Separate train-only mixed calibration, never a native reference overwrite.

All q entries are measured at batch16. The native slice is checked in full
against the parent's immutable cache before its exact values are reused.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch

from fh12.calibration import AXIS16, RADII, axis16_q, _write_npz_immutable
from fh12.data import canonical_sha, sha256_file, check_deadline, write_immutable_json
from qg40.calibration import select_calibration_indices
from qg40.reference_parity import ATOL, RTOL, isolated_teacher
from qg40.model import state_hash
from gfb20.augmentation import (MEDIAN_TOKENS, VIEW_RECIPE, prepare_augmentation,
                                verify_augmentation_cache)
from gfb20.data import GFB20Dataset

BATCH = 16


def weighted_median_tokens(values, gamma_axis):
    """The ordinary exact pooled median with the native token duplicated."""
    x = np.asarray(values)
    if x.shape[gamma_axis] != 3 or not np.isfinite(x).all():
        raise ValueError('Finite three-gamma values required')
    return float(np.median(np.take(x, MEDIAN_TOKENS, axis=gamma_axis)))


def verify_native_slice(measured, original):
    measured, original = np.asarray(measured), np.asarray(original)
    if (measured.shape != original.shape or measured.ndim != 2 or measured.shape[1] != 4
            or not np.isfinite(measured).all() or not np.isfinite(original).all()
            or (measured < 0).any() or (original < 0).any()):
        raise ValueError('Native q slice shape/numerical mismatch')
    difference = np.abs(measured.astype(np.float64) - original.astype(np.float64))
    if not np.allclose(measured, original, atol=ATOL, rtol=RTOL):
        raise ValueError(f'Native gamma1 AXIS16 parity failed: max={difference.max()}')
    return dict(passed=True, population=int(len(original)), views=4, batch_size=BATCH,
        atol=ATOL, rtol=RTOL, max_abs_error=float(difference.max()),
        mean_abs_error=float(difference.mean()),
        note='Full gamma1 recomputation at batch16; original native cache retained after parity. '
             'FP32 batch-shape kernel differences are bounded by pre-existing native parity tolerance.')


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _rows(dataset, ids, gamma, device, view=None):
    rows = [dataset.get_view(int(i), 0 if view is None else view,
        augment=view is not None, gamma_id=gamma) for i in ids]
    return tuple(torch.stack([row[k] for row in rows]).to(device) for k in range(5))


def runtime_identity(device):
    value = dict(torch=torch.__version__, numpy=np.__version__, device=str(device),
        batch_size=BATCH, dtype='float32', amp=False,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic)
    if str(device).startswith('cuda'):
        value.update(gpu_name=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                     cudnn=torch.backends.cudnn.version())
    return value


def compute_mixed_calibration(model, dataset, indices, native_q, *, device='cuda',
        batch_size=16, deadline_utc=None, progress_path=None, identity=None,
        synthetic_test=False):
    """Resumable whole-population q plus pooled pixel-error calibration.

    A completed chunk's checksum is checked on resume; a torn/incomplete chunk
    is recomputed. An expired deadline publishes no consumable reference.
    """
    if batch_size != BATCH:
        raise ValueError('GFB20 q generation/parity is fixed to batch16')
    indices = np.asarray(indices)
    n = dataset.base_count
    if (dataset.split != 'train' or not dataset.has_gt or model.bands != 4
            or len(dataset) != n or indices.ndim != 1 or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= n):
        raise ValueError('Mixed calibration requires distinct valid train-only base IDs')
    if not synthetic_test and (n != 19809 or not np.array_equal(indices, select_calibration_indices(n))):
        raise ValueError('Mixed calibration requires GF2 train19809 and seed1234 train3072')
    native_q = np.asarray(native_q)
    if native_q.shape != (n, 4) or not np.isfinite(native_q).all() or (native_q < 0).any():
        raise ValueError('Original full native q cache is required')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Mixed calibration requires FP32 Teacher')
    if progress_path is None:
        with tempfile.TemporaryDirectory(prefix='gfb20-cal-') as temporary:
            return compute_mixed_calibration(model, dataset, indices, native_q, device=device,
                batch_size=batch_size, deadline_utc=deadline_utc,
                progress_path=Path(temporary) / 'progress.h5', identity=identity,
                synthetic_test=synthetic_test)
    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = dataset.base(0)[0].shape[-2:]
    teacher_state = state_hash(model.state_dict())
    contract = dict(binding=identity, teacher_state_sha256=teacher_state,
        indices=indices.tolist(), native_q_sha256=_digest(native_q), runtime=runtime_identity(device),
        recipe=VIEW_RECIPE, synthetic_test=bool(synthetic_test), count=n)
    with h5py.File(path, 'a') as store:
        if 'identity_sha256' not in store.attrs:
            store.attrs['identity_sha256'] = canonical_sha(contract)
            store.create_dataset('q', (n, 4, 3), dtype='float32', fillvalue=np.nan)
            store.create_dataset('per_radius_q', (n, 4, 3, 4), dtype='float32', fillvalue=np.nan)
            store.create_dataset('delta', (n, 4, 3, 2), dtype='float32', fillvalue=np.nan)
            store.create_dataset('errors', (len(indices), 3, height, width), dtype='float32', fillvalue=np.nan)
            store.create_dataset('q_checksums', ((n + 15)//16, 4, 3), dtype='S64')
            store.create_dataset('e_checksums', ((len(indices) + 15)//16, 3), dtype='S64')
            store.flush()
        if store.attrs['identity_sha256'] != canonical_sha(contract):
            raise ValueError('Cannot resume mixed calibration under a different source/Teacher/runtime')
        with isolated_teacher(model):
            for gamma in range(3):
                for chunk, start in enumerate(range(0, len(indices), BATCH)):
                    check_deadline(deadline_utc)
                    stop = min(start + BATCH, len(indices))
                    previous = store['e_checksums'][chunk, gamma].decode()
                    if previous:
                        if previous != _digest(store['errors'][start:stop, gamma]):
                            raise ValueError('Completed mixed error chunk changed')
                        continue
                    gt, _, ms, lp, pan = _rows(dataset, indices[start:stop], gamma, device)
                    # Same gamma input, no geometry augmentation for tau.
                    output = model(pan, ms, lp)['y']
                    errors = (output.float() - gt.float()).abs().mean(dim=1).cpu().numpy()
                    if not np.isfinite(errors).all():
                        raise ValueError('Nonfinite mixed Teacher error')
                    store['errors'][start:stop, gamma] = errors
                    store.flush()
                    store['e_checksums'][chunk, gamma] = _digest(errors)
                    store.flush()
                for view in range(4):
                    for chunk, start in enumerate(range(0, n, BATCH)):
                        check_deadline(deadline_utc)
                        stop = min(start + BATCH, n)
                        previous = store['q_checksums'][chunk, view, gamma].decode()
                        if previous:
                            parts = [store[key][start:stop, view, gamma]
                                     for key in ('q', 'per_radius_q', 'delta')]
                            if previous != canonical_sha([_digest(x) for x in parts]):
                                raise ValueError('Completed mixed q chunk changed')
                            continue
                        _, _, ms, _, pan = _rows(dataset, np.arange(start, stop), gamma, device, view)
                        # Gain precedes this probe; axis16_q warps the supplied Pgamma.
                        values = axis16_q(model, pan, ms)
                        parts = [value.detach().cpu().numpy() for value in values]
                        if any(not np.isfinite(x).all() for x in parts) or (parts[0] < 0).any():
                            raise ValueError('Nonfinite mixed AXIS16 q/delta')
                        for key, value in zip(('q', 'per_radius_q', 'delta'), parts):
                            store[key][start:stop, view, gamma] = value
                        store.flush()
                        store['q_checksums'][chunk, view, gamma] = canonical_sha([_digest(x) for x in parts])
                        store.flush()
        q, errors = store['q'][:], store['errors'][:]
        parity = verify_native_slice(q[:, :, 1], native_q)
        online_native = q[:, :, 1].copy()
        q[:, :, 1] = native_q  # Reuse only after complete native parity.
        raw_tau = weighted_median_tokens(errors, 1)
        tau = max(raw_tau, 1e-6)
        qref = weighted_median_tokens(q[indices], 2)
        if not np.isfinite(qref) or qref <= 0:
            raise ValueError('Invalid mixed q_ref; zero cannot be repaired with epsilon')
        arrays = dict(q=q, calibration_indices=indices.astype(np.int64),
            native_online_q=online_native, per_radius_q=store['per_radius_q'][:],
            native_delta=store['delta'][:])
    if state_hash(model.state_dict()) != teacher_state:
        raise ValueError('Teacher tensors changed during mixed calibration')
    check_deadline(deadline_utc)
    calibration = dict(schema='GFB20_MIXED_CALIBRATION_v1', synthetic_test=bool(synthetic_test),
        tau_R=tau, q_ref=qref, tau_raw_pooled_median=raw_tau, tau_floor_used=raw_tau < 1e-6,
        calibration_n=len(indices), calibration_seed=1234,
        calibration_indices_sha256=canonical_sha(indices.tolist()), q_shape=list(q.shape),
        tau_rule='median all calibration pixels of band-mean absolute error; no geometry',
        q_ref_rule='median calibration IDs x four geometry views x duplicated gamma tokens',
        median_tokens=list(MEDIAN_TOKENS), gamma_probabilities=[.25, .5, .25],
        AXIS16=[list(x) for x in AXIS16], radii=list(RADII), native_slice_parity=parity,
        teacher_state_sha256=teacher_state, runtime=runtime_identity(device),
        q_formula='mean16(mean_xy(abs(c(Warp(Pgamma,epsilon),M)+epsilon-c(Pgamma,M))))',
        error_population=int(errors.size + errors[:, 1].size),
        q_ref_population=int(q[indices].size + q[indices, :, 1].size),
        gamma_tau_unweighted=[float(np.median(errors[:, gamma])) for gamma in range(3)],
        full_train_q_median_diagnostic_only=float(np.median(q)))
    return calibration, arrays


def _native_paths(binding):
    """Keep original manifest untouched; resolve through authenticated binder."""
    original = binding.get('native_manifest', binding)
    result = {}
    for key in ('teacher_checkpoint', 'q_cache_path'):
        path = Path(binding.get(key, original[key]))
        if not path.is_absolute():
            raise ValueError('Mixed preparation needs binder-resolved original artifact paths')
        result[key] = str(path.resolve(strict=True))
    return result


def _native_inputs(binding, dataset, *, production=True):
    reference = binding.get('native_manifest', binding)
    paths = _native_paths(binding)
    n = dataset.base_count
    if production and (n != 19809 or reference.get('sensor') != 'GF2'
            or reference.get('server') != 's5' or reference.get('teacher_update') != 100000):
        raise ValueError('Mixed reference is restricted to local s5 GF2 R5_100')
    for path_key, hash_key in (('teacher_checkpoint', 'teacher_checkpoint_sha256'),
                              ('q_cache_path', 'q_cache_sha256')):
        if sha256_file(paths[path_key]) != reference[hash_key]:
            raise ValueError(f'Native reference changed: {path_key}')
    if (reference.get('train_sha256') != sha256_file(dataset.raw_h5_path)
            or reference.get('train_lpan_sha256') != sha256_file(dataset.lp_path)):
        raise ValueError('Mixed calibration does not match original train/LP reference')
    with np.load(paths['q_cache_path'], allow_pickle=False) as original:
        indices, q = original['calibration_indices'].copy(), original['q'].copy()
    if (not np.array_equal(indices, select_calibration_indices(n))
            or canonical_sha(indices.tolist()) != reference['calibration_indices_sha256']
            or q.dtype != np.float32 or q.shape != (n, 4) or not np.isfinite(q).all() or (q < 0).any()
            or float(np.median(q[indices])) != reference['q_ref']
            or not np.isfinite(reference['tau_R']) or reference['tau_R'] < 1e-6
            or reference['q_ref'] <= 0):
        raise ValueError('Original reference calibration/q identity invalid')
    return reference, indices, q


def prepare_mixed_reference(model, dataset, native_reference, output_dir, *,
        device='cuda', deadline_utc=None, source_identity=None):
    """Prepare once per local R5_100, shared by both parents and fresh arms."""
    from safetensors.torch import load_file
    reference, indices, native_q = _native_inputs(native_reference, dataset)
    paths = _native_paths(native_reference)
    if state_hash(model.state_dict()) != state_hash(load_file(paths['teacher_checkpoint'], device='cpu')):
        raise ValueError('Mixed calibration Teacher differs from native reference weights')
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / 'mixed_reference_manifest.json'
    if destination.exists():
        load_mixed_reference(destination, native_reference, dataset)
        return destination
    aug_path = prepare_augmentation(dataset, folder, deadline_utc)
    adapter = GFB20Dataset(dataset, aug_path, profile='PANMIX')
    binding = dict(native_reference_sha256=canonical_sha(reference),
        source_identity=source_identity,
        augmentation_manifest_sha256=sha256_file(aug_path),
        numerical_source_sha256={name: sha256_file(Path(__file__).parent / name)
            for name in ('augmentation.py', 'data.py', 'calibration.py')})
    calibration, arrays = compute_mixed_calibration(model, adapter, indices, native_q,
        device=device, deadline_utc=deadline_utc,
        progress_path=folder / 'mixed_calibration.progress.h5', identity=binding)
    _native_inputs(native_reference, dataset)  # Refuse source changes during long preparation.
    if any(sha256_file(Path(__file__).parent / name) != value
           for name, value in binding['numerical_source_sha256'].items()):
        raise ValueError('Mixed numerical source changed during calibration')
    q_path, cal_path = folder / 'mixed_q_cache.npz', folder / 'mixed_calibration.json'
    _write_npz_immutable(q_path, arrays)
    calibration.update(binding)
    write_immutable_json(cal_path, calibration)
    manifest = dict(schema='GFB20_MIXED_REFERENCE_v1', sensor='GF2', server='s5',
        native_reference_id=reference['reference_id'], native_reference=reference,
        native_artifact_paths=paths,
        tau_R=calibration['tau_R'], q_ref=calibration['q_ref'],
        native_tau_R=reference['tau_R'], native_q_ref=reference['q_ref'],
        teacher_checkpoint_sha256=reference['teacher_checkpoint_sha256'],
        q_cache_path=str(q_path), q_cache_sha256=sha256_file(q_path),
        calibration_path=str(cal_path), calibration_sha256=sha256_file(cal_path),
        augmentation_manifest_path=str(aug_path), augmentation_manifest_sha256=sha256_file(aug_path),
        calibration_indices_sha256=calibration['calibration_indices_sha256'],
        q_shape=calibration['q_shape'], native_slice_parity=calibration['native_slice_parity'],
        runtime=calibration['runtime'], source_identity=source_identity)
    check_deadline(deadline_utc)
    write_immutable_json(destination, manifest)
    load_mixed_reference(destination, native_reference, dataset)
    return destination


def load_mixed_reference(path, native_reference=None, dataset=None):
    manifest = json.loads(Path(path).read_text())
    if (manifest.get('schema') != 'GFB20_MIXED_REFERENCE_v1'
            or manifest.get('server') != 's5' or manifest.get('sensor') != 'GF2'):
        raise ValueError('Not a GF2 s5 mixed reference')
    reference = manifest['native_reference']
    paths = manifest['native_artifact_paths']
    if native_reference is not None:
        if (native_reference.get('native_manifest', native_reference) != reference
                or _native_paths(native_reference) != paths):
            raise ValueError('Mixed reference points at another native Teacher/reference')
    for pk, sk in (('q_cache_path', 'q_cache_sha256'), ('calibration_path', 'calibration_sha256'),
                   ('augmentation_manifest_path', 'augmentation_manifest_sha256')):
        if sha256_file(manifest[pk]) != manifest[sk]:
            raise ValueError(f'Mixed asset was modified: {pk}')
    with np.load(manifest['q_cache_path'], allow_pickle=False) as src:
        q, indices, online = src['q'].copy(), src['calibration_indices'].copy(), src['native_online_q'].copy()
    n = q.shape[0]
    if (q.shape != (n, 4, 3) or n != 19809 or list(q.shape) != manifest['q_shape']
            or not np.isfinite(q).all() or (q < 0).any()
            or not np.array_equal(indices, select_calibration_indices(n))
            or canonical_sha(indices.tolist()) != manifest['calibration_indices_sha256']
            or weighted_median_tokens(q[indices], 2) != manifest['q_ref']
            or manifest['q_ref'] <= 0 or not np.isfinite(manifest['tau_R']) or manifest['tau_R'] < 1e-6):
        raise ValueError('Mixed q shape/weighted calibration/scale mismatch')
    for path_key, sha_key in (('teacher_checkpoint', 'teacher_checkpoint_sha256'),
                              ('q_cache_path', 'q_cache_sha256')):
        if sha256_file(paths[path_key]) != reference[sha_key]:
            raise ValueError('Original Teacher/native q changed')
    with np.load(paths['q_cache_path'], allow_pickle=False) as src:
        native_q = src['q']
        if not np.array_equal(q[:, :, 1], native_q):
            raise ValueError('Mixed gamma1 slice differs from original native q')
        if verify_native_slice(online, native_q) != manifest['native_slice_parity']:
            raise ValueError('Mixed native parity receipt mismatch')
    calibration = json.loads(Path(manifest['calibration_path']).read_text())
    for key in ('tau_R', 'q_ref', 'q_shape', 'calibration_indices_sha256', 'native_slice_parity', 'runtime'):
        if calibration.get(key) != manifest[key]:
            raise ValueError(f'Mixed calibration manifest disagrees on {key}')
    if calibration.get('synthetic_test') or calibration.get('calibration_n') != 3072:
        raise ValueError('Synthetic/incomplete calibration cannot enter production')
    expected_sources = {name: sha256_file(Path(__file__).parent / name)
                        for name in ('augmentation.py', 'data.py', 'calibration.py')}
    if calibration.get('numerical_source_sha256') != expected_sources:
        raise ValueError('Mixed calibration numerical source differs from current implementation')
    if dataset is not None:
        _native_inputs(dict(native_manifest=reference, **paths), dataset)
        verify_augmentation_cache(manifest['augmentation_manifest_path'], dataset, full=False)
    return manifest, q
