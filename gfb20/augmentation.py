"""Train-only, source-DN PAN detail views, with resumable immutable caches.

Reflect means whole-sample reflection (no repeated edge), as in torch reflect
and numpy.pad(mode='reflect'); OpenCV names this BORDER_REFLECT_101.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

from fh12.data import (RECIPE, AUGMENTATION, canonical_sha, sha256_file,
                       check_deadline, write_immutable_json)
from tools.repair_lpan import make_lpan

GAMMAS = (0.75, 1.0, 1.25)
PROBABILITIES = (0.25, 0.5, 0.25)
MEDIAN_TOKENS = (0, 1, 1, 2)
VIEW_RECIPE = dict(schema='GFB20_PAN_DETAIL_GAIN_v1', gammas=list(GAMMAS),
    probabilities=list(PROBABILITIES), median_tokens=list(MEDIAN_TOKENS),
    sigma=1.0, kernel_size=7, padding='reflect_no_repeated_edge',
    separable=True, generation_dtype='float64_DN', cache_dtype='float32_DN',
    gamma1_bypass=True, clamp=False, LP_recipe=RECIPE, geometry=AUGMENTATION,
    order='source PAN -> gain -> LP -> fixed HV/ROT4 -> warp', train_only=True)


def gaussian_detail_view(pan, gamma_id):
    """No clipping/rounding; gamma1 returns the *same object*, untouched."""
    if gamma_id not in range(3):
        raise ValueError('gamma_id must be 0, 1, or 2')
    if gamma_id == 1:
        return pan
    x = np.asarray(pan)
    if x.dtype != np.float64 or x.ndim != 4 or x.shape[1] != 1:
        raise ValueError('Gain generation requires original float64 DN N1HW')
    if min(x.shape[-2:]) <= 3 or not np.isfinite(x).all():
        raise ValueError('Invalid PAN source for reflect k7')
    axis = np.arange(-3, 4, dtype=np.float64)
    kernel = np.exp(-axis * axis / 2)
    kernel /= kernel.sum()
    blur = np.stack([cv2.sepFilter2D(p[0], -1, kernel, kernel,
        borderType=cv2.BORDER_REFLECT_101)[None] for p in x])
    return blur + GAMMAS[gamma_id] * (x - blur)


def _source_contract(dataset):
    if (dataset.split != 'train' or not dataset.has_gt or dataset.bands != 4
            or dataset.max_pixel != 1023 or len(dataset) != dataset.base_count):
        raise ValueError('PAN views require the GF2 base-ID train dataset only')
    source_sha, lp_sha = sha256_file(dataset.raw_h5_path), sha256_file(dataset.lp_path)
    if getattr(dataset, 'source_hashes', {}) != dict(source=source_sha, lp=lp_sha):
        raise ValueError('Native dataset source/LP changed after its arrays were loaded')
    return dict(source_path=str(Path(dataset.raw_h5_path).resolve()),
        source_sha256=source_sha,
        native_lp_path=str(Path(dataset.lp_path).resolve()),
        native_lp_sha256=lp_sha, count=dataset.base_count,
        dataset_sha256=getattr(dataset, 'contract_hash', None), recipe=VIEW_RECIPE,
        opencv_version=cv2.__version__, numpy_version=np.__version__,
        gain_source_sha256=sha256_file(__file__))


def _expected_chunk(source, native_lp, start, stop):
    original = np.asarray(source['pan'][start:stop], dtype=np.float64)
    pans, lps = [], []
    for gamma in range(3):
        viewed = gaussian_detail_view(original, gamma)
        pans.append(viewed.astype(np.float32))
        lps.append(np.asarray(native_lp['lpan'][start:stop]) if gamma == 1
                   else make_lpan(viewed).astype(np.float32))
    return np.stack(pans, axis=1), np.stack(lps, axis=1)


def verify_augmentation_cache(manifest, dataset, deadline_utc=None, *, full=True):
    """Consumer SHA checks; publication additionally verifies every raw sample."""
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    if (manifest.get('schema') != 'GFB20_AUGMENTATION_v1'
            or manifest.get('source') != _source_contract(dataset)
            or not manifest.get('complete') or not manifest.get('native_bitwise')):
        raise ValueError('Augmentation source/protocol/identity mismatch')
    path = Path(manifest['cache_path'])
    if sha256_file(path) != manifest['cache_sha256']:
        raise ValueError('Augmentation cache SHA mismatch')
    n = dataset.base_count
    with h5py.File(path, 'r') as cache:
        if (cache['pan'].shape != (n, 3, 1, 64, 64)
                or cache['lpan'].shape != (n, 3, 1, 16, 16)
                or cache['pan'].dtype != np.dtype('float32')
                or cache['lpan'].dtype != np.dtype('float32')):
            raise ValueError('Incomplete/incorrect train augmentation cache')
        if full:
            with h5py.File(dataset.raw_h5_path, 'r') as src, h5py.File(dataset.lp_path, 'r') as lp:
                for start in range(0, n, 64):
                    check_deadline(deadline_utc)
                    pans, lps = _expected_chunk(src, lp, start, min(n, start + 64))
                    if (not np.array_equal(cache['pan'][start:start+64], pans)
                            or not np.array_equal(cache['lpan'][start:start+64], lps)):
                        raise ValueError(f'Augmented PAN/LP correspondence failed at {start}')
    return manifest


def prepare_augmentation(dataset, output_dir, deadline_utc=None):
    """Resume chunk writes, then reverify all samples before publication.

    Partial files are not consumable manifests. Existing source/native LP is
    never changed; deadline pauses retain verified-prefix progress.
    """
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / 'augmentation_manifest.json'
    if destination.exists():
        verify_augmentation_cache(destination, dataset, deadline_utc, full=False)
        return destination
    contract = _source_contract(dataset)
    n = dataset.base_count
    partial, final = folder / 'pan_views.partial.h5', folder / 'pan_views.h5'
    if not final.exists():
        with h5py.File(partial, 'a') as dst:
            identity = canonical_sha(contract)
            if 'contract_sha256' not in dst.attrs:
                dst.attrs['contract_sha256'] = identity
                dst.create_dataset('pan', (n, 3, 1, 64, 64), dtype='float32')
                dst.create_dataset('lpan', (n, 3, 1, 16, 16), dtype='float32')
                dst.create_dataset('done', ((n + 63) // 64,), dtype='bool')
                dst.flush()
            if dst.attrs['contract_sha256'] != identity:
                raise ValueError('Cannot resume augmentation under another source/runtime')
            with h5py.File(dataset.raw_h5_path, 'r') as src, h5py.File(dataset.lp_path, 'r') as lp:
                for chunk, start in enumerate(range(0, n, 64)):
                    check_deadline(deadline_utc)
                    pans, lps = _expected_chunk(src, lp, start, min(n, start + 64))
                    if dst['done'][chunk]:
                        if (not np.array_equal(dst['pan'][start:start+64], pans)
                                or not np.array_equal(dst['lpan'][start:start+64], lps)):
                            raise ValueError('Previously completed augmentation chunk was changed')
                    else:
                        dst['pan'][start:start+64] = pans
                        dst['lpan'][start:start+64] = lps
                        dst.flush()
                        dst['done'][chunk] = True
                        dst.flush()
            if not np.asarray(dst['done']).all():
                raise ValueError('Augmentation cache is incomplete')
        os.chmod(partial, 0o444)
        os.link(partial, final)
    manifest = dict(schema='GFB20_AUGMENTATION_v1', source=contract,
        cache_path=str(final), cache_sha256=sha256_file(final), complete=True,
        native_bitwise=True, population=n, shape_pan=[n, 3, 1, 64, 64],
        shape_lpan=[n, 3, 1, 16, 16])
    verify_augmentation_cache(manifest, dataset, deadline_utc, full=True)
    write_immutable_json(destination, manifest)
    return destination
