"""Append-only, train-only GF2 PAN/LP gamma shards.

The bank is distribution-independent: changing token weights never regenerates
pixels, and adding a gamma publishes a new manifest, not a mutable old bank.
"""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import h5py
import numpy as np

from fh12.data import (RECIPE, AUGMENTATION, canonical_sha, sha256_file,
                       check_deadline, write_immutable_json)
from tools.repair_lpan import make_lpan

GAMMA_UNION = (.5, .75, .875, 1., 1.125, 1.25, 1.5)
VIEW_RECIPE = dict(schema='GFP40_PAN_DETAIL_GAIN_v1', sigma=1., kernel_size=7,
    padding='reflect_no_repeated_edge', separable=True, generation_dtype='float64_DN',
    cache_dtype='float32_DN', gamma1_bypass=True, clamp=False, LP_recipe=RECIPE,
    geometry=AUGMENTATION, order='source PAN -> gain -> LP -> fixed HV/ROT4 -> warp',
    train_only=True)


def gamma_key(gamma):
    if float(gamma) not in GAMMA_UNION:
        raise ValueError('Gamma is outside the registered seven-slice union')
    return f'g{round(float(gamma) * 1000):04d}'


def gaussian_detail_view(pan, gamma):
    """Gamma is the physical gain, not a family-local index. No clamp/round."""
    gamma_key(gamma)
    if float(gamma) == 1.:
        return pan
    x = np.asarray(pan)
    if (x.dtype != np.float64 or x.ndim != 4 or x.shape[1] != 1
            or min(x.shape[-2:]) <= 3 or not np.isfinite(x).all()):
        raise ValueError('Gain generation requires finite original float64 DN N1HW')
    axis = np.arange(-3, 4, dtype=np.float64)
    kernel = np.exp(-axis * axis / 2); kernel /= kernel.sum()
    blur = np.stack([cv2.sepFilter2D(p[0], -1, kernel, kernel,
        borderType=cv2.BORDER_REFLECT_101)[None] for p in x])
    return blur + float(gamma) * (x - blur)


def _source_contract(dataset):
    if (dataset.split != 'train' or not dataset.has_gt or dataset.bands != 4
            or dataset.max_pixel != 1023 or len(dataset) != dataset.base_count):
        raise ValueError('PAN views require the GF2 base-ID train dataset only')
    source, lp = sha256_file(dataset.raw_h5_path), sha256_file(dataset.lp_path)
    if getattr(dataset, 'source_hashes', {}) != dict(source=source, lp=lp):
        raise ValueError('Native dataset source/LP changed after loading')
    # Content identity is portable; original absolute location is not identity.
    return dict(source_sha256=source, native_lp_sha256=lp, count=dataset.base_count,
        recipe=VIEW_RECIPE, opencv_version=cv2.__version__, numpy_version=np.__version__,
        gain_source_sha256=sha256_file(__file__))


def _expected_chunk(source, lp, start, stop, gamma):
    original = np.asarray(source['pan'][start:stop], dtype=np.float64)
    pan = gaussian_detail_view(original, gamma)
    low = np.asarray(lp['lpan'][start:stop]) if gamma == 1 else make_lpan(pan)
    return pan.astype(np.float32), low.astype(np.float32)


def _verify_shard(shard, dataset, deadline_utc=None, *, full=False):
    if (shard.get('schema') != 'GFP40_PAN_SHARD_v1' or not shard.get('complete')
            or shard.get('source') != _source_contract(dataset)):
        raise ValueError('PAN shard source/recipe identity mismatch')
    gamma_key(shard['gamma'])
    path = Path(shard['cache_path'])
    if sha256_file(path) != shard['cache_sha256']:
        raise ValueError('PAN shard SHA mismatch')
    n = dataset.base_count
    with h5py.File(path, 'r') as cache:
        if (cache['pan'].shape != (n, 1, 64, 64) or cache['lpan'].shape != (n, 1, 16, 16)
                or any(cache[k].dtype != np.dtype('float32') for k in ('pan', 'lpan'))):
            raise ValueError('PAN shard incomplete population/shape/dtype')
        if full:
            with h5py.File(dataset.raw_h5_path, 'r') as src, h5py.File(dataset.lp_path, 'r') as lp:
                for start in range(0, n, 64):
                    check_deadline(deadline_utc)
                    expected = _expected_chunk(src, lp, start, min(n, start+64), shard['gamma'])
                    if any(not np.array_equal(cache[k][start:start+64], val)
                           for k, val in zip(('pan', 'lpan'), expected)):
                        raise ValueError('PAN/LP shard disagrees with original DN recipe')
    return shard


def prepare_gamma_shard(dataset, output_dir, gamma, deadline_utc=None, *, reuse_manifest=None):
    """Resume chunk-wise, verify all pixels, publish immutable single-gamma file."""
    gamma = float(gamma); key = gamma_key(gamma)
    folder = Path(output_dir).resolve() / 'pan_shards' / key
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / 'manifest.json'
    if destination.exists():
        _verify_shard(json.loads(destination.read_text()), dataset, deadline_utc)
        return destination
    source = _source_contract(dataset); n = dataset.base_count
    reused = None
    if reuse_manifest is not None and gamma in (.75, 1., 1.25):
        old = json.loads(Path(reuse_manifest).read_text())
        if (old.get('schema') != 'GFB20_AUGMENTATION_v1' or not old.get('complete')
                or not old.get('native_bitwise') or old.get('population') != n
                or old['source']['source_sha256'] != source['source_sha256']
                or old['source']['native_lp_sha256'] != source['native_lp_sha256']
                or any(old['source']['recipe'].get(k) != v for k,v in VIEW_RECIPE.items() if k != 'schema')
                or sha256_file(old['cache_path']) != old['cache_sha256']):
            raise ValueError('B20 PAN/LP cache identity mismatch')
        reused = dict(status='REUSED_EXTERNAL', original_manifest=str(Path(reuse_manifest).resolve()),
            original_manifest_sha256=sha256_file(reuse_manifest), cache_path=old['cache_path'],
            cache_sha256=old['cache_sha256'], historical_cost_not_recharged=True)
    partial, final = folder / 'views.partial.h5', folder / 'views.h5'
    identity = canonical_sha(dict(source=source, gamma=gamma))
    if not final.exists():
        with h5py.File(partial, 'a') as dst:
            if 'identity_sha256' not in dst.attrs:
                dst.attrs['identity_sha256'] = identity
                dst.create_dataset('pan', (n, 1, 64, 64), dtype='float32')
                dst.create_dataset('lpan', (n, 1, 16, 16), dtype='float32')
                dst.create_dataset('done', ((n+63)//64,), dtype='bool'); dst.flush()
            if dst.attrs['identity_sha256'] != identity:
                raise ValueError('Cannot resume gamma shard with different source/recipe')
            with h5py.File(dataset.raw_h5_path, 'r') as src, h5py.File(dataset.lp_path, 'r') as lp, \
                    (h5py.File(reused['cache_path'], 'r') if reused else nullcontext()) as external:
                for chunk, start in enumerate(range(0, n, 64)):
                    check_deadline(deadline_utc)
                    if external is None:
                        values = _expected_chunk(src, lp, start, min(n, start+64), gamma)
                    else:
                        gid = (.75, 1., 1.25).index(gamma)
                        if (external['pan'].shape != (n, 3, 1, 64, 64)
                                or external['lpan'].shape != (n, 3, 1, 16, 16)):
                            raise ValueError('B20 PAN/LP bank incomplete')
                        values = tuple(external[k][start:start+64, gid] for k in ('pan', 'lpan'))
                    for name, val in zip(('pan', 'lpan'), values):
                        if dst['done'][chunk]:
                            if not np.array_equal(dst[name][start:start+64], val):
                                raise ValueError('Previously completed PAN shard chunk changed')
                        else:
                            dst[name][start:start+64] = val
                    dst.flush(); dst['done'][chunk] = True; dst.flush()
            if not dst['done'][:].all():
                raise ValueError('Incomplete gamma shard')
        os.chmod(partial, 0o444); os.link(partial, final)
    manifest = dict(schema='GFP40_PAN_SHARD_v1', complete=True, source=source, gamma=gamma,
        cache_path=str(final), cache_sha256=sha256_file(final), native_bitwise=gamma == 1.,
        reuse_provenance=reused)
    _verify_shard(manifest, dataset, deadline_utc, full=True)
    write_immutable_json(destination, manifest)
    return destination


def prepare_augmentation(dataset, output_dir, gammas=(.75, 1., 1.25), deadline_utc=None, *, reuse_manifest=None):
    gammas = [float(g) for g in gammas]
    if len(set(gammas)) != len(gammas) or gammas != sorted(gammas) or 1. not in gammas:
        raise ValueError('Gamma bank requires distinct ordered support including native')
    shards = []
    for gamma in gammas:
        path = prepare_gamma_shard(dataset, output_dir, gamma, deadline_utc, reuse_manifest=reuse_manifest)
        shards.append(dict(path=str(path), sha256=sha256_file(path),
                           **json.loads(path.read_text())))
    manifest = dict(schema='GFP40_AUGMENTATION_v1', complete=True,
        source=_source_contract(dataset), gammas=gammas, shards=shards,
        population=dataset.base_count, gamma_bank_sha256=canonical_sha(
            [dict(gamma=s['gamma'], cache_sha256=s['cache_sha256']) for s in shards]))
    destination = Path(output_dir).resolve() / ('augmentation_' + canonical_sha(gammas)[:16] + '.json')
    write_immutable_json(destination, manifest)
    return destination


def verify_augmentation_cache(manifest, dataset, deadline_utc=None, *, full=False):
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    if (manifest.get('schema') != 'GFP40_AUGMENTATION_v1' or not manifest.get('complete')
            or manifest.get('source') != _source_contract(dataset)
            or manifest.get('population') != dataset.base_count):
        raise ValueError('Augmentation bank source/protocol/identity mismatch')
    if ([s['gamma'] for s in manifest['shards']] != manifest['gammas']
            or manifest['gammas'] != sorted(set(manifest['gammas']))
            or 1. not in manifest['gammas']):
        raise ValueError('Augmentation support invalid')
    for shard in manifest['shards']:
        if sha256_file(shard['path']) != shard['sha256']:
            raise ValueError('PAN shard manifest SHA mismatch')
        original = json.loads(Path(shard['path']).read_text())
        if original != {k:v for k,v in shard.items() if k not in ('path', 'sha256')}:
            raise ValueError('PAN shard reference mismatch')
        _verify_shard(original, dataset, deadline_utc, full=full)
    if manifest['gamma_bank_sha256'] != canonical_sha(
            [dict(gamma=s['gamma'], cache_sha256=s['cache_sha256']) for s in manifest['shards']]):
        raise ValueError('PAN bank identity mismatch')
    return manifest
