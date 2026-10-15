"""Server-side v3 scene verification, without training or publishing metrics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import h5py
import numpy as np

from .common import atomic_json, canonical_sha, file_sha, load_json


def scan_array(dataset, *, binary=False, row_chunk=256):
    """Stream all pixels; never materialize a scene or all eight bands."""
    if dataset.ndim != 4 or dataset.shape[0] != 1 or min(dataset.shape) <= 0:
        raise ValueError('Expected single-scene NCHW storage')
    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError('row_chunk must be a positive integer')
    stats = []
    for channel in range(dataset.shape[1]):
        minimum, maximum, count, zero, above = float('inf'), float('-inf'), 0, 0, 0
        for row in range(0, dataset.shape[2], row_chunk):
            chunk = dataset[0, channel, row:row + row_chunk, :]
            if not np.isfinite(chunk).all():
                raise ValueError('Nonfinite input: ' + dataset.name)
            low, high = float(chunk.min()), float(chunk.max())
            if binary and (low < 0 or high > 1 or not np.isin(chunk, [0, 1]).all()):
                raise ValueError('Nonbinary validity mask: ' + dataset.name)
            if not binary and low < 0:
                raise ValueError('Negative unnormalized DN: ' + dataset.name)
            minimum, maximum = min(minimum, low), max(maximum, high)
            count += chunk.size
            zero += int(np.count_nonzero(chunk == 0))
            above += int(np.count_nonzero(chunk > 2047))
        stats.append(dict(count=count, minimum=minimum, maximum=maximum,
                          zero_count=zero, above_2047_count=above))
    return dict(shape=list(dataset.shape), dtype=str(dataset.dtype),
                scan='ALL_PIXELS_BOUNDED_ROW_STRIPES', bands=stats)


def verify_split_generation(handle, provenance):
    """Independently replay the declared PCG64 block rule; do not change it."""
    height, width = handle['native/ms'].shape[-2:]
    if (provenance.get('schema') != 'PANPV_SPATIAL_SPLIT_v1'
            or provenance.get('seed') != 261002 or provenance.get('rng') != 'numpy.PCG64'
            or provenance.get('scene_id') != str(handle.attrs['scene_id'])
            or provenance.get('scene_shape_hw') != [height, width]):
        raise ValueError('Unexpected split namespace, seed or scene shape')
    if (provenance['block_size'], provenance['buffer'], provenance['patch_size']) != (1024, 128, 64):
        raise ValueError('Unexpected v3 split recipe')
    blocks, rows, usable = [], [], []
    for row in range(0, height, 1024):
        for col in range(0, width, 1024):
            bh, bw = min(1024, height - row), min(1024, width - col)
            candidates = [(r, c) for r in range(row + 128, row + bh - 191, 64)
                          for c in range(col + 128, col + bw - 191, 64)]
            if candidates:
                usable.append(len(blocks))
            blocks.append([row, col, bh, bw, -1])
            rows.append(candidates)
    namespace = dict(schema='PANPV_SPATIAL_SPLIT_v1', seed=261002,
                     scene_id=str(handle.attrs['scene_id']))
    rng_seed = int(canonical_sha(namespace)[:32], 16)
    permutation = np.random.Generator(np.random.PCG64(rng_seed)).permutation(np.array(usable, dtype=np.int32))
    held = max(1, (len(usable) + 5) // 10)
    for position, block_id in enumerate(permutation):
        blocks[block_id][4] = 0 if position < len(usable) - 2 * held else (1 if position < len(usable) - held else 2)
    candidate = np.array([(r, c, blocks[b][4], b) for b, entries in enumerate(rows)
                          for r, c in entries], dtype='<i4').reshape(-1, 4)
    if not np.array_equal(handle['splits/blocks'][:], blocks):
        raise ValueError('Block assignments differ from frozen PCG64 recipe')
    if not np.array_equal(handle['splits/candidate_patch_index'][:], candidate):
        raise ValueError('Candidate patch list differs from frozen recipe')
    keep = handle['splits/candidate_keep'][:]
    if keep.shape != (len(candidate),) or not np.isin(keep, [0, 1]).all():
        raise ValueError('Invalid candidate keep flags')
    if not np.array_equal(candidate[keep.astype(bool)], handle['splits/patch_index'][:]):
        raise ValueError('Filtered patch list does not match candidate/keep list')
    return dict(replayed=True, candidates=len(candidate), retained=int(keep.sum()),
                namespace_sha256=canonical_sha(namespace),
                candidate_sha256=hashlib.sha256(candidate.tobytes()).hexdigest(),
                keep_sha256=hashlib.sha256(keep.tobytes()).hexdigest(),
                content_filter_recomputed=False)


def verify(args):
    from .v3_data import SceneV3, RRPatchV3
    if args.output.exists():
        raise ValueError('Preserve old verification receipts; choose a new output')
    package = load_json(args.package_receipt)
    if package['state'] != 'TRANSFER_VERIFIED_NOT_TRAINING_ADMISSION':
        raise ValueError('Verified package transfer receipt is required')
    root = Path(package['input_root']).resolve(strict=True)
    for relative in ('canonical_mapping.json', 'dataset_contract.json'):
        if file_sha(root / relative) != package['files'][relative]['sha256']:
            raise ValueError('Package metadata changed after extraction')
    mapping, contract = load_json(root / 'canonical_mapping.json'), load_json(root / 'dataset_contract.json')
    if [v['scene_id'] for v in mapping['scenes']] != contract['selected_scene_order']:
        raise ValueError('Scene order mismatch')
    result = dict(schema='PANPV_V3_SERVER_DATA_VERIFICATION_v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  state='VERIFYING', input_root=str(root),
                  package_receipt_sha256=file_sha(args.package_receipt), scenes=[],
                  formal_training_started=False, campaign_admission=False, sheet_writes=0,
                  dataset_scope='WITHIN_SCENE_ADAPTATION_FILTERED_SPATIAL_HOLDOUT',
                  rr_protocol=contract['rr_profile']['id'],
                  rr_benchmark_parity=False, original_vendor_tiff_pixels_reverified=False,
                  fr_scope='TILE_SCOPE_RETAINED; v3 tile coordinates not yet frozen',
                  remaining=['Exact donor weights/config delivery and strict-load/FT0 parity',
                             'Numerical RR/LP parity receipt',
                             'Real-model tile-context parity, frozen FR tile list and final evaluator',
                             'End-to-end campaign runner and dedicated report integration'])
    atomic_json(args.output, result)
    for entry in mapping['scenes']:
        supplied_path = root / entry['file']
        path = supplied_path.resolve(strict=True)
        if not path.is_relative_to(root) or supplied_path.is_symlink():
            raise ValueError('Scene path leaves verified package')
        sha = package['files'][entry['file']]['sha256']
        if sha != entry['file_sha256']:
            raise ValueError('Canonical and package scene SHA mismatch')
        scene = SceneV3(path, expected_sha256=sha)
        record = dict(scene.summary())
        record['category'] = contract['scene_selection'][entry['scene_id']]['category']
        record['patch_smokes'] = []
        for split in ('train', 'val', 'audit'):
            dataset = RRPatchV3(scene, split=split)
            item = dataset[0]
            record['patch_smokes'].append(dict(split=split, count=len(dataset),
                raw_index=int(item['raw_index']), coords=list(map(int, item['coords'])),
                shapes={k: list(item[k].shape) for k in ('pan', 'ms', 'lpan', 'gt', 'mask')},
                mask_pixels=int(item['mask'].sum())))
        with h5py.File(path, 'r') as handle:
            import json
            provenance = json.loads(handle['metadata/split_provenance_json'][()])
            record['split_replay'] = verify_split_generation(handle, provenance)
            for metadata, expected in [('rr_profile', contract['rr_profile']),
                                       ('lp_profile', contract['lp_profile']),
                                       ('normalization', contract['normalization'])]:
                actual = json.loads(handle[f'metadata/{metadata}_json'][()])
                if actual != expected:
                    raise ValueError(metadata + ' differs from package contract')
            record['array_statistics'] = {}
            for group in ('native', 'rr'):
                for name, value in handle[group].items():
                    if group == 'rr' and name in ('target', 'target_valid'):
                        continue  # exact native hard links already checked by SceneV3
                    record['array_statistics'][f'{group}/{name}'] = scan_array(value, binary='valid' in name)
        scene._check_identity()
        scene.close()
        result['scenes'].append(record)
        atomic_json(args.output, result)
        print('Scene arrays / frozen split verified: ' + entry['scene_id'], flush=True)
    result['state'] = 'DATA_ARRAYS_AND_FROZEN_SPLITS_VERIFIED_NOT_CAMPAIGN_ADMISSION'
    atomic_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
