"""Read-only exact-DN crop audit; never establishes an acquisition identity.

Search one eight-band uint16 MS GeoTIFF for every NCHW H5 patch under all
eight D4 orientations. Candidate hashes are only an accelerator: admission
always requires np.array_equal over the complete, same-band-order crop.
Neither inputs nor experiment manifests are modified. A negative result is
limited to this tile, direct DN, integer crops and D4; resampling, DN changes,
band permutations, other SpaceNet tiles and geographic identity are untested.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import tifffile


SCOPE = {
    'comparison': 'exact_full_crop_array_equal',
    'pixel_values': 'direct_DN_no_normalization_no_tolerance',
    'band_order': 'unchanged_eight_bands',
    'geometry': 'integer_MS_pixel_crop_and_all_8_D4_orientations',
    'coordinates': 'zero_based_y_x_in_source_MS_grid',
    'not_tested': ['other_source_tiles', 'resampling', 'radiometric_changes',
                   'band_permutations', 'geographic_identity',
                   'native_PAN_correspondence', 'acquisition_lineage'],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def d4(patch, k, flip):
    """Patch-to-source orientation: horizontal flip first, then k CCW turns."""
    return np.rot90(patch[:, :, ::-1] if flip else patch, k, axes=(-2, -1))


def vector_hash(values):
    """FNV-like uint64 hash of eight uint16 band values; collisions are safe."""
    values = np.asarray(values)
    if values.shape[0] != 8:
        raise ValueError('Exactly eight bands are required')
    out = np.full(values.shape[1:], np.uint64(14695981039346656037), dtype=np.uint64)
    with np.errstate(over='ignore'):
        for band in values:
            out = (out ^ band.astype(np.uint64)) * np.uint64(1099511628211)
    return out


def anchor(patch):
    """Select one reproducible interior pixel, favoring spectrally varied DN."""
    h, w = patch.shape[-2:]
    options = [(y, x) for y in sorted({h // 4, h // 2, 3 * h // 4})
               for x in sorted({w // 4, w // 2, 3 * w // 4})]
    return max(options, key=lambda p: (float(np.ptp(patch[:, p[0], p[1]])),
                                      float(np.sum(patch[:, p[0], p[1]]))))


def compatible(patch):
    return (np.isfinite(patch).all() and np.all(patch >= 0)
            and np.all(patch <= 65535) and np.array_equal(patch, np.floor(patch)))


def query_hashes(patch):
    hashes = set()
    for flip in (False, True):
        for k in range(4):
            view = d4(patch, k, flip)
            y, x = anchor(view)
            hashes.add(int(vector_hash(view[:, y, x])))
    return hashes


def index_candidates(source, queries, block_rows=128, max_candidates=4096):
    """Bounded scan, avoiding a Python dictionary for every source pixel.

    The dictionary contains only requested hashes; stored coordinates per hash
    are capped. Overflows are returned explicitly and can never imply absence.
    """
    if source.ndim != 3 or source.shape[0] != 8 or source.dtype != np.uint16:
        raise ValueError('Source must be CHW eight-band uint16')
    if block_rows <= 0 or max_candidates <= 0:
        raise ValueError('Scan bounds must be positive')
    keys = np.array(sorted(queries), dtype=np.uint64)
    result = {int(k): [] for k in keys}
    counts = Counter()
    if not len(keys):
        return result, counts
    for y0 in range(0, source.shape[1], block_rows):
        block = vector_hash(source[:, y0:y0 + block_rows]).reshape(-1)
        positions = np.searchsorted(keys, block)
        valid = positions < len(keys)
        found = np.flatnonzero(valid)
        found = found[keys[positions[found]] == block[found]]
        if not len(found):
            continue
        # Group only matched pixels; large constant regions need no Python loop
        # over millions of source coordinates.
        matched_keys = block[found]
        order = np.argsort(matched_keys, kind='stable')
        ordered_keys = matched_keys[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_keys)) + 1, len(order)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            key = int(ordered_keys[start])
            counts[key] += int(stop - start)
            room = max_candidates - len(result[key])
            if room > 0:
                selected = found[order[start:min(stop, start + room)]]
                result[key].extend((int(y0 + p // source.shape[2]),
                                    int(p % source.shape[2])) for p in selected)
    return result, counts


def match_patch(source, patch, candidates, candidate_counts, max_candidates=4096,
                max_matches=256):
    if patch.ndim != 3 or patch.shape[0] != 8:
        raise ValueError('Patch must be eight-band CHW')
    if not compatible(patch):
        return {'status': 'NO_EXACT_MATCH_UINT16_DN_INCOMPATIBLE', 'matches': [],
                'search_complete': True, 'matches_truncated': False,
                'anchor_candidates_truncated': False}
    matches = []
    checked = set()
    candidate_overflow = False
    matches_truncated = False
    anchor_vector_hits = 0
    for flip in (False, True):
        for k in range(4):
            view = d4(patch, k, flip)
            ay, ax = anchor(view)
            key = int(vector_hash(view[:, ay, ax]))
            candidate_overflow |= candidate_counts.get(key, 0) > max_candidates
            for sy, sx in candidates.get(key, []):
                if not np.array_equal(source[:, sy, sx], view[:, ay, ax]):
                    continue
                anchor_vector_hits += 1
                y, x = sy - ay, sx - ax
                h, w = view.shape[-2:]
                if min(y, x) < 0 or y + h > source.shape[1] or x + w > source.shape[2]:
                    continue
                identity = (y, x, k, flip)
                if identity in checked:
                    continue
                checked.add(identity)
                if np.array_equal(source[:, y:y + h, x:x + w], view):
                    if len(matches) < max_matches:
                        matches.append({'source_y': y, 'source_x': x,
                                        'height': h, 'width': w,
                                        'patch_to_source_flip_x': flip,
                                        'patch_to_source_rot90_ccw': k})
                    else:
                        matches_truncated = True
    complete = not candidate_overflow and not matches_truncated
    return {'status': ('EXACT_MATCH' if matches else
                       'INCOMPLETE_NO_MATCH_FOUND' if not complete else 'NO_EXACT_MATCH'),
            'matches': matches, 'search_complete': complete,
            'matches_truncated': matches_truncated,
            'anchor_candidates_truncated': candidate_overflow,
            'anchor_vector_hits': anchor_vector_hits,
            'full_crop_comparisons': len(checked)}


def read_tiff(path, max_pixels=32_000_000, max_bytes=600_000_000):
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            raise ValueError('Exactly one TIFF image series is required')
        series = tif.series[0]
        shape = series.shape
        if len(shape) != 3 or (shape[0] == 8) == (shape[-1] == 8):
            raise ValueError('Unambiguous CHW or HWC eight-band TIFF required')
        if series.dtype != np.uint16:
            raise ValueError('Audit expects raw uint16 DN TIFF')
        if np.prod(shape) // 8 > max_pixels or np.prod(shape) * 2 > max_bytes:
            raise ValueError('TIFF exceeds explicit in-memory source bound')
        image = series.asarray()
        metadata = {'stored_shape': list(shape), 'axes': series.axes,
                    'dtype': str(series.dtype)}
    return (image if shape[0] == 8 else image.transpose(2, 0, 1)), metadata


def default_datasets(root):
    root = Path(root)
    return [('train', root / 'train_wv3.h5', 'gt'),
            ('valid', root / 'valid_wv3.h5', 'gt'),
            ('fr20', root / 'full_examples_mat20/test_wv3_OrigScale_mat20.h5', 'ms'),
            ('rr20', root / 'reduced_examples_h5/test_wv3_multiExm1.h5', 'gt')]


def audit(tiff, datasets, *, block_rows=128, max_candidates=4096, max_matches=256,
          max_source_pixels=32_000_000):
    tiff = Path(tiff).resolve(strict=True)
    initial = {tiff: stamp(tiff)}
    source, tiff_metadata = read_tiff(tiff, max_source_pixels)
    source_info = {'path': str(tiff), 'sha256': sha256(tiff), **tiff_metadata,
                   'chw_shape': list(source.shape),
                   'per_band_min': source.min(axis=(1, 2)).tolist(),
                   'per_band_max': source.max(axis=(1, 2)).tolist()}
    queries = set()
    dataset_infos = []
    for label, path, key in datasets:
        path = Path(path).resolve(strict=True)
        initial[path] = stamp(path)
        with h5py.File(path, 'r') as h:
            data = h[key]
            if data.ndim != 4 or data.shape[1] != 8:
                raise ValueError('H5 datasets must be NCHW with eight bands')
            low, high = np.inf, -np.inf
            incompatible_rows = []
            for row in range(len(data)):
                patch = data[row]
                if np.isfinite(patch).all():
                    low, high = min(low, float(patch.min())), max(high, float(patch.max()))
                if compatible(patch):
                    queries.update(query_hashes(patch))
                else:
                    incompatible_rows.append(row)
            dataset_infos.append({'label': label, 'path': str(path), 'key': key,
                                  'sha256': sha256(path), 'shape': list(data.shape),
                                  'dtype': str(data.dtype), 'dn_min': low if np.isfinite(low) else None,
                                  'dn_max': high if np.isfinite(high) else None,
                                  'uint16_incompatible_rows': incompatible_rows})
        print(f'Anchors collected: {label} ({data.shape[0]} rows)', file=sys.stderr)
    candidates, counts = index_candidates(source, queries, block_rows, max_candidates)
    results = []
    totals = Counter()
    for info in dataset_infos:
        rows = []
        with h5py.File(info['path'], 'r') as h:
            for row in range(h[info['key']].shape[0]):
                result = match_patch(source, h[info['key']][row], candidates, counts,
                                     max_candidates, max_matches)
                result['row'] = row
                rows.append(result)
                totals[result['status']] += 1
        results.append({**info, 'counts': dict(Counter(r['status'] for r in rows)), 'rows': rows})
    for path, before in initial.items():
        if stamp(path) != before:
            raise RuntimeError('Input changed during audit: ' + str(path))
    return {'schema': 'SPACENET_PANCOLLECTION_DIRECT_DN_AUDIT_V1',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': SCOPE,
            'source': source_info, 'summary': dict(totals), 'datasets': results,
            'limits': {'source_pixel_limit': max_source_pixels, 'scan_block_rows': block_rows,
                       'max_source_anchor_candidates_per_hash': max_candidates,
                       'max_matches_per_patch_including_orientations': max_matches},
            'anchor_index': {'requested_unique_hashes': len(queries),
                             'source_hash_hits': sum(counts.values()),
                             'overflowed_hashes': sum(v > max_candidates for v in counts.values())},
            'software': {'numpy': np.__version__, 'h5py': h5py.__version__,
                         'tifffile': tifffile.__version__},
            'interpretation': 'Only direct-DN integer crop/D4 matches in this one tile were tested. '
                              'No exact match does not exclude resampling or other tiles. '
                              'A positive is a pixel correspondence candidate, not acquisition proof.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiff', required=True)
    parser.add_argument('--wv3-root', default='data/PanCollection/WV3')
    parser.add_argument('--output', required=True, help='New JSON path; existing files are never overwritten')
    parser.add_argument('--block-rows', type=int, default=128)
    parser.add_argument('--max-candidates', type=int, default=4096)
    parser.add_argument('--max-matches', type=int, default=256)
    parser.add_argument('--max-source-pixels', type=int, default=32_000_000)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('Output already exists')
    if min(args.max_candidates, args.max_matches, args.block_rows, args.max_source_pixels) <= 0:
        parser.error('All resource limits must be positive')
    result = audit(args.tiff, default_datasets(args.wv3_root), block_rows=args.block_rows,
                   max_candidates=args.max_candidates, max_matches=args.max_matches,
                   max_source_pixels=args.max_source_pixels)
    with open(args.output, 'x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result['summary'], sort_keys=True))


if __name__ == '__main__':
    main()
