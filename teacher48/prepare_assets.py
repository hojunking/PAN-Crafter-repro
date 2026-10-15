"""Offline native-scene preparation, not automatic training admission.

Source pixels are read-only. New FR crops preserve native PAN/MS; only LMS may
be explicitly generated with the repository's interp23tap recipe. Exact GT
matching recovers candidate source coordinates, never acquisition identity or
the missing native PAN. Outputs are staging artifacts, not Sources manifests.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from align.resample import interp23tap
from ta2.evaluation import digest, file_sha, atomic_json
from .data import SENSORS

SCHEMA = 'TEACHER48_NATIVE_SCENES_V1'
LMS_RECIPE = 'REPO_INTERP23TAP_R4_FLOAT64_GLOBAL_WRAP_v1'


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be a nonempty provenance string')
    return value


def _ints(values, n, *, minimum=0):
    if (not isinstance(values, (list, tuple)) or len(values) != n
            or any(type(v) is not int or v < minimum for v in values)):
        raise ValueError(f'Expected {n} explicit integers >= {minimum}')
    return tuple(values)


class Raster:
    """Explicit-layout local H5/NPY raster with bounded read-only windows.

    NCHW requires an explicit index; the index is NOT a source-scene identity.
    File SHA is verified once; modifications during reading fail closed.
    """
    def __init__(self, root, ref):
        self.path = (Path(root) / ref['path']).resolve(strict=True)
        self.ref = ref
        self.stamp = self._stamp()
        if not self.path.is_file() or file_sha(self.path) != ref['sha256']:
            raise ValueError('Source file SHA mismatch: ' + str(self.path))
        self.verify()
        self.handle = None
        self.layout = ref['layout']
        if self.path.suffix.lower() in ('.h5', '.hdf5'):
            self.handle = h5py.File(self.path, 'r')
            try:
                self.array = self.handle[ref['key']]
                if (not isinstance(self.array, h5py.Dataset)
                        or Path(self.array.file.filename).resolve() != self.path
                        or self.array.is_virtual or self.array.external):
                    raise ValueError('External/virtual H5 dependencies must be materialized and independently pinned')
            except Exception:
                self.handle.close()
                raise
        elif self.path.suffix.lower() == '.npy':
            self.array = np.load(self.path, mmap_mode='r', allow_pickle=False)
        else:
            raise ValueError('Prepare adapter supports explicit H5/NPY; no implicit TIFF/MAT conversion')
        try:
            shape = self.array.shape
            self.index = ref.get('index')
            if self.layout == 'NCHW':
                if (len(shape) != 4 or type(self.index) is not int
                        or not 0 <= self.index < shape[0]):
                    raise ValueError('NCHW requires a valid explicit index')
                self.shape = shape[1:]
            elif self.layout == 'CHW' and len(shape) == 3:
                self.shape = shape
            elif self.layout == 'HWC' and len(shape) == 3:
                self.shape = (shape[2], shape[0], shape[1])
            elif self.layout == 'HW' and len(shape) == 2:
                self.shape = (1, *shape)
            else:
                raise ValueError('Source shape/layout mismatch')
            if min(self.shape) <= 0 or self.array.dtype.kind not in 'uif':
                raise ValueError('Nonempty real numeric raster required')
            self.verify()
        except Exception:
            self.close()
            raise

    def _stamp(self):
        stat = self.path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def read(self, y, x, h, w):
        self.verify()
        _ints([y, x, h, w], 4)
        if min(h, w) <= 0 or y+h > self.shape[1] or x+w > self.shape[2]:
            raise ValueError('Window outside source raster')
        if self.layout == 'NCHW':
            result = self.array[self.index, :, y:y+h, x:x+w]
        elif self.layout == 'CHW':
            result = self.array[:, y:y+h, x:x+w]
        elif self.layout == 'HWC':
            result = self.array[y:y+h, x:x+w, :].transpose(2, 0, 1)
        else:
            result = self.array[y:y+h, x:x+w][None]
        result = np.asarray(result)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite source pixels; no invented nodata replacement')
        self.verify()
        return result

    def verify(self):
        if self.stamp != self._stamp():
            raise ValueError('Native source changed during preparation')

    def wrapped(self, y, x, h, w):
        # Read wrap-around context from the WHOLE scene, not periodic patch edges.
        def segments(start, length, size):
            while length:
                pos = start % size
                count = min(length, size-pos)
                yield pos, count
                start += count
                length -= count
        return np.concatenate([
            np.concatenate([self.read(yy, xx, hh, ww)
                            for xx, ww in segments(x, w, self.shape[2])], axis=-1)
            for yy, hh in segments(y, h, self.shape[1])], axis=-2)

    def close(self):
        if self.handle is not None:
            self.handle.close()
        elif hasattr(getattr(self, 'array', None), '_mmap'):
            self.array._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def lms_window(ms, pan_crop):
    """Global interp23tap-equivalent crop with sufficient LR context.

    Two x2 stages have radius <=33 HR pixels; 16 LR pixels of halo exceeds
    that support. Physical scene boundaries wrap exactly as the named recipe.
    This is not a claim of matching an unknown dataset generation recipe.
    """
    y, x, h, w = _ints(pan_crop, 4)
    if any(v % 4 for v in (y, x, h, w)) or min(h, w) <= 0:
        raise ValueError('LMS crop must be aligned to the explicit ratio-four grid')
    halo = 16
    context = ms.wrapped(y//4-halo, x//4-halo, h//4+2*halo, w//4+2*halo)
    with torch.no_grad():
        up = interp23tap(torch.from_numpy(context.astype(np.float64))[None], 4)[0].numpy()
    return up[:, 4*halo:4*halo+h, 4*halo:4*halo+w].copy()


class NativeScene:
    def __init__(self, root, scene, dataset):
        self.scene = scene
        self.dataset = dataset
        self.bands, self.peak = SENSORS[dataset]
        for field in ('source_scene_id', 'acquisition_id', 'native_pan_grid',
                      'source_identity_basis', 'geometry_basis'):
            _text(scene[field], field)
        if scene.get('role') != 'native_training_source':
            raise ValueError('Only declared native training sources may be prepared; not benchmark FR')
        if scene.get('radiometry') != 'native_DN' or scene.get('ratio') != 4:
            raise ValueError('Explicit native DN and ratio-four geometry required')
        self.origin = _ints(scene['pan_origin_native_yx'], 2)
        self.lms_mode = scene['lms_mode']
        if self.lms_mode not in ('provided_global', LMS_RECIPE):
            raise ValueError('Choose provided_global LMS or explicitly recorded interp23tap recipe')
        self.stack = ExitStack()
        self.evidence_refs = []
        try:
            # Require immutable source provenance, not role inferred from shape.
            for field in ('native_source_evidence', 'geometry_evidence'):
                ref = scene[field]
                if file_sha((Path(root)/ref['path']).resolve()) != ref['sha256']:
                    raise ValueError('Missing/stale native source evidence')
                self.evidence_refs.append((Path(root)/ref['path'], ref['sha256']))
            self.pan = self.stack.enter_context(Raster(root, scene['pan']))
            self.ms = self.stack.enter_context(Raster(root, scene['ms']))
            if (self.pan.shape[0] != 1 or self.ms.shape[0] != self.bands
                    or self.pan.shape[1:] != tuple(4*v for v in self.ms.shape[1:])):
                raise ValueError('Native PAN/MS common-FOV ratio/band mismatch')
            self.lms = None
            if self.lms_mode == 'provided_global':
                support = scene['lms_support']
                _ints([support['radius_native_pan']], 1)
                if support['boundary'] not in ('wrap', 'bounded'):
                    raise ValueError('Provided LMS dependency boundary must be wrap or bounded')
                _text(support['basis'], 'LMS support basis')
                self.support_radius, self.support_boundary = support['radius_native_pan'], support['boundary']
                self.lms = self.stack.enter_context(Raster(root, scene['lms']))
                if self.lms.shape != (self.bands, *self.pan.shape[1:]):
                    raise ValueError('Provided native LMS must share the PAN grid')
            else:
                # Conservative footprint of the actual 16-LR-pixel context read.
                self.support_radius, self.support_boundary = 64, 'wrap'
            # Prevent simply renaming any currently registered public source.
            catalog = Path(__file__).resolve().parents[1]/'ablr2/sensor_sources.json'
            registered = json.loads(catalog.read_text())['sensors']
            forbidden = {v['sha256'] for s in registered.values() for v in s['splits'].values()}
            if any(scene[k]['sha256'] in forbidden for k in ('pan', 'ms')):
                raise ValueError('Registered RR/benchmark corpus cannot be relabelled as native training')
        except Exception:
            self.stack.close()
            raise

    def crop(self, crop):
        y, x, h, w = _ints(crop, 4)
        if any(v % 4 for v in crop) or min(h, w) <= 0:
            raise ValueError('Native crop must preserve the ratio-four grid')
        pan = self.pan.read(y, x, h, w)
        if pan.min() < 0 or pan.max() > self.peak:
            raise ValueError('PAN pixels outside native sensor DN range')
        ms = self.ms.read(y//4, x//4, h//4, w//4)
        lms = self.lms.read(y, x, h, w) if self.lms is not None else lms_window(self.ms, crop)
        return {'pan': pan, 'ms': ms, 'lms': lms}

    def verify(self):
        for raster in (self.pan, self.ms, self.lms):
            if raster is not None:
                raster.verify()
        for path, sha in self.evidence_refs:
            if file_sha(path) != sha:
                raise ValueError('Native evidence changed during preparation')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stack.close()


def intersects(a, b, margin):
    y, x, h, w = a
    yy, xx, hh, ww = b
    return y < yy+hh+margin and yy < y+h+margin and x < xx+ww+margin and xx < x+w+margin


def support_regions(crop, radius, shape, boundary):
    """Rectangular conservative dependency supports, including scene wrap."""
    def axis(start, length, size):
        start, length = start-radius, length+2*radius
        if boundary == 'bounded':
            lo, hi = max(0, start), min(size, start+length)
            return [(lo, hi-lo)]
        if length >= size:
            return [(0, size)]
        start %= size
        first = min(length, size-start)
        return [(start, first)] + ([(0, length-first)] if first < length else [])
    y, x, h, w = crop
    return [[yy, xx, hh, ww] for yy, hh in axis(y, h, shape[0])
            for xx, ww in axis(x, w, shape[1])]


def crop_native(spec_path, output, *, patch_hw, stride_hw, margin):
    """Prepare FR H5 + truthful lineage, excluding declared held-out supports.

    Results remain STAGED_NOT_ADMITTED: external acquisition/geographic identity,
    RR mapping, prior acceptance and official eval lineage are separate checks.
    Refuses any existing output directory; source files are never overwritten.
    """
    spec_path = Path(spec_path).resolve(strict=True)
    spec_bytes = spec_path.read_bytes()
    spec_sha = hashlib.sha256(spec_bytes).hexdigest()
    spec = json.loads(spec_bytes)
    if spec.get('schema') != SCHEMA or spec.get('dataset') not in SENSORS:
        raise ValueError('Native source catalog schema/dataset mismatch')
    patch_hw = _ints(patch_hw, 2, minimum=4)
    stride_hw = _ints(stride_hw, 2, minimum=4)
    if any(v % 4 for v in (*patch_hw, *stride_hw)):
        raise ValueError('Patch/stride must preserve MS pixel-grid origin')
    if type(margin) is not int or margin < 0:
        raise ValueError('Explicit nonnegative native-PAN split margin required')
    _text(spec['split_basis'], 'split_basis')
    held = spec['heldout_crops']
    if not isinstance(held, list):
        raise ValueError('Explicit heldout crop list is required, even if empty')
    for row in held:
        _text(row['acquisition_id'], 'heldout acquisition_id')
        _ints(row['source_crop_native_pan_yxhw'], 4)
        if min(row['source_crop_native_pan_yxhw'][2:]) <= 0:
            raise ValueError('Heldout crop extent must be positive')
    scenes = spec['scenes']
    if not scenes or len({s['source_scene_id'] for s in scenes}) != len(scenes):
        raise ValueError('Nonempty unique real scene records required')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, files, scene_records, excluded = [], {}, [], []
    try:
        for i, entry in enumerate(scenes):
            with NativeScene(spec_path.parent, entry, spec['dataset']) as source:
                positions = []
                for y in range(0, source.pan.shape[1]-patch_hw[0]+1, stride_hw[0]):
                    for x in range(0, source.pan.shape[2]-patch_hw[1]+1, stride_hw[1]):
                        native = [y+source.origin[0], x+source.origin[1], *patch_hw]
                        dependencies = support_regions([y,x,*patch_hw], source.support_radius,
                                                       source.pan.shape[1:], source.support_boundary)
                        dependencies = [[dy+source.origin[0], dx+source.origin[1], dh, dw]
                                        for dy,dx,dh,dw in dependencies]
                        if any(entry['acquisition_id'] == r['acquisition_id'] and
                               intersects(dep, r['source_crop_native_pan_yxhw'], margin)
                               for r in held for dep in dependencies):
                            excluded.append({'source_scene_id': entry['source_scene_id'], 'crop': native})
                        else:
                            positions.append((y, x, native))
                if not positions:
                    raise ValueError('No non-heldout native crops for scene ' + entry['source_scene_id'])
                name = f'native_fr_{i:04d}.h5'
                path = output/name
                with h5py.File(path, 'x') as f:
                    f.attrs['schema'] = 'TEACHER48_NATIVE_FR_STAGING_V1'
                    f.attrs['source_scene_id'] = entry['source_scene_id']
                    f.attrs['acquisition_id'] = entry['acquisition_id']
                    f.attrs['source_catalog_sha256'] = spec_sha
                    f.attrs['lms_recipe'] = source.lms_mode
                    for j, (y, x, native) in enumerate(positions):
                        arrays = source.crop([y, x, *patch_hw])
                        if j == 0:
                            for key, a in arrays.items():
                                f.create_dataset(key, (len(positions), *a.shape), dtype=a.dtype,
                                                 chunks=(1, *a.shape), compression='lzf')
                        for key, a in arrays.items():
                            f[key][j] = a
                        rows.append({'sample_id': f'{entry["source_scene_id"]}:FR:y{native[0]}:x{native[1]}:h{native[2]}:w{native[3]}',
                            'source_scene_id': entry['source_scene_id'], 'acquisition_id': entry['acquisition_id'],
                            'geometry_evidence': entry['geometry_basis'], 'file_id': name, 'row': j,
                            'grid_name': entry['native_pan_grid'], 'source_crop_native_pan_yxhw': native,
                            'preparation_augmentation': {'hflip': False, 'vflip': False, 'rot90': 0},
                            'native_fr_training': True})
                files[name] = {'path': str(path), 'sha256': file_sha(path)}
                scene_records.append(entry)
                source.verify()
        if file_sha(spec_path) != spec_sha:
            raise ValueError('Native source catalog changed during preparation')
        result = {'schema': 'TEACHER48_FR_PREPARATION_V1', 'status': 'STAGED_NOT_ADMITTED',
            'dataset': spec['dataset'], 'source_catalog': {'path': str(spec_path), 'sha256': spec_sha},
            'files': files, 'samples': {'fr_train': rows}, 'scenes': scene_records,
            'patch_hw': patch_hw, 'stride_hw': stride_hw, 'split_margin_native_pan': margin,
            'declared_heldout_exclusions': excluded, 'split_basis': spec['split_basis'],
            'geographic_or_near_duplicate_certification': 'NOT_INFERRED_FROM_COORDINATES_OR_SHAPE',
            'pending': ['RR_SOURCE_MAPPING_AND_GRID_PHASE', 'REVIEWED_SCENE_PRIORS',
                        'COMPLETE_TRAIN_EVAL_SEPARATION', 'OFFICIAL_EVAL_LINEAGE', 'SOURCES_ADMISSION'],
            'recipe_source_sha256': file_sha(Path(__file__).resolve().parents[1]/'align/resample.py')}
        atomic_json(output/'preparation.json', result, immutable=True)
        return result
    except Exception as exc:
        # Keep partial artifacts recoverably; never mistake them for usable data.
        atomic_json(output/'FAILED.json', {'status': 'PREPARATION_FAILED', 'error': str(exc)}, immutable=True)
        raise


def exact_gt_candidates(gt, source_ms, *, max_candidates=128):
    """Find exact all-band native-MS matches over the eight dihedral transforms.

    No NCC threshold or near-match is promoted to lineage. Multiple locations
    or transforms remain ambiguous. Candidate discovery uses four spatial
    anchors; final equality tests every pixel in every band.
    """
    gt, source_ms = np.asarray(gt), np.asarray(source_ms)
    if (gt.ndim != 3 or source_ms.ndim != 3 or gt.shape[0] != source_ms.shape[0]
            or min(gt.shape) <= 0 or min(source_ms.shape) <= 0
            or not np.isfinite(gt).all() or not np.isfinite(source_ms).all()):
        raise ValueError('Finite compatible CHW GT and native MS required')
    if type(max_candidates) is not int or max_candidates < 2:
        raise ValueError('Candidate limit must retain ambiguity (>=2)')
    found = []
    for flip in (False, True):
        for rotation in range(4):
            # Saved patch = rot90(hflip(source_crop), k); invert in reverse order.
            target = np.rot90(gt, -rotation, axes=(-2, -1))
            if flip:
                target = target[..., ::-1]
            c, h, w = target.shape
            ny, nx = source_ms.shape[1]-h+1, source_ms.shape[2]-w+1
            if min(ny, nx) <= 0:
                continue
            mask = np.ones((ny, nx), dtype=bool)
            for band, yy, xx in ((0,0,0), (c-1,h-1,w-1), (0,h//2,w//2), (c//2,h//3,w//3)):
                mask &= source_ms[band, yy:yy+ny, xx:xx+nx] == target[band, yy, xx]
            for y, x in zip(*np.nonzero(mask)):
                if np.array_equal(source_ms[:, y:y+h, x:x+w], target):
                    found.append({'native_ms_crop_yxhw': [int(y), int(x), h, w],
                                  'preparation_augmentation': {'hflip': flip, 'vflip': False, 'rot90': rotation}})
                    if len(found) >= max_candidates:
                        return {'status': 'AMBIGUOUS', 'candidates': found, 'truncated': True, 'accepted': False}
    status = 'UNMATCHED' if not found else 'AMBIGUOUS' if len(found) > 1 else 'EXACT_GT_ONLY_CANDIDATE'
    return {'status': status, 'candidates': found, 'truncated': False, 'accepted': False,
            'remaining_verification': 'Native PAN counterpart and RR degradation/grid phase must also be verified'}
