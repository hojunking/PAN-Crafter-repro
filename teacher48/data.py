"""Hash-pinned prepared observations with actual source-scene/crop lineage.

This adapter never makes native FR from RR, interpolates LMS, or discovers a
training split by filename. Preparation/scene-estimation evidence is an input,
not a claim reconstructed from an HDF5 row number.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from panda_method.data import BatchLineage, PriorBinding, augment_batch, bind_scene_priors
from panda_method.geometry import ScenePriorRecord, ScenePriorRegistry, VerifiedGridTransform, transform_prior
from ta2.evaluation import digest, file_sha

GROUPS = ('rr_train', 'fr_train', 'rr_eval', 'fr_eval')
SENSORS = {'WV3': (8, 2047.), 'QB': (4, 2047.), 'GF2': (4, 1023.)}


def evidence(root, ref):
    """Require an existing immutable evidence file, not merely an assertion."""
    path = (root / ref['path']).resolve()
    if not path.is_file() or file_sha(path) != ref['sha256']:
        raise ValueError('Missing/stale evidence: ' + str(path))
    return path


class Sources:
    def __init__(self, path, *, dataset, baseline, scan=True):
        self.path = Path(path).resolve()
        self.root = self.path.parent
        self.manifest = m = json.loads(self.path.read_text())
        self.sha256 = file_sha(self.path)
        self.dataset, self.bands, self.peak = dataset, *SENSORS[dataset]
        if (m.get('schema') != 'TEACHER48_SOURCES_V1' or m.get('dataset') != dataset
                or m.get('radiometry') != 'native_DN' or m.get('ratio') != 4):
            raise ValueError('Explicit native DN dataset/ratio source manifest required')
        for name in ('scene_mapping_evidence', 'native_fr_training_evidence',
                     'lms_provenance_evidence', 'prior_acceptance_evidence',
                     'split_separation_evidence', 'evaluation_protocol_evidence'):
            evidence(self.root, m[name])
        separation = json.loads(evidence(self.root, m['split_separation_evidence']).read_text())
        if (separation.get('accepted') is not True
                or separation.get('exact_duplicates') != 0
                or separation.get('near_duplicates') != 0
                or separation.get('protocol') is None
                or separation.get('sample_index_sha256') != digest(m['samples'])):
            raise ValueError('Exact/near-duplicate audit must bind this complete sample index')
        registry = ScenePriorRegistry(ScenePriorRecord(**r) for r in m['scene_priors'])
        self.samples = m['samples']
        if set(self.samples) != set(GROUPS):
            raise ValueError('Four separate RR/FR train/eval groups required')
        ids, pairs = set(), []
        for group in GROUPS:
            rows = self.samples[group]
            if not rows or (group.endswith('eval') and len(rows) != 20):
                raise ValueError('Training cannot be empty; official evaluation needs all20')
            scene_ids = set()
            for row in rows:
                sample, scene = row['sample_id'], row['source_scene_id']
                if not isinstance(sample, str) or not sample or sample in ids:
                    raise ValueError('Globally unique actual sample IDs required')
                ids.add(sample); pairs.append((sample, scene)); scene_ids.add(scene)
                registry.record(scene)
                if (not row.get('acquisition_id') or not row.get('geometry_evidence')
                        or not row.get('source_crop_native_pan_yxhw')
                        or not row.get('preparation_augmentation')):
                    raise ValueError('Actual acquisition/crop/preparation lineage missing')
                y, x, h, w = row['source_crop_native_pan_yxhw']
                if any(type(v) is not int for v in (y,x,h,w)) or min(y, x) < 0 or min(h, w) <= 0:
                    raise ValueError('Invalid source crop in common native PAN coordinates')
                prep = row['preparation_augmentation']
                if (set(prep) != {'hflip','vflip','rot90'} or type(prep['hflip']) is not bool
                        or type(prep['vflip']) is not bool or type(prep['rot90']) is not int
                        or prep['rot90'] not in range(4)):
                    raise ValueError('Prepared patch hflip/vflip/rot90 transform must be explicit')
                if type(row['row']) is not int or row['row'] < 0:
                    raise ValueError('Invalid H5 row')
                if group == 'fr_train' and row.get('native_fr_training') is not True:
                    raise ValueError('FR training must be a native training observation')
            if group.endswith('eval') and len({r['scene_id'] for r in rows}) != len(rows):
                raise ValueError('Unique official scene IDs required (not inferred H5 indices)')
        self.binding = PriorBinding(registry, sample_scene_pairs=pairs,
                                    grid_transforms=[VerifiedGridTransform(**v) for v in m['grid_transforms']])
        self.mapping_sha256 = self.binding.fingerprint()
        self.group_hashes = {g: digest({'manifest_sha256': self.sha256, 'group': g}) for g in GROUPS}
        self._check_crops()
        self.files = {}
        for key, ref in m['files'].items():
            self.files[key] = evidence(self.root, ref)
        self._file_stamps = {key: (p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino)
                             for key,p in self.files.items()}
        if scan:
            self._scan(baseline)

    def _check_crops(self):
        # Conservative explicit same-acquisition intersection check, independent
        # of the externally supplied near-duplicate audit.
        held = self.samples['rr_eval'] + self.samples['fr_eval']
        margin = self.manifest['split_separation_margin_native_pan']
        if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
            raise ValueError('Explicit nonnegative split buffer required')
        for a in self.samples['rr_train'] + self.samples['fr_train']:
            y, x, h, w = a['source_crop_native_pan_yxhw']
            for b in held:
                if a['acquisition_id'] != b['acquisition_id']:
                    continue
                by, bx, bh, bw = b['source_crop_native_pan_yxhw']
                if (y < by + bh + margin and by < y + h + margin
                        and x < bx + bw + margin and bx < x + w + margin):
                    raise ValueError('Train/evaluation source crops overlap or violate buffer')

    def arrays(self, group, index):
        row = self.samples[group][index]
        path=self.files[row['file_id']]; st=path.stat()
        if (st.st_size,st.st_mtime_ns,st.st_ino) != self._file_stamps[row['file_id']]:
            raise ValueError('Pinned source file changed after admission')
        with h5py.File(self.files[row['file_id']], 'r') as f:
            keys = ('pan', 'ms', 'lms') + (('gt',) if group.startswith('rr') else ())
            keys += (('lpan',) if self.manifest['uses_lpan'] else ())
            # Retain stored precision for official GT/PAN/LMS/MS metrics.
            # Only the network-input adapter below converts to FP32.
            return {key: np.asarray(f[self.manifest['h5_keys'][key]][row['row']])
                    for key in keys}

    def _scan(self, baseline):
        seen_train, held = set(), set()
        for group in GROUPS:
            group_seen=set(); row_seen=set()
            hw = (baseline['RR_train_HW'] if group == 'rr_train' else baseline['FR_train_HW']
                  if group == 'fr_train' else [256, 256] if group == 'rr_eval' else [512, 512])
            if isinstance(hw, int):
                hw = [hw, hw]
            h, w = hw
            expected = {'pan': (1, h, w), 'ms': (self.bands, h//4, w//4),
                        'lms': (self.bands, h, w), 'gt': (self.bands, h, w),
                        'lpan': (1, h//4, w//4)}
            for i in range(len(self.samples[group])):
                row=self.samples[group][i]
                key=(self.manifest['files'][row['file_id']]['sha256'],row['row'])
                if group.endswith('eval') and key in row_seen:
                    raise ValueError('Duplicate official evaluation file/row under different scene IDs')
                row_seen.add(key)
                arrays = self.arrays(group, i)
                for name, a in arrays.items():
                    if a.shape != expected[name] or not np.isfinite(a).all():
                        raise ValueError(f'Invalid native geometry/finite values: {group}/{i}/{name}')
                    # Dataset MS/LMS/LPAN may contain filter ringing; preserve
                    # finite overshoot exactly as the existing source protocol.
                    if name in ('pan', 'gt') and (a.min() < 0 or a.max() > self.peak):
                        raise ValueError('Source is not native sensor DN')
                fingerprint = hashlib.sha256(np.ascontiguousarray(arrays['pan']).tobytes()
                                             + np.ascontiguousarray(arrays['ms']).tobytes()).hexdigest()
                if group.endswith('eval') and fingerprint in group_seen:
                    raise ValueError('Duplicate official evaluation observations')
                group_seen.add(fingerprint)
                (seen_train if group.endswith('train') else held).add(fingerprint)
        if seen_train & held:
            raise ValueError('Exact train/evaluation observation duplicate')

    def batch(self, group, indices, *, device='cpu'):
        rows = [self.samples[group][i] for i in indices]
        arrays = [self.arrays(group, i) for i in indices]
        tensors = {key: torch.from_numpy(np.stack([a[key] for a in arrays]).astype(np.float32)).to(device).mul(2/self.peak).sub(1)
                   for key in arrays[0]}
        grids = {r['grid_name'] for r in rows}
        if len(grids) != 1:
            raise ValueError('One prepared grid per microbatch required')
        lineage = BatchLineage(split='train' if group.endswith('train') else 'test',
            resolution=group[:2].upper(), source_manifest_sha256=self.group_hashes[group],
            prior_mapping_sha256=self.mapping_sha256,
            scene_ids=tuple(r['source_scene_id'] for r in rows),
            sample_ids=tuple(r['sample_id'] for r in rows), grid_name=next(iter(grids)),
            geometry_evidence='; '.join(r['geometry_evidence'] for r in rows),
            native_training_evidence=json.dumps(self.manifest['native_fr_training_evidence']))
        batch = bind_scene_priors(pan=tensors['pan'], ms=tensors['ms'], native_lms=tensors['lms'],
            lineage=lineage, binding=self.binding, gt=tensors.get('gt'), lpan=tensors.get('lpan'))
        # H5 pixels already include this preparation transform; transform ONLY
        # their canonical scene priors, before any online augmentation.
        priors = torch.stack([transform_prior(batch.prior_dy_dx[i], **row['preparation_augmentation'])
                              for i,row in enumerate(rows)])
        return replace(batch, prior_dy_dx=priors)

    def expected_ids(self, group):
        return tuple(r['scene_id'] for r in self.samples[group])


class TrainingStream:
    """Independent shuffle/augmentation generators; no global RNG mutation."""
    def __init__(self, sources, group, *, seed, augmentation_seed, microbatch, accumulation,
                 augmentation, device):
        self.sources, self.group, self.device = sources, group, device
        if group not in ('rr_train','fr_train'):
            raise ValueError('Training stream cannot use evaluation observations')
        if any(type(v) is not int or v <= 0 for v in (microbatch,accumulation)):
            raise ValueError('Positive integer microbatch/accumulation required')
        self.microbatch, self.accumulation, self.augmentation = microbatch, accumulation, augmentation
        self.rng = np.random.default_rng(seed)
        self.augment_rng = np.random.default_rng(augmentation_seed)
        self.order, self.cursor = [], 0
        if set(augmentation) != {'hflip_probability', 'vflip_probability', 'rot90'}:
            raise ValueError('Pin hflip_probability/vflip_probability/rot90 explicitly')
        if any(not 0 <= augmentation[k] <= 1 for k in ('hflip_probability', 'vflip_probability')):
            raise ValueError('Invalid augmentation probability')
        if type(augmentation['rot90']) is not bool:
            raise ValueError('Explicit uniform rot90 toggle required')

    def _sample(self):
        if self.cursor == len(self.order):
            self.order = self.rng.permutation(len(self.sources.samples[self.group])).tolist()
            self.cursor = 0
        i = self.order[self.cursor]; self.cursor += 1
        batch = self.sources.batch(self.group, [i], device=self.device)
        a = self.augmentation
        return augment_batch(batch, hflip=bool(self.augment_rng.random() < a['hflip_probability']),
                             vflip=bool(self.augment_rng.random() < a['vflip_probability']),
                             rot90=int(self.augment_rng.integers(4)) if a['rot90'] else 0)

    def __call__(self):
        result = []
        for _ in range(self.accumulation):
            samples = [self._sample() for _ in range(self.microbatch)]
            first = samples[0]
            lineage = replace(first.lineage,
                scene_ids=tuple(b.lineage.scene_ids[0] for b in samples),
                sample_ids=tuple(b.lineage.sample_ids[0] for b in samples),
                geometry_evidence='; '.join(b.lineage.geometry_evidence for b in samples))
            fields = {k: torch.cat([getattr(b, k) for b in samples]) if getattr(first, k) is not None else None
                      for k in ('pan', 'ms', 'native_lms', 'prior_dy_dx', 'gt', 'lpan')}
            result.append(replace(first, lineage=lineage, **fields))
        return result

    def state(self):
        return {'shuffle': self.rng.bit_generator.state, 'augmentation': self.augment_rng.bit_generator.state,
                'order': self.order, 'cursor': self.cursor}
