"""Family-local training adapter; official evaluation remains native G20."""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from gfp40.augmentation import verify_augmentation_cache


class GFP40Dataset(Dataset):
    def __init__(self, native_dataset, augmentation_manifest=None, arm='NATIVE0', family='G025'):
        from gfp40.plan import family_for
        if arm not in ('NATIVE0', 'CTRL', 'MIX'):
            raise ValueError('Unbound/unknown input arm')
        self.native, self.arm, self.profile = native_dataset, arm, arm
        self.family = family_for('G025' if family == 'NATIVE' else family)
        self.gammas = list(self.family['gammas']); self.native_gamma_id = self.gammas.index(1.)
        for key in ('base_count', 'has_gt', 'bands', 'max_pixel', 'split',
                    'raw_h5_path', 'lp_path', 'spec', 'source_hashes', 'contract_hash'):
            if hasattr(native_dataset, key): setattr(self, key, getattr(native_dataset, key))
        self.augment = native_dataset.split == 'train'; self.views = {}
        if augmentation_manifest is not None:
            manifest = verify_augmentation_cache(augmentation_manifest, native_dataset)
            if not set(self.gammas).issubset(manifest['gammas']):
                raise ValueError('Augmentation bank lacks registered family slices')
            for shard in manifest['shards']:
                if shard['gamma'] in self.gammas and shard['gamma'] != 1.:
                    with h5py.File(shard['cache_path'], 'r') as cache:
                        self.views[shard['gamma']] = {key: cache[key][:] for key in ('pan', 'lpan')}
        if arm == 'MIX' and set(self.gammas) - {1.} != set(self.views):
            raise ValueError('MIX requires complete verified family PAN/LP bank')

    def __len__(self): return self.base_count

    def get_view(self, index, rot=0, augment=False, gamma_id=None):
        gamma_id = self.native_gamma_id if gamma_id is None else int(gamma_id)
        if gamma_id not in range(len(self.gammas)):
            raise ValueError('Invalid family-local gamma ID')
        gamma = self.gammas[gamma_id]
        if self.split != 'train' and (gamma != 1. or augment):
            raise ValueError('Validation/RR/FR must remain native')
        row = list(self.native.get_view(index, rot, augment=augment))
        if gamma != 1.:
            if gamma not in self.views:
                raise ValueError('Augmented view lacks full verified cache')
            for position, key in ((3, 'lpan'), (4, 'pan')):
                x = self.views[gamma][key][int(index)]
                if augment: x = np.rot90(x[:, ::-1, ::-1], int(rot), axes=(1, 2))
                row[position] = torch.from_numpy(np.array(x, dtype=np.float32, copy=True)).mul_(2/self.max_pixel).sub_(1)
        row[-1] = torch.cat((row[-1], torch.tensor([gamma_id, gamma_id, -1], dtype=torch.int64)))
        return tuple(row)

    def base(self, index): return self.get_view(index)

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            if len(index) != 4:
                raise ValueError('Sampler must provide source, geometry, drawn gamma, uniform53')
            source, view, drawn, uniform = map(int, index)
            if drawn not in range(len(self.gammas)) or not 0 <= uniform < 2**53:
                raise ValueError('Invalid gamma draw metadata')
            effective = drawn if self.arm == 'MIX' else self.native_gamma_id
            row = list(self.get_view(source, view, augment=True, gamma_id=effective))
            row[-1][-2:] = torch.tensor([drawn, uniform], dtype=torch.int64)
            return tuple(row)
        return self.base(int(index))
