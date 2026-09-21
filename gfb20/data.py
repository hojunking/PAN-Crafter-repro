"""GFB20 view adapter; all native evaluation remains the original G20 path."""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from gfb20.augmentation import verify_augmentation_cache


class GFB20Dataset(Dataset):
    def __init__(self, native_dataset, augmentation_manifest=None, profile='BASE'):
        if profile not in {'BASE', 'H010', 'K1', 'E100', 'NATIVE0', 'NAT_MIXCAL', 'PANMIX'}:
            raise ValueError('Unbound/unknown input profile')
        self.native, self.profile = native_dataset, profile
        for key in ('base_count', 'has_gt', 'bands', 'max_pixel', 'split',
                    'raw_h5_path', 'lp_path', 'spec', 'source_hashes', 'contract_hash'):
            if hasattr(native_dataset, key):
                setattr(self, key, getattr(native_dataset, key))
        self.augment = native_dataset.split == 'train'
        self.views = None
        if augmentation_manifest is not None:
            manifest = verify_augmentation_cache(augmentation_manifest, native_dataset, full=False)
            with h5py.File(manifest['cache_path'], 'r') as cache:
                # Only the train adapter loads these; native RR/FR is unchanged.
                self.views = {key: cache[key][:] for key in ('pan', 'lpan')}
        if profile == 'PANMIX' and self.views is None:
            raise ValueError('PANMIX requires complete, verified augmentation cache')

    def __len__(self):
        return self.base_count

    def get_view(self, index, rot=0, augment=False, gamma_id=1):
        if gamma_id not in (0, 1, 2):
            raise ValueError('Invalid gamma ID')
        if self.split != 'train' and (gamma_id != 1 or augment):
            raise ValueError('Validation/RR/FR must remain native')
        row = list(self.native.get_view(index, rot, augment=augment))
        if gamma_id != 1:
            if self.views is None:
                raise ValueError('Augmented view requested without verified full cache')
            for position, key in ((3, 'lpan'), (4, 'pan')):
                x = self.views[key][int(index), gamma_id]
                if augment:
                    x = np.rot90(x[:, ::-1, ::-1], int(rot), axes=(1, 2))
                row[position] = torch.from_numpy(np.array(x, dtype=np.float32, copy=True)).mul_(2 / self.max_pixel).sub_(1)
        row[-1] = torch.cat((row[-1], torch.tensor([gamma_id, gamma_id])))
        return tuple(row)

    def base(self, index):
        return self.get_view(index)

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            if len(index) != 3:
                raise ValueError('Training sampler must provide source, geometry, drawn-gamma')
            source, view, drawn = map(int, index)
            if drawn not in (0, 1, 2):
                raise ValueError('Invalid drawn gamma')
            effective = drawn if self.profile == 'PANMIX' else 1
            row = list(self.get_view(source, view, augment=True, gamma_id=effective))
            row[-1][-1] = drawn
            return tuple(row)
        return self.base(int(index))
