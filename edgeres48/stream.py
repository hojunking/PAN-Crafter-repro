"""Independent RR data/augmentation streams, never an FR training loader."""
from __future__ import annotations
import copy
import math
import h5py
import numpy as np


class RRStream:
    def __init__(self, sources, config, *, device):
        self.sources, self.config, self.device = sources, config, device
        b, seeds = config['baseline_resolved'], config['stream_seeds']
        self.micro, self.accumulation = b['microbatch_RR'], b['accumulation_RR']
        self.augmentation = b['augmentation']
        self.rng = np.random.default_rng(seeds['rr_data'])
        self.aug = np.random.default_rng(seeds['rr_augmentation'])
        with h5py.File(sources._verify('train'), 'r') as f:
            self.count = len(f['pan'])
        if self.count < self.micro:
            raise ValueError('Training split cannot supply even one microbatch')
        self.order = self.rng.permutation(self.count).tolist()
        self.cursor, self.epoch = 0, 0

    def _rows(self):
        rows = []
        while len(rows) < self.micro:
            take = min(self.micro-len(rows), self.count-self.cursor)
            rows.extend(self.order[self.cursor:self.cursor+take]); self.cursor += take
            if self.cursor == self.count:
                self.epoch += 1
                self.order = self.rng.permutation(self.count).tolist(); self.cursor = 0
        return rows

    def next_update(self):
        batches = []
        for _ in range(self.accumulation):
            a = self.augmentation
            transform = (bool(self.aug.random() < a['hflip_probability']),
                         bool(self.aug.random() < a['vflip_probability']),
                         int(self.aug.integers(4)) if a['rot90'] else 0)
            batches.append(self.sources.batch('train', self._rows(), device=self.device,
                                             augmentation=transform))
        return batches

    def state_dict(self):
        return copy.deepcopy({'rng': self.rng.bit_generator.state,
            'augmentation_rng': self.aug.bit_generator.state, 'order': self.order,
            'cursor': self.cursor, 'epoch': self.epoch, 'count': self.count,
            'sampling': 'SHUFFLED_EPOCH_NO_DROP', 'augmentation_scope': 'ONE_SHARED_MICROBATCH_TRANSFORM'})


def learning_rates(config, update):
    """One-based warmup; first post-warmup cosine point; exact endpoint min."""
    b, horizon = config['baseline_resolved'], config['planned_horizon']
    warmup = b['warmup_updates0']
    if type(update) is not int or not 1 <= update <= horizon:
        raise ValueError('Scheduler update must lie in [1, planned_horizon]')
    def rate(peak):
        if warmup and update <= warmup:
            return peak * update / warmup
        phase = (update-warmup)/(horizon-warmup)
        return b['min_lr'] + .5*(peak-b['min_lr'])*(1+math.cos(math.pi*phase))
    return {'A': rate(config['lr_A']), 'T': rate(config['lr_U'])}


def set_learning_rates(trainer, config):
    rates = learning_rates(config, trainer.updates+1)
    for group in trainer.optimizer.param_groups:
        group['lr'] = rates[group['role']]
    return rates
