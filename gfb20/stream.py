"""Explicit paired sample/view stream and independent counter-based gamma draws."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import torch

from fh12.training import BatchStream


def gamma_ids(seed, offset, count):
    """SplitMix64 counter PRNG; quarters map to [.75,1,1,1.25].

    The gamma stream cannot consume source/view/worker/global RNG. A counter
    names each *consumed* sample, so worker prefetch never commits draws.
    """
    if any(int(v) != v or v < 0 for v in (seed, offset, count)):
        raise ValueError('Gamma seed/counter/count must be nonnegative integers')
    with np.errstate(over='ignore'):
        z = np.arange(offset, offset + count, dtype=np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return torch.from_numpy(np.asarray([0, 1, 1, 2], dtype=np.int64)[(z & np.uint64(3)).astype(np.int64)].copy())


class PairedBatchStream(BatchStream):
    """BatchStream adapter pins all four offsets, without profile input."""
    def __init__(self, n, batch_size, seed):
        self.seed = int(seed)
        super().__init__(n, batch_size, self.seed)

    @property
    def completed_updates(self):
        return self.epoch * self.count + self.cursor

    @property
    def gamma_counter(self):
        return self.completed_updates * self.batch_size

    @property
    def worker_seed(self):
        return self.seed + 500000 + self.epoch

    def batch_at(self, index):
        if not 0 <= index < self.count:
            raise ValueError('Batch index is outside this epoch')
        start = index * self.batch_size
        gamma = gamma_ids(self.seed + 600000,
                          (self.epoch * self.count + index) * self.batch_size, self.batch_size)
        return [(int(self.order[j]), int(self.rotations[j]), int(gamma[j - start]))
                for j in range(start, start + self.batch_size)]

    def remaining_batches(self):
        return [self.batch_at(i) for i in range(self.cursor, self.count)]

    def advance(self):
        if self.cursor >= self.count:
            raise ValueError('Cannot commit past this epoch')
        self.cursor += 1

    def state_dict(self):
        return dict(super().state_dict(), seed=self.seed, gamma_seed=self.seed + 600000,
                    gamma_counter=self.gamma_counter, rng_mapping='GFB20_OFFSETS_v1')

    def load_state_dict(self, value):
        if (value.get('seed') != self.seed or value.get('gamma_seed') != self.seed + 600000
                or value.get('rng_mapping') != 'GFB20_OFFSETS_v1'):
            raise ValueError('B20 paired stream RNG identity changed')
        super().load_state_dict(value)
        if value.get('gamma_counter') != self.gamma_counter:
            raise ValueError('Gamma counter is not paired with consumed sampler updates')

    def initial_prefix(self, count=256):
        probe = type(self)(self.n, self.batch_size, self.seed)
        rows = []
        while len(rows) < count:
            for batch in probe.remaining_batches():
                rows.extend(batch)
                if len(rows) >= count:
                    break
            probe.new_epoch()
        rows = [list(row) for row in rows[:count]]
        digest = lambda x: hashlib.sha256(json.dumps(x, separators=(',', ':')).encode()).hexdigest()
        return dict(count=count, sample_view_sha256=digest([r[:2] for r in rows]),
                    sample_view_gamma_sha256=digest(rows), triples=rows,
                    seed_mapping=dict(source=self.seed + 300000, geometry=self.seed + 400000,
                                      workers=self.seed + 500000, gamma=self.seed + 600000),
                    gamma_algorithm='SplitMix64-counter; low2bits->[0,1,1,2]')
