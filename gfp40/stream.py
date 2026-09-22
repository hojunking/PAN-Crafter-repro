"""Paired source/geometry/uniform stream, independent of distribution CDF."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import torch

from fh12.training import BatchStream


def gamma_uniforms(seed, offset, count):
    if any(int(v) != v or v < 0 for v in (seed, offset, count)):
        raise ValueError('Uniform seed/counter/count must be nonnegative integers')
    with np.errstate(over='ignore'):
        z = np.arange(offset, offset+count, dtype=np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return torch.from_numpy((z >> np.uint64(11)).astype(np.int64))


def map_uniforms(uniforms, family='G025'):
    from gfp40.plan import family_for
    spec = family_for('G025' if family == 'NATIVE' else family)
    values = np.asarray(uniforms, dtype=np.int64)
    if ((values < 0) | (values >= 2**53)).any():
        raise ValueError('Uniform53 outside [0, 2**53)')
    thresholds = np.cumsum(np.asarray(spec['tokens'], dtype=np.int64))[:-1]
    # All registered denominators are powers of two, hence exact integer CDF.
    denominator = sum(spec['tokens'])
    if 2**53 % denominator: raise ValueError('Nonexact registered CDF denominator')
    thresholds *= 2**53 // denominator
    return torch.from_numpy(np.searchsorted(thresholds, values, side='right').astype(np.int64))


def gamma_ids(seed, offset, count, family='G025'):
    return map_uniforms(gamma_uniforms(seed, offset, count), family)


class PairedBatchStream(BatchStream):
    def __init__(self, n, batch_size, seed, family='G025'):
        from gfp40.plan import family_for
        self.seed = int(seed); self.family = 'G025' if family == 'NATIVE' else family
        self.native_gamma_id = list(family_for(self.family)['gammas']).index(1.)
        super().__init__(n, batch_size, self.seed)

    @property
    def completed_updates(self): return self.epoch*self.count+self.cursor
    @property
    def gamma_counter(self): return self.completed_updates*self.batch_size
    @property
    def worker_seed(self): return self.seed+500000+self.epoch

    def batch_at(self, index):
        if not 0 <= index < self.count: raise ValueError('Batch outside epoch')
        start = index*self.batch_size
        uniforms = gamma_uniforms(self.seed+600000, (self.epoch*self.count+index)*self.batch_size, self.batch_size)
        gammas = map_uniforms(uniforms, self.family)
        return [(int(self.order[j]), int(self.rotations[j]), int(gammas[j-start]), int(uniforms[j-start]))
                for j in range(start, start+self.batch_size)]

    def remaining_batches(self): return [self.batch_at(i) for i in range(self.cursor, self.count)]
    def advance(self):
        if self.cursor >= self.count: raise ValueError('Cannot commit past epoch')
        self.cursor += 1

    def state_dict(self):
        return dict(super().state_dict(), seed=self.seed, family=self.family,
            gamma_seed=self.seed+600000, gamma_counter=self.gamma_counter,
            rng_mapping='GFP40_UNIFORM53_OFFSETS_v1')

    def load_state_dict(self, value):
        if (value.get('seed') != self.seed or value.get('family') != self.family
                or value.get('gamma_seed') != self.seed+600000
                or value.get('rng_mapping') != 'GFP40_UNIFORM53_OFFSETS_v1'):
            raise ValueError('P40 paired stream RNG/family identity changed')
        super().load_state_dict(value)
        if value.get('gamma_counter') != self.gamma_counter:
            raise ValueError('Uniform counter differs from consumed updates')

    def initial_prefix(self, count=256):
        probe = type(self)(self.n, self.batch_size, self.seed, self.family); rows=[]
        while len(rows) < count:
            for batch in probe.remaining_batches():
                rows.extend(batch)
                if len(rows) >= count: break
            probe.new_epoch()
        rows = [list(row) for row in rows[:count]]
        digest = lambda val: hashlib.sha256(json.dumps(val, separators=(',', ':')).encode()).hexdigest()
        paired = [[r[0], r[1], r[3]] for r in rows]
        return dict(count=count, family=self.family, triples=rows, draws=rows,
            sample_view_uniform=paired,
            sample_view_sha256=digest([r[:2] for r in rows]),
            sample_view_uniform_sha256=digest(paired), uniform_sha256=digest([r[3] for r in rows]),
            sample_view_gamma_sha256=digest(rows),
            seed_mapping=dict(source=self.seed+300000, geometry=self.seed+400000,
                workers=self.seed+500000, gamma=self.seed+600000),
            gamma_algorithm='SplitMix64-counter; high53bits uniform; exact integer family CDF')


def effective_gamma_histogram(meta, family='G025'):
    from gfp40.plan import family_for
    gammas = family_for('G025' if family == 'NATIVE' else family)['gammas']
    array = np.asarray(meta)
    if array.ndim != 2 or array.shape[1] != 7: raise ValueError('Expected seven-field metadata')
    return {str(gamma): int((array[:, 4] == index).sum()) for index, gamma in enumerate(gammas)}
