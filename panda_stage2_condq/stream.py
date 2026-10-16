"""Paired RR-only order/augmentation, including exact resume state."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StreamSpec:
    data_seed: int
    augmentation_seed: int
    # These are the actual current donor24 baseline probabilities, not .5.
    hflip_probability: float = 1.
    vflip_probability: float = 1.
    rot90: bool = True

    def __post_init__(self):
        for value in (self.data_seed, self.augmentation_seed):
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError('Explicit nonnegative paired stream seeds required')
        for value in (self.hflip_probability, self.vflip_probability):
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Finite augmentation probability in [0,1] required')
        if type(self.rot90) is not bool:
            raise ValueError('Explicit rotation switch required')


class TrainStream:
    def __init__(self, registered, spec, *, microbatch_size, effective_batch_size, device='cpu'):
        from .sources import RegisteredRRTrain
        if not isinstance(registered, RegisteredRRTrain) or not isinstance(spec, StreamSpec):
            raise TypeError('Registered train sources and explicit paired StreamSpec required')
        if (type(microbatch_size) is not int or type(effective_batch_size) is not int
                or microbatch_size <= 0 or effective_batch_size <= 0
                or effective_batch_size % microbatch_size):
            raise ValueError('Positive divisible batch sizes required')
        self.registered, self.spec, self.device = registered, spec, device
        self.micro, self.total = microbatch_size, effective_batch_size
        self.count = len(registered.sample_ids)
        if self.count < self.micro:
            raise ValueError('Train population smaller than one microbatch')
        self.rng = np.random.default_rng(spec.data_seed)
        self.aug = np.random.default_rng(spec.augmentation_seed)
        self.order = self.rng.permutation(self.count).tolist()
        self.cursor, self.epoch = 0, 0
        self._contract = {'schema': 'PANDA_S2_CONDQ_RR_STREAM_V1', 'stream': asdict(spec),
            'inventory': registered.inventory.manifest(), 'microbatch_size': self.micro,
            'effective_batch_size': self.total, 'sampling': 'SHUFFLED_EPOCH_NO_DROP',
            'augmentation_scope': 'ONE_SHARED_MICROBATCH_TRANSFORM'}

    def _rows(self):
        rows = []
        while len(rows) < self.micro:
            take = min(self.micro - len(rows), self.count - self.cursor)
            rows.extend(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            if self.cursor == self.count:
                self.epoch += 1
                self.order = self.rng.permutation(self.count).tolist()
                self.cursor = 0
        return rows

    def next_update(self):
        self.registered.verify()
        if (asdict(self.spec) != self._contract['stream'] or self.micro != self._contract['microbatch_size']
                or self.total != self._contract['effective_batch_size']
                or self.registered.inventory.manifest() != self._contract['inventory']):
            raise ValueError('RR stream source/augmentation/batch contract changed')
        batches = []
        for _ in range(self.total // self.micro):
            aug = (bool(self.aug.random() < self.spec.hflip_probability),
                   bool(self.aug.random() < self.spec.vflip_probability),
                   int(self.aug.integers(4)) if self.spec.rot90 else 0)
            batches.append(self.registered.batch(self._rows(), device=self.device, augmentation=aug))
        return batches

    def state_dict(self):
        self.registered.verify()
        return deepcopy({'contract': self._contract, 'order': self.order, 'cursor': self.cursor,
                         'epoch': self.epoch, 'rng': self.rng.bit_generator.state,
                         'augmentation_rng': self.aug.bit_generator.state})

    def load_state_dict(self, state):
        self.registered.verify()
        if (state.get('contract') != self._contract or not isinstance(state.get('order'), list)
                or any(type(i) is not int for i in state['order'])
                or sorted(state['order']) != list(range(self.count))
                or type(state.get('cursor')) is not int or not 0 <= state['cursor'] < self.count
                or type(state.get('epoch')) is not int or state['epoch'] < 0):
            raise ValueError('RR stream donor/source/augmentation/order resume mismatch')
        rng, aug = np.random.default_rng(), np.random.default_rng()
        rng.bit_generator.state = deepcopy(state['rng'])
        aug.bit_generator.state = deepcopy(state['augmentation_rng'])
        self.rng, self.aug = rng, aug
        self.order, self.cursor, self.epoch = list(state['order']), state['cursor'], state['epoch']
