"""Original FH12 sample/view stream, with a digest of all 50K updates."""
from __future__ import annotations

import hashlib
import numpy as np
from fh12.training import BatchStream
from fh12.data import build_dataset


def completed_updates(stream):
    return stream.epoch * stream.count + stream.cursor


def stream_manifest(n, batch_size, seed, total_updates=50000):
    """Hash the consumed index/rotation pairs, not worker-prefetch state.

    The original DataLoader receives explicit (index, rotation) tuples. H and V
    flips are always applied for these tuples; worker RNG does not choose views.
    This probe has independent generators and never consumes training RNG.
    """
    stream = BatchStream(n, batch_size, seed)
    digest = hashlib.sha256()
    left = int(total_updates)
    if left < 1:
        raise ValueError('A stream requires positive optimizer updates')
    while left:
        take = min(left, stream.count)
        pairs = np.stack((stream.order[:take * batch_size].numpy(),
                          stream.rotations[:take * batch_size].numpy()), axis=1)
        digest.update(pairs.astype('<i8', copy=False).tobytes())
        left -= take
        if left:
            stream.new_epoch()
    return dict(protocol='FH12_EXPLICIT_TUPLE_FIXED_HV_THEN_ROT_v1',
        n=int(n), batch_size=int(batch_size), seed=int(seed),
        total_updates=int(total_updates), drop_last=True, workers=4,
        data_seed=int(seed)+300000, augmentation_seed=int(seed)+400000,
        loader_seed='seed+500000+epoch', pair_dtype='<i8',
        sample_view_sequence_sha256=digest.hexdigest())


def epoch_loader(dataset, stream, seed, workers=4, pin_memory=True):
    """Exact original FH20R1 feeder construction, including worker generator."""
    import torch
    from torch.utils.data import DataLoader
    if stream.cursor == stream.count:
        stream.new_epoch()
    generator = torch.Generator().manual_seed(int(seed)+500000+stream.epoch)
    return DataLoader(dataset, batch_sampler=stream.remaining_batches(),
        num_workers=workers, pin_memory=pin_memory, generator=generator)
