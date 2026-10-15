"""Process-local, byte-preserving RAM cache for verified PANPV v3 RR splits.

No transformed data is written to disk. Public reads return owned tensor copies,
so in-place augmentation or a caller cannot mutate later samples. Original H5
identity remains checked on every read, even after all pixels are cached.
"""
from __future__ import annotations

import copy
import hashlib
import numbers
from pathlib import Path
import time

import torch

from .common import canonical_sha
from .training import _validate_item
from .v3_data import RRPatchV3


TENSOR_FIELDS = ("pan", "ms", "lpan", "gt", "mask", "coords")
FIELDS = frozenset(TENSOR_FIELDS) | {"raw_index", "provisional"}
RAM_RESERVE_BYTES = 4 * 1024 ** 3


def available_memory_bytes():
    """Linux MemAvailable includes reclaimable cache, unlike free-page counts."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot verify RAM headroom for materialized RR data")


class CachedRRDataset:
    def __init__(self, source):
        if not isinstance(source, RRPatchV3) or source.provisional or not source.normalize:
            raise ValueError("Only externally SHA-verified, normalized RRPatchV3 data can be cached")
        if not len(source):
            raise ValueError("Cannot cache an empty RR split")
        started = time.perf_counter()
        self._scene = source.scene
        self._scene._check_identity()
        self._length = len(source)
        self.split = source.split
        self.provisional = False
        first = source[0]
        self._validate(first)
        self._bytes = self._length * (sum(first[k].numel() * first[k].element_size()
                                          for k in TENSOR_FIELDS) + 8)
        if available_memory_bytes() < self._bytes + RAM_RESERVE_BYTES:
            raise MemoryError("SAFE_STOP_RAM: RR cache would leave less than 4 GiB available")
        self._storage = {key: torch.empty((self._length, *first[key].shape), dtype=first[key].dtype,
                                        device="cpu") for key in TENSOR_FIELDS}
        self._raw_indices = torch.empty(self._length, dtype=torch.int64)
        digest = hashlib.sha256()
        for index in range(self._length):
            item = first if index == 0 else source[index]
            self._validate(item)
            for key in TENSOR_FIELDS:
                tensor = item[key]
                storage = self._storage[key]
                if tensor.shape != storage.shape[1:] or tensor.dtype != storage.dtype:
                    raise ValueError("RR sample tensor shape/dtype changed while caching")
                storage[index].copy_(tensor)
                digest.update(key.encode() + str(tensor.dtype).encode())
                digest.update(storage[index].numpy().tobytes())
            self._raw_indices[index] = item["raw_index"]
            digest.update(int(item["raw_index"]).to_bytes(8, "little", signed=True))
        self._scene._check_identity()
        self._manifest = dict(schema="PANPV_RAM_RR_CACHE_v1", scene_id=self._scene.scene_id,
                              source_sha256=self._scene.sha256,
                              split_sha256=self._scene.summary()["patch_index_sha256"],
                              split=self.split, count=self._length, bytes=self._bytes,
                              content_sha256=digest.hexdigest(),
                              implementation="readonly_CPU_materialization_with_owned_read_copies",
                              normalization="UNCHANGED_FROM_VERIFIED_RRPatchV3",
                              seconds=time.perf_counter() - started)
        self._manifest["identity_sha256"] = canonical_sha({k: v for k, v in self._manifest.items()
                                                           if k != "seconds"})

    @staticmethod
    def _validate(item):
        if set(item) != FIELDS or item["provisional"] is not False:
            raise ValueError("Unexpected/provisional RR sample schema")
        if _validate_item(item) <= 0:
            raise ValueError("Empty-support RR sample cannot enter a verified cache")
        coords = item["coords"]
        if (not isinstance(coords, torch.Tensor) or coords.device.type != "cpu"
                or coords.dtype != torch.int64 or coords.shape != (4,)
                or isinstance(item["raw_index"], bool) or not isinstance(item["raw_index"], numbers.Integral)
                or item["raw_index"] < 0):
            raise ValueError("Unexpected coordinate/sample identity")

    @property
    def manifest(self):
        return copy.deepcopy(self._manifest)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, numbers.Integral):
            raise TypeError("RR cache requires a nonnegative integer sample index")
        if index < 0 or index >= self._length:
            raise IndexError(index)
        self._scene._check_identity()
        result = {key: tensor[int(index)].clone() for key, tensor in self._storage.items()}
        result.update(raw_index=int(self._raw_indices[int(index)]), provisional=False)
        return result
