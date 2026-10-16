"""Registered RR-only inputs for the additive Conditional-q implementation.

The inventory identifies stored H5 rows, not claimed geographic scene IDs.
No validation/test/FR arrays are opened by calibration or training here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import h5py
import torch

from panda_method.calibration import canonical_sha256
from panda_method.data import require_sha256
from panda_method.edge_data import EdgeDataSources, EdgeObservationBatch
from panda_method.global_data import file_sha


@dataclass(frozen=True)
class TrainInventory:
    sensor: str
    sample_ids: tuple
    source_manifest_sha256: str
    catalog_sha256: str
    source_h5_sha256: str

    @property
    def sample_ids_sha256(self):
        return canonical_sha256(sorted(self.sample_ids))

    def manifest(self):
        return {"sensor": self.sensor, "source_split": "RR_train_only",
                "sample_ids": list(self.sample_ids),
                "sample_ids_sha256": self.sample_ids_sha256,
                "sample_count": len(self.sample_ids),
                "source_manifest_sha256": self.source_manifest_sha256,
                "catalog_sha256": self.catalog_sha256,
                "source_h5_sha256": self.source_h5_sha256,
                "sample_identity": "REGISTERED_H5_ROW_NOT_GEOGRAPHIC_SCENE",
                "native_lms_provenance": "PROVIDED_STORED_LMS",
                "radiometry": "2*stored_DN/peak-1", "train_HW": [64, 64]}


def validate_training_batch(batch, binding: Mapping, *, expected_sample_ids=None,
                            require_unaugmented=False):
    """Validate actual typed lineage and tensors, never caller flags alone."""
    if not isinstance(batch, EdgeObservationBatch):
        raise TypeError("Registered EdgeObservationBatch required")
    sensor = binding["sensor"]
    bands = {"WV3": 8, "GF2": 4, "QB": 4}.get(sensor)
    if bands is None:
        raise ValueError("Unknown sensor")
    require_sha256(binding["source_manifest_sha256"], "source manifest")
    batch.validate(bands, training=True, require_gt=True)
    line = batch.lineage
    if (line.sensor != sensor or line.source_manifest_sha256 != binding["source_manifest_sha256"]
            or line.reference != "PROVIDED_STORED_LMS" or line.grid_name != "PAN_RR"):
        raise ValueError("RR train sensor/manifest/native stored LMS binding mismatch")
    if batch.pan.shape[-2:] != (64, 64):
        raise ValueError("Registered RR train requires native 64x64 PAN patches")
    if any(t.dtype != torch.float32 for t in (batch.pan, batch.ms, batch.native_lms, batch.gt)):
        raise ValueError("Stage 2 requires finite FP32 observations without rescaling")
    ids = tuple(line.sample_ids)
    for name in ids:
        prefix = sensor + ":train:"
        suffix = name[len(prefix):] if name.startswith(prefix) else ""
        if not suffix.isdigit() or str(int(suffix)) != suffix:
            raise ValueError("Only canonical registered RR train sample IDs allowed")
        if "sample_count" in binding and (type(binding["sample_count"]) is not int
                or not 0 <= int(suffix) < binding["sample_count"]):
            raise ValueError("Sample outside pinned RR train population")
    if expected_sample_ids is not None and not set(ids).issubset(set(expected_sample_ids)):
        raise ValueError("Sample outside pinned RR train inventory")
    if require_unaugmented and any(a != (False, False, 0) for a in line.augmentations):
        raise ValueError("Calibration must consume unaugmented registered train rows")
    return batch


def batch_metadata(batch, binding=None, *, expected_sample_ids=None, require_unaugmented=False):
    if binding is None:
        if not isinstance(batch, EdgeObservationBatch):
            raise TypeError("Registered EdgeObservationBatch required")
        binding = {"sensor": batch.lineage.sensor,
                   "source_manifest_sha256": batch.lineage.source_manifest_sha256}
    validate_training_batch(batch, binding, expected_sample_ids=expected_sample_ids,
                            require_unaugmented=require_unaugmented)
    line = batch.lineage
    return {"sensor": line.sensor, "split": line.split, "resolution": line.resolution,
            "source_manifest_sha256": line.source_manifest_sha256,
            "native_lms_provenance": line.reference, "sample_ids": tuple(line.sample_ids),
            "augmentations": tuple(line.augmentations),
            "unaugmented": all(a == (False, False, 0) for a in line.augmentations),
            "sample_ids_sha256": canonical_sha256(sorted(line.sample_ids))}


class RegisteredRRTrain:
    """Pin the current catalog, actual train H5 and the complete row inventory."""
    def __init__(self, sources):
        if not isinstance(sources, EdgeDataSources):
            raise TypeError("Verified EdgeDataSources required")
        self.sources = sources
        path = sources._verify("train")
        with h5py.File(path, "r") as handle:
            count = len(handle["pan"])
            expected = {"pan": (count, 1, 64, 64),
                        "ms": (count, sources.bands, 16, 16),
                        "lms": (count, sources.bands, 64, 64),
                        "gt": (count, sources.bands, 64, 64)}
            if count < 1:
                raise ValueError("Empty RR train population")
            for key, shape in expected.items():
                value = handle.get(key)
                if (not isinstance(handle.get(key, getlink=True), h5py.HardLink)
                        or not isinstance(value, h5py.Dataset) or value.shape != shape
                        or value.is_virtual or value.external or value.dtype.kind not in "uif"):
                    raise ValueError("Registered RR train native tensor storage/shape mismatch")
        sources._verify("train")
        self.inventory = TrainInventory(sources.sensor,
            tuple(f"{sources.sensor}:train:{i}" for i in range(count)),
            sources.manifest_sha("train"), sources.catalog_sha256,
            sources.entry["splits"]["train"]["sha256"])
        self._expected_ids = frozenset(self.inventory.sample_ids)
        self._manifest = self.inventory.manifest()

    @classmethod
    def from_catalog(cls, catalog_path, *, root, sensor, lms_overrides=None):
        return cls(EdgeDataSources(catalog_path, root=root, sensor=sensor,
                                   lms_overrides=lms_overrides))

    @property
    def sensor(self):
        return self.sources.sensor

    @property
    def sample_ids(self):
        return self.inventory.sample_ids

    def verify(self):
        self.sources._verify("train")
        if "train" in self.sources.lms_overrides:
            self.sources._verify("train", lms=True)
        if (file_sha(self.sources.catalog_path) != self.inventory.catalog_sha256
                or self.sources.manifest_sha("train") != self.inventory.source_manifest_sha256
                or self.inventory.manifest() != self._manifest):
            raise ValueError("RR source registry/inventory changed after pinning")
        return self.inventory

    def batch(self, rows, *, device="cpu", augmentation=(False, False, 0)):
        self.verify()
        batch = self.sources.batch("train", list(rows), device=device, augmentation=augmentation)
        return validate_training_batch(batch, self._manifest, expected_sample_ids=self._expected_ids)

    def calibration_batches(self, batch_size, *, device="cpu"):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Explicit positive calibration batch size required")
        self.verify()
        count = len(self.sample_ids)
        for first in range(0, count, batch_size):
            batch = self.batch(range(first, min(first + batch_size, count)), device=device)
            validate_training_batch(batch, self._manifest, expected_sample_ids=self._expected_ids,
                                    require_unaugmented=True)
            yield batch
        self.verify()
