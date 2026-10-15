"""Cache byte parity and ownership checks on synthetic v3 H5 inputs."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import torch

from .common import file_sha
from .datasetcache import CachedRRDataset, TENSOR_FIELDS
from .test_v3_data import write_scene
from .v3_data import RRPatchV3, SceneV3


class DatasetCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.path = write_scene(Path(self.temporary.name) / "scene.h5")
        with h5py.File(self.path, "r+") as source:
            source["rr/pan"][0, 0, 128, 128] = 0
            source["rr/pan"][0, 0, 128, 129] = 3000
            source["rr/pan_valid"][0, 0, 128, 130] = 0
        self.scene = SceneV3(self.path, expected_sha256=file_sha(self.path))

    def tearDown(self):
        self.scene.close()
        self.temporary.cleanup()

    def test_byte_identical_all_fields_all_splits(self):
        for split in ("train", "val", "audit"):
            source = RRPatchV3(self.scene, split)
            cache = CachedRRDataset(source)
            self.assertEqual(len(source), len(cache))
            for index in range(len(source)):
                expected, actual = source[index], cache[index]
                self.assertEqual(set(expected), set(actual))
                for key in expected:
                    if isinstance(expected[key], torch.Tensor):
                        self.assertEqual(expected[key].numpy().tobytes(), actual[key].numpy().tobytes())
                    else:
                        self.assertEqual(expected[key], actual[key])
            self.assertFalse(cache.provisional)
            self.assertEqual(cache.manifest["source_sha256"], self.scene.sha256)

    def test_mutating_every_returned_tensor_does_not_modify_cache_or_source(self):
        source = RRPatchV3(self.scene)
        cache = CachedRRDataset(source)
        modified = cache[0]
        for key in TENSOR_FIELDS:
            modified[key].zero_()
        modified["raw_index"] = 999
        modified["provisional"] = True
        reference = source[0]
        for key in TENSOR_FIELDS:
            torch.testing.assert_close(cache[0][key], reference[key], rtol=0, atol=0)
        self.assertEqual(cache[0]["raw_index"], reference["raw_index"])
        self.assertFalse(cache[0]["provisional"])

    def test_tensor_views_between_reads_have_distinct_storage(self):
        cache = CachedRRDataset(RRPatchV3(self.scene))
        one, two = cache[0], cache[0]
        for key in TENSOR_FIELDS:
            self.assertNotEqual(one[key].data_ptr(), two[key].data_ptr())

    def test_manifest_returns_no_mutable_reference(self):
        cache = CachedRRDataset(RRPatchV3(self.scene))
        value = cache.manifest
        value["source_sha256"] = "changed"
        self.assertEqual(cache.manifest["source_sha256"], self.scene.sha256)

    def test_cache_never_reloads_h5_for_reads(self):
        cache = CachedRRDataset(RRPatchV3(self.scene))
        with patch.object(self.scene, "read_region", side_effect=AssertionError("Unexpected H5 read")):
            cache[0]
            cache[0]

    def test_source_mutation_still_rejected_after_cache_materialization(self):
        cache = CachedRRDataset(RRPatchV3(self.scene))
        stat = self.path.stat()
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(ValueError, "changed since"):
            cache[0]

    def test_raw_or_unverified_data_not_promoted_by_cache(self):
        with self.assertRaisesRegex(ValueError, "SHA-verified"):
            CachedRRDataset(RRPatchV3(self.scene, normalize=False))
        with SceneV3(self.path, verify_bytes=False) as diagnostic:
            with self.assertRaisesRegex(ValueError, "SHA-verified"):
                CachedRRDataset(RRPatchV3(diagnostic))

    def test_low_memory_fails_without_mutating_source(self):
        before = file_sha(self.path)
        with patch("panpv.datasetcache.available_memory_bytes", return_value=1):
            with self.assertRaisesRegex(MemoryError, "SAFE_STOP_RAM"):
                CachedRRDataset(RRPatchV3(self.scene))
        self.assertEqual(file_sha(self.path), before)

    def test_indices_and_content_digest_are_stable(self):
        source = RRPatchV3(self.scene)
        first, second = CachedRRDataset(source), CachedRRDataset(source)
        self.assertEqual(first.manifest["content_sha256"], second.manifest["content_sha256"])
        self.assertEqual(first.manifest["identity_sha256"], second.manifest["identity_sha256"])
        for index in (-1, len(first)):
            with self.assertRaises(IndexError):
                first[index]
        for index in (False, 1.2, "0"):
            with self.assertRaises(TypeError):
                first[index]


if __name__ == "__main__":
    unittest.main()
