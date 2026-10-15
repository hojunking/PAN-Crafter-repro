"""CPU-only synthetic checks; no private input or real experiment is modified."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import h5py
import numpy as np

from panpv.data import (BANDS, META_COLUMNS, RRDataset, _rect_conflicts,
                        freeze_split, inspect_rr_source, make_split_candidate,
                        prepare_lpan_cache, validate_split)


def write_source(path, *, side=12, size=64, coordinates=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = coordinates if coordinates is not None else [(x*size, y*size) for y in range(side) for x in range(side)]
    meta = np.array([[0, x, y, 0] for x, y in coords], dtype=np.int64)
    n = len(meta)
    with h5py.File(path, "w") as h5:
        h5.attrs.update(bands=BANDS, meta_cols=META_COLUMNS, ratio=4, img_scale=2047,
                        sensor="wv2", scene_index=json.dumps(["SCENE_A"]),
                        wald="Synthetic test arrays, not a verified scientific degradation")
        h5.create_dataset("meta", data=meta)
        pan = h5.create_dataset("pan", (n, 1, size, size), dtype="uint16")
        gt = h5.create_dataset("gt", (n, 8, size, size), dtype="uint16")
        ms = h5.create_dataset("ms", (n, 8, size//4, size//4), dtype="uint16")
        yy, xx = np.mgrid[:size, :size]
        for i, (_, x, y, _) in enumerate(meta):
            values = ((xx+x)*3+(yy+y)*7) % 2048
            pan[i, 0] = values
            gt[i] = np.repeat(values[None], 8, axis=0)
            ms[i] = np.repeat(values[2::4, 2::4][None], 8, axis=0)
    return path


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, **kwargs):
        return write_source(self.root / "input" / "rr.h5", **kwargs)

    def freeze(self, candidate):
        return freeze_split(candidate, self.root / "split.json", recipe_verified=True,
                            recipe_evidence=dict(test_only=True, declared_kernel_radius_in_rr=0,
                                                 lp_radius_rr=20, interpolation_radius_rr=8,
                                                 global_coordinate_origin_xy=[0, 0],
                                                 source_verification_reference="synthetic fixture"))

    def test_candidate_determinism_and_three_disjoint_partitions(self):
        source = self.source()
        a = make_split_candidate(source, block_rr=256)
        b = make_split_candidate(source, block_rr=256)
        self.assertEqual(a, b)
        self.assertTrue(a["provisional"])
        self.assertFalse(a["frozen"])
        self.assertTrue(validate_split(a))
        ids = [set(a["partitions"][key]) for key in ("train", "val", "audit")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertGreater(len(a["excluded_ids"]), 0)
        labels = {i: k for k, values in a["partitions"].items() for i in values}
        for i, j in _rect_conflicts(np.asarray(a["source"]["meta"]), 64, 128):
            self.assertFalse(i in labels and j in labels and labels[i] != labels[j])

    def test_freeze_requires_actual_recipe_evidence(self):
        candidate = make_split_candidate(self.source(), block_rr=256)
        with self.assertRaisesRegex(ValueError, "BLOCKED_DEGRADATION"):
            freeze_split(candidate, self.root / "split.json")
        with self.assertRaisesRegex(ValueError, "BLOCKED_DEGRADATION"):
            freeze_split(candidate, self.root / "split.json", recipe_verified=True)
        frozen = self.freeze(candidate)
        self.assertTrue(frozen["frozen"])
        self.assertFalse(frozen["provisional"])
        self.assertTrue(validate_split(frozen))
        self.assertEqual(frozen, self.freeze(candidate))

    def test_mutated_partition_is_rejected(self):
        candidate = make_split_candidate(self.source(), block_rr=256)
        candidate["partitions"]["train"][0] = candidate["partitions"]["audit"][0]
        with self.assertRaisesRegex(ValueError, "BLOCKED_IDENTITY"):
            validate_split(candidate)

    def test_unstructured_or_inadequate_recipe_evidence_is_blocked(self):
        candidate = make_split_candidate(self.source(), block_rr=256)
        with self.assertRaisesRegex(ValueError, "structured"):
            freeze_split(candidate, self.root / "split.json", recipe_verified=True,
                         recipe_evidence="assume known")
        evidence = dict(declared_kernel_radius_in_rr=50, lp_radius_rr=20,
                        interpolation_radius_rr=8, global_coordinate_origin_xy=[0, 0],
                        source_verification_reference="synthetic 78-pixel required support")
        with self.assertRaisesRegex(ValueError, "smaller than composed"):
            freeze_split(candidate, self.root / "split.json", recipe_verified=True,
                         recipe_evidence=evidence)

    def test_missing_manifest_hash_cannot_bypass_freeze(self):
        candidate = make_split_candidate(self.source(), block_rr=256)
        candidate.update(frozen=True, provisional=False, recipe_verified=True)
        with self.assertRaisesRegex(ValueError, "manifest digest"):
            validate_split(candidate)

    def test_diagonal_overlap_and_duplicate_rejected(self):
        source = self.source(side=3)
        with h5py.File(source, "r+") as h5:
            h5["meta"][1] = [0, 32, 32, 0]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            inspect_rr_source(source)

    def test_touching_rectangles_are_not_duplicate(self):
        data = np.array([[0, 0, 0, 0], [0, 64, 64, 0], [0, 64, 0, 0]])
        self.assertEqual(list(_rect_conflicts(data, 64)), [])
        self.assertEqual(len(list(_rect_conflicts(data, 64, 1))), 3)

    def test_external_links_rejected(self):
        source = self.source(side=3)
        with h5py.File(source, "r+") as h5:
            h5["outside"] = h5py.ExternalLink("/not/allowed/file.h5", "/x")
        with self.assertRaisesRegex(ValueError, "external/soft"):
            inspect_rr_source(source)

    def test_source_semantics_are_required(self):
        source = self.source(side=3)
        with h5py.File(source, "r+") as h5:
            del h5.attrs["meta_cols"]
        with self.assertRaisesRegex(ValueError, "provider"):
            inspect_rr_source(source)

    def test_global_lpan_uses_neighboring_tiles_and_matches_full_mosaic(self):
        source = self.source(side=4)
        manifest = prepare_lpan_cache(source, self.root / "cache", max_chunk_rows=64)
        actual = np.load(manifest["path"])
        with h5py.File(source) as h5:
            mosaic = np.zeros((256, 256), dtype=np.float64)
            for i, (_, x, y, _) in enumerate(h5["meta"][:]):
                mosaic[y:y+64, x:x+64] = h5["pan"][i, 0]
        kernel = cv2.getGaussianKernel(41, 1.98)
        kernel = kernel @ kernel.T
        expected = cv2.filter2D(mosaic, -1, kernel, borderType=cv2.BORDER_REPLICATE)[2::4, 2::4]
        for i in range(16):
            y, x = divmod(i, 4)
            np.testing.assert_allclose(actual[i, 0], expected[y*16:(y+1)*16, x*16:(x+1)*16], rtol=0, atol=0.00013)
        self.assertEqual(manifest["context_mode"], "GLOBAL_PROVIDED_RR_MOSAIC")
        self.assertFalse(manifest["rr_degraded_again"])
        self.assertEqual(manifest, prepare_lpan_cache(source, self.root / "cache", max_chunk_rows=256))
        self.assertFalse(list((self.root / "cache").glob(".rr-pan-mosaic-*")))

    def test_sparse_lpan_uses_full_tile_before_training_subcrop(self):
        coords = [(x*1024, y*1024) for y in range(2) for x in range(3)]
        source = self.source(size=256, coordinates=coords)
        candidate = make_split_candidate(source, guard_rr=64, block_rr=512)
        frozen = self.freeze(candidate)
        lp = prepare_lpan_cache(source, self.root / "cache")
        data = RRDataset(source, frozen, "train", lp, training_crop=64)
        self.assertEqual(len(data), len(frozen["partitions"]["train"])*16)
        first, inner = data[0], data[5]
        self.assertEqual(tuple(first["pan"].shape), (1, 64, 64))
        self.assertEqual(tuple(first["ms"].shape), (8, 16, 16))
        self.assertEqual(tuple(first["gt"].shape), (8, 64, 64))
        self.assertEqual(tuple(first["lpan"].shape), (1, 16, 16))
        self.assertFalse(first["mask"][0, :28].any())
        self.assertTrue(inner["mask"].all())
        self.assertEqual(first["raw_index"], inner["raw_index"])
        parent = first["raw_index"]
        original_lp = np.load(lp["path"], mmap_mode="r")[parent, :, 16:32, 16:32]
        np.testing.assert_array_equal(inner["lpan"].numpy(), original_lp.astype(np.float32)*(np.float32(2/2047))-np.float32(1))
        val = RRDataset(source, frozen, "val", lp)
        self.assertEqual(tuple(val[0]["gt"].shape), (8, 256, 256))
        with self.assertRaises(ValueError):
            RRDataset(source, frozen, "val", lp, training_crop=64)

    def test_provisional_dataset_is_explicitly_diagnostic_only(self):
        source = self.source(side=4)
        candidate = make_split_candidate(source, guard_rr=0, block_rr=128)
        lp = prepare_lpan_cache(source, self.root / "cache")
        with self.assertRaisesRegex(ValueError, "provisional"):
            RRDataset(source, candidate, "train", lp)
        data = RRDataset(source, candidate, "train", lp, allow_provisional=True)
        self.assertTrue(data[0]["provisional"])

    def test_normalization_mismatch_is_blocked(self):
        source = self.source()
        candidate = make_split_candidate(source, guard_rr=64, block_rr=256)
        lp = prepare_lpan_cache(source, self.root / "cache")
        with self.assertRaisesRegex(ValueError, "BLOCKED_NORMALIZATION"):
            RRDataset(source, self.freeze(candidate), "train", lp, max_dn=65535)


if __name__ == "__main__":
    unittest.main()
