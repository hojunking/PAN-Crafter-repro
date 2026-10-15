"""Small synthetic checks of independent v3 scans and split replay.

No provider Python is imported, and no downloaded dataset is required.
"""
import hashlib
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from panpv.v3_preflight import scan_array, verify_split_generation


SCENE = "SYNTHETIC_SCENE"
NAMESPACE_SHA = "de9f938d7e73575168b7e1c70c87eda98d6091a2af0bae4520f082049c352f30"
# Golden PCG64 permutations for this explicit namespace, independent of replay.
PERMUTATIONS = {
    (2048, 2048): [3, 2, 0, 1],
    (2384, 2384): [8, 2, 3, 7, 6, 1, 5, 4, 0],
    (2336, 2336): [4, 3, 0, 1],
}


class RecordingDataset:
    """Proxy that rejects any attempt to read a whole scene/band collection."""
    def __init__(self, dataset, row_bound):
        self.dataset = dataset
        self.row_bound = row_bound
        self.reads = []
        self.ndim, self.shape = dataset.ndim, dataset.shape
        self.dtype, self.name = dataset.dtype, dataset.name

    def __getitem__(self, key):
        batch, channel, rows, columns = key
        if batch != 0 or not isinstance(channel, int):
            raise AssertionError("Must select exactly one NCHW batch/channel")
        if not isinstance(rows, slice) or rows.stop - rows.start > self.row_bound:
            raise AssertionError("Unbounded stripe")
        if columns != slice(None):
            raise AssertionError("Unexpected width selection")
        self.reads.append(key)
        return self.dataset[key]


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="panpv_preflight_test_")
        self.path = Path(self.temporary.name) / "synthetic.h5"

    def tearDown(self):
        self.temporary.cleanup()

    def scan(self, data, **kwargs):
        with h5py.File(self.path, "w") as handle:
            source = handle.create_dataset("image", data=data)
            return scan_array(source, **kwargs)

    def make_split_fixture(self, shape=(2048, 2048), filtered=False):
        height, width = shape
        blocks = []
        for row in range(0, height, 1024):
            for col in range(0, width, 1024):
                blocks.append([row, col, min(1024, height-row), min(1024, width-col), -1])
        order = PERMUTATIONS[shape]
        holdout = max(1, (len(order)+5)//10)
        for position, block_id in enumerate(order):
            blocks[block_id][4] = (0 if position < len(order)-2*holdout else
                                   1 if position < len(order)-holdout else 2)
        # Independently enumerate the entire global 64-pixel grid, then retain
        # cells whose 320x320 dependency region fits the assigned block.
        candidates = []
        for block_id, (r0, c0, bh, bw, code) in enumerate(blocks):
            if code < 0:
                continue
            for row in range(0, height, 64):
                if row-128 < r0 or row+64+128 > r0+bh:
                    continue
                for col in range(0, width, 64):
                    if col-128 >= c0 and col+64+128 <= c0+bw:
                        candidates.append((row, col, code, block_id))
        candidates = np.asarray(candidates, dtype="<i4").reshape(-1, 4)
        keep = np.ones(len(candidates), dtype=np.uint8)
        if filtered:
            keep[::17] = 0
        provenance = dict(schema="PANPV_SPATIAL_SPLIT_v1", scene_id=SCENE,
                          scene_shape_hw=list(shape), seed=261002,
                          block_size=1024, buffer=128, patch_size=64,
                          coordinate_frame="native_ms_pixels", bounds="half_open",
                          patch_origin_grid=[64, 64], patch_origin_phase=[0, 0],
                          rng="numpy.PCG64", rng_seed_hex=NAMESPACE_SHA[:32],
                          seed_namespace=dict(schema="PANPV_SPATIAL_SPLIT_v1",
                                              scene_id=SCENE, seed=261002),
                          seed_namespace_sha256=NAMESPACE_SHA)
        with h5py.File(self.path, "w") as handle:
            handle.attrs["scene_id"] = SCENE
            # Unwritten chunked array allocates no 8-band scene in RAM/on disk.
            handle.create_dataset("native/ms", shape=(1, 8, height, width),
                                  dtype="uint16", chunks=(1, 1, 64, 64))
            handle.create_dataset("splits/blocks", data=np.asarray(blocks, dtype="<i4"))
            handle.create_dataset("splits/candidate_patch_index", data=candidates)
            handle.create_dataset("splits/candidate_keep", data=keep)
            handle.create_dataset("splits/patch_index", data=candidates[keep.astype(bool)])
        return provenance, candidates, keep

    def test_scan_counts_every_band_and_partial_final_stripe(self):
        array = np.arange(2*7*5, dtype=np.float32).reshape(1, 2, 7, 5)
        array[0, 1, -1, -1] = 2048
        with h5py.File(self.path, "w") as handle:
            source = RecordingDataset(handle.create_dataset("image", data=array), row_bound=3)
            result = scan_array(source, row_chunk=3)
            self.assertEqual(len(source.reads), 6)
            self.assertEqual([key[2].start for key in source.reads], [0, 3, 6, 0, 3, 6])
        self.assertEqual(result["shape"], [1, 2, 7, 5])
        self.assertEqual(result["scan"], "ALL_PIXELS_BOUNDED_ROW_STRIPES")
        self.assertEqual(result["bands"], [
            dict(count=35, minimum=0., maximum=34., zero_count=1, above_2047_count=0),
            dict(count=35, minimum=35., maximum=2048., zero_count=0, above_2047_count=1)])

    def test_constant_zero_is_valid_dn_not_implicit_nodata(self):
        result = self.scan(np.zeros((1, 8, 3, 4), dtype=np.uint16), row_chunk=2)
        self.assertEqual(len(result["bands"]), 8)
        for band in result["bands"]:
            self.assertEqual(band, dict(count=12, minimum=0., maximum=0., zero_count=12,
                                        above_2047_count=0))

    def test_above_nominal_max_is_counted_not_silently_clipped(self):
        values = np.array([0, 2047, 2048, 65535], dtype=np.uint16).reshape(1, 1, 1, 4)
        band = self.scan(values)["bands"][0]
        self.assertEqual(band["maximum"], 65535.)
        self.assertEqual(band["above_2047_count"], 2)

    def test_binary_mask_accepts_both_values(self):
        values = np.array([0, 1, 1, 0, 1, 1], dtype=np.uint8).reshape(1, 1, 2, 3)
        band = self.scan(values, binary=True, row_chunk=1)["bands"][0]
        self.assertEqual((band["count"], band["zero_count"], band["maximum"]), (6, 2, 1.))

    def test_fractional_or_out_of_range_masks_are_rejected(self):
        for bad in (0.5, -1., 2.):
            values = np.ones((1, 1, 5, 3), dtype=np.float32)
            values[0, 0, -1, -1] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "Nonbinary"):
                self.scan(values, binary=True, row_chunk=2)

    def test_nonfinite_in_final_stripe_is_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            values = np.ones((1, 2, 5, 3), dtype=np.float32)
            values[0, 1, -1, -1] = bad
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "Nonfinite"):
                self.scan(values, row_chunk=2)

    def test_negative_unnormalized_dn_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Negative"):
            self.scan(np.array([-0.01], dtype=np.float32).reshape(1, 1, 1, 1))

    def test_wrong_axes_or_batch_size_are_rejected(self):
        for shape in ((3, 4), (8, 3, 4), (2, 1, 3, 4)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.scan(np.zeros(shape, dtype=np.uint8))

    def test_empty_axes_are_rejected_not_inf_statistics(self):
        for shape in ((0, 1, 3, 4), (1, 0, 3, 4), (1, 1, 0, 4), (1, 1, 3, 0)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.scan(np.zeros(shape, dtype=np.uint8))

    def test_row_chunk_must_be_positive_integer(self):
        for chunk in (0, -1, False, True, 1.5):
            with self.subTest(row_chunk=chunk), self.assertRaises((TypeError, ValueError)):
                self.scan(np.ones((1, 1, 3, 4), dtype=np.uint8), row_chunk=chunk)

    def test_golden_pcg64_assignments_and_candidates(self):
        provenance, candidates, keep = self.make_split_fixture()
        with h5py.File(self.path, "r") as handle:
            self.assertEqual(handle["splits/blocks"][:, 4].tolist(), [1, 2, 0, 0])
            result = verify_split_generation(handle, provenance)
        self.assertEqual(result["candidates"], 576)
        self.assertEqual(result["retained"], 576)
        self.assertEqual(result["namespace_sha256"], NAMESPACE_SHA)
        self.assertEqual(result["candidate_sha256"], hashlib.sha256(candidates.tobytes()).hexdigest())
        self.assertEqual(result["keep_sha256"], hashlib.sha256(keep.tobytes()).hexdigest())
        self.assertFalse(result["content_filter_recomputed"])

    def test_partial_blocks_and_partial_last_axes(self):
        provenance, _, _ = self.make_split_fixture((2384, 2384))
        with h5py.File(self.path, "r") as handle:
            result = verify_split_generation(handle, provenance)
            self.assertEqual(handle["splits/blocks"][-1, :4].tolist(), [2048, 2048, 336, 336])
        self.assertEqual(result["candidates"], 625)

    def test_unusable_small_partial_blocks_remain_unassigned(self):
        provenance, _, _ = self.make_split_fixture((2336, 2336))
        with h5py.File(self.path, "r") as handle:
            result = verify_split_generation(handle, provenance)
            self.assertEqual(int(np.count_nonzero(handle["splits/blocks"][:, 4] == -1)), 5)
        self.assertEqual(result["candidates"], 576)

    def test_frozen_filter_subset_is_preserved_not_recomputed(self):
        provenance, candidates, keep = self.make_split_fixture(filtered=True)
        with h5py.File(self.path, "r") as handle:
            result = verify_split_generation(handle, provenance)
        self.assertEqual(result["retained"], int(keep.sum()))
        self.assertLess(result["retained"], len(candidates))
        self.assertFalse(result["content_filter_recomputed"])

    def test_block_assignment_tampering_is_rejected(self):
        provenance, _, _ = self.make_split_fixture()
        with h5py.File(self.path, "r+") as handle:
            handle["splits/blocks"][0, 4] = 0
            with self.assertRaisesRegex(ValueError, "Block assignments"):
                verify_split_generation(handle, provenance)

    def test_candidate_coordinate_tampering_is_rejected(self):
        provenance, _, _ = self.make_split_fixture()
        with h5py.File(self.path, "r+") as handle:
            handle["splits/candidate_patch_index"][0, 0] += 4
            with self.assertRaisesRegex(ValueError, "Candidate patch list"):
                verify_split_generation(handle, provenance)

    def test_nonbinary_keep_flags_are_rejected(self):
        provenance, _, _ = self.make_split_fixture()
        with h5py.File(self.path, "r+") as handle:
            handle["splits/candidate_keep"][0] = 2
            with self.assertRaisesRegex(ValueError, "Invalid candidate keep"):
                verify_split_generation(handle, provenance)

    def test_keep_wrong_shape_is_rejected(self):
        provenance, _, keep = self.make_split_fixture()
        with h5py.File(self.path, "r+") as handle:
            del handle["splits/candidate_keep"]
            handle.create_dataset("splits/candidate_keep", data=keep[:-1])
            with self.assertRaisesRegex(ValueError, "Invalid candidate keep"):
                verify_split_generation(handle, provenance)

    def test_filtered_patch_list_tampering_is_rejected(self):
        provenance, _, _ = self.make_split_fixture()
        with h5py.File(self.path, "r+") as handle:
            handle["splits/patch_index"][0, 0] += 64
            with self.assertRaisesRegex(ValueError, "Filtered patch list"):
                verify_split_generation(handle, provenance)

    def test_changed_split_parameters_are_rejected(self):
        provenance, _, _ = self.make_split_fixture()
        for key, value in (("block_size", 512), ("buffer", 64), ("patch_size", 128)):
            with h5py.File(self.path, "r") as handle, self.subTest(key=key), self.assertRaises(ValueError):
                verify_split_generation(handle, dict(provenance, **{key: value}))

    def test_declared_namespace_must_match_replayed_namespace(self):
        provenance, _, _ = self.make_split_fixture()
        for key, value in (("schema", "unrelated"), ("seed", 17), ("scene_id", "OTHER"),
                           ("scene_shape_hw", [1024, 1024]), ("rng", "numpy.MT19937")):
            with h5py.File(self.path, "r") as handle, self.subTest(key=key), self.assertRaises(ValueError):
                verify_split_generation(handle, dict(provenance, **{key: value}))


if __name__ == "__main__":
    unittest.main()
