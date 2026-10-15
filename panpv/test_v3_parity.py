"""Small synthetic oracles; never import the delivered provider implementation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from panpv.v3_parity import (PATHS, _sha, boundary_indices, compare_float32,
                            independent_taps, probe_windows, recompute_window, verify_scene)


class ReadRecorder:
    def __init__(self, array):
        self.array, self.shape, self.reads = array, array.shape, []

    def __getitem__(self, key):
        self.reads.append(key)
        if len(key) != 4 or any(not isinstance(k, slice) or k.start is None or k.stop is None
                                for k in key[-2:]):
            raise AssertionError("A complete source array was requested")
        if any(k.stop-k.start > 128 for k in key[-2:]):
            raise AssertionError("Unbounded source read")
        return self.array[key]


def reference_full(array, kind):
    """Small full-image 2-D convolution oracle with explicit np.pad boundaries."""
    _, channels, height, width = array.shape
    result = np.empty((1, channels, height//4, width//4), dtype=np.float32)
    for band in range(channels):
        offsets, weights, padding = independent_taps(kind, band)
        left, right = max(0, -int(offsets.min())), max(0, int(offsets.max()))
        padded = np.pad(array[0, band].astype(np.float64), ((left, right), (left, right)),
                        mode="edge" if padding == "replicate" else "reflect")
        kernel = weights[:, None]*weights[None, :]
        for row in range(height//4):
            for col in range(width//4):
                square = padded[np.ix_(left+4*row+offsets, left+4*col+offsets)]
                result[0, band, row, col] = np.sum(square*kernel, dtype=np.float64)
    return result


class IndependentV3ParityTests(unittest.TestCase):
    def test_declared_asymmetric_taps_and_phases(self):
        for kind, band, expected in [("rr_pan", 0, (-9, 12)), ("rr_ms", 0, (-6, 9)),
                                     ("rr_ms", 7, (-7, 10)), ("lp", 0, (-18, 22))]:
            offsets, weights, _ = independent_taps(kind, band)
            self.assertEqual((int(offsets[0]), int(offsets[-1])), expected)
            self.assertAlmostEqual(float(weights.sum()), 1., places=14)
            self.assertAlmostEqual(float(offsets @ weights), 2. if kind == "lp" else 1.5, places=13)
        with self.assertRaises(ValueError):
            independent_taps("rr_ms", 8)

    def test_boundary_padding_is_physical_not_local(self):
        indices = np.arange(-5, 9)
        ramp = np.arange(4)
        reflected = np.pad(ramp, (5, 5), mode="reflect")
        repeated = np.pad(ramp, (5, 5), mode="edge")
        np.testing.assert_array_equal(ramp[boundary_indices(indices, 4, "reflect")], reflected)
        np.testing.assert_array_equal(ramp[boundary_indices(indices, 4, "replicate")], repeated)
        np.testing.assert_array_equal(boundary_indices(indices, 1, "reflect"), np.zeros(len(indices)))

    def test_bounded_recomputation_matches_full_2d_oracle_all_bands(self):
        rng = np.random.default_rng(23)
        for kind, channels in (("rr_pan", 1), ("rr_ms", 8), ("lp", 1)):
            array = rng.integers(0, 2048, (1, channels, 128, 160), dtype=np.uint16)
            expected = reference_full(array, kind)
            reader = ReadRecorder(array)
            for window in ((0, 0, 16, 16), (16, 24, 16, 16), (7, 11, 16, 16)):
                row, col, height, width = window
                actual = recompute_window(reader, window, kind)
                report = compare_float32(actual, expected[0, :, row:row+height, col:col+width])
                self.assertTrue(report["passed"], report)
            self.assertTrue(reader.reads)

    def test_window_phase_is_global_and_not_probe_origin(self):
        y, x = np.mgrid[:256, :256]
        array = (3*y+5*x).astype(np.float32)[None, None]
        for kind, phase in (("rr_pan", 1.5), ("lp", 2.)):
            actual = recompute_window(ReadRecorder(array), [17, 21, 3, 4], kind)
            oy, ox = np.mgrid[17:20, 21:25]
            expected = (3*(4*oy+phase)+5*(4*ox+phase)).astype(np.float32)[None]
            np.testing.assert_allclose(actual, expected, atol=0, rtol=0)

    def test_invalid_or_unbounded_windows_fail_before_source_read(self):
        reader = ReadRecorder(np.zeros((1, 1, 128, 128), np.float32))
        for window in ([0, 0, 17, 16], [-1, 0, 16, 16], [30, 0, 16, 16], [0., 0, 16, 16]):
            with self.assertRaises(ValueError):
                recompute_window(reader, window, "lp")
        self.assertEqual(reader.reads, [])

    def test_two_ulp_limit_zero_and_nonfinite(self):
        reference = np.array([0., 1., 1024.], dtype=np.float32)
        two = np.nextafter(np.nextafter(reference, np.float32(np.inf)), np.float32(np.inf))
        three = np.nextafter(two, np.float32(np.inf))
        self.assertTrue(compare_float32(two, reference)["passed"])
        self.assertFalse(compare_float32(three, reference)["passed"])
        broken = reference.copy()
        broken[0] = np.nan
        self.assertFalse(compare_float32(broken, reference)["finite"])
        self.assertFalse(compare_float32(np.array([np.finfo(np.float32).max], np.float32),
                                         np.array([1.], np.float32))["passed"])

    def test_default_probes_include_edges_and_seams(self):
        windows = probe_windows((1800, 2198))
        self.assertIn([0, 0, 16, 16], windows)
        self.assertIn([1784, 2182, 16, 16], windows)
        for seam in (64, 128, 256, 512, 1024):
            self.assertTrue(any(row == seam-8 for row, _, _, _ in windows))
            self.assertTrue(any(col == seam-8 for _, col, _, _ in windows))
        self.assertEqual(len(windows), len({tuple(window) for window in windows}))

    def _write_small_scene(self, path):
        rng = np.random.default_rng(191)
        profiles = {key: {"test_only": key} for key in ("rr_profile", "lp_profile")}
        arrays = {"native/pan": rng.integers(0, 2048, (1, 1, 128, 128), dtype=np.uint16),
                  "native/ms": rng.integers(0, 2048, (1, 8, 32, 32), dtype=np.uint16)}
        for target, (source, kind, _) in PATHS.items():
            arrays[target] = reference_full(arrays[source], kind)
        with h5py.File(path, "w") as file:
            file.attrs.update(schema="PANPV_NATIVE_RR_SCENE_v3", complete=True, scene_id="SYNTHETIC")
            for key, profile in profiles.items():
                file.create_dataset("metadata/"+key+"_json", data=json.dumps(profile))
            for name, array in arrays.items():
                file.create_dataset(name, data=array)
        return {key: _sha(value) for key, value in profiles.items()}

    def test_h5_readonly_integration_and_profile_binding(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"scene.h5"
            hashes = self._write_small_scene(path)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "frozen v3 contract"):
                verify_scene(path)
            with patch("panpv.v3_parity.PROFILE_SHA256", hashes):
                result = verify_scene(path)
            self.assertTrue(result["passed"], result)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(set(result["results"]), set(PATHS))
            self.assertEqual(result["results"]["rr/ms"]["bands"], 8)

    def test_corrupted_derived_and_external_links_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"scene.h5"
            hashes = self._write_small_scene(path)
            with h5py.File(path, "r+") as file:
                file["rr/ms"][0, 7, 0, 0] += 1
            with patch("panpv.v3_parity.PROFILE_SHA256", hashes):
                result = verify_scene(path)
            self.assertFalse(result["passed"])
            self.assertGreater(result["results"]["rr/ms"]["failed_elements"], 0)
            with h5py.File(path, "r+") as file:
                del file["rr/ms"]
                file["rr/ms"] = h5py.ExternalLink("not_accessed.h5", "/ms")
            with patch("panpv.v3_parity.PROFILE_SHA256", hashes):
                with self.assertRaisesRegex(ValueError, "internal hard-linked"):
                    verify_scene(path)


if __name__ == "__main__":
    unittest.main()
