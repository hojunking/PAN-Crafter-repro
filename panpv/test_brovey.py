"""Actual GDAL (not mocked) invariants for the eight-band Brovey baseline."""
import unittest
import xml.etree.ElementTree as ET

import numpy as np
try:
    from osgeo import gdal
except ImportError:
    raise unittest.SkipTest('Actual Brovey tests require the dedicated PANPV GDAL image; no mock substitute')

from panpv.brovey import brovey8, config_metadata, _xml


class BroveyTests(unittest.TestCase):
    def test_constant_band_ratios_all_eight_outputs(self):
        ms = np.broadcast_to(np.arange(1, 9, dtype=np.float32)[:, None, None] * 100, (8, 4, 5)).copy()
        pan = np.full((16, 20), 900, dtype=np.float32)
        actual = brovey8(pan, ms)
        expected = np.broadcast_to((np.arange(1, 9) * 200)[:, None, None], actual.shape)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-4)

    def test_constant_ms_ramp_pan_and_eighth_band_preserved(self):
        ms = np.ones((8, 5, 7), dtype=np.float32) * 70
        pan = np.arange(20 * 28, dtype=np.float32).reshape(20, 28)
        actual = brovey8(pan, ms)
        np.testing.assert_allclose(actual, np.broadcast_to(pan, actual.shape), rtol=2e-6, atol=2e-4)

    def test_no_assumed_bit_depth_clip_or_rounding(self):
        ms = np.ones((8, 4, 4), dtype=np.float32) * 1000
        pan = np.full((16, 16), 8192.375, dtype=np.float32)
        actual = brovey8(pan, ms)
        np.testing.assert_array_equal(actual, np.broadcast_to(pan, actual.shape))
        self.assertGreater(float(actual.max()), 2047)

    def test_zero_pseudo_pan_yields_zero_without_epsilon(self):
        result = brovey8(np.full((16, 20), 700, np.float32), np.zeros((8, 4, 5), np.float32))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_zero_pan_is_valid_not_nodata(self):
        result = brovey8(np.zeros((16, 20), np.float32), np.ones((8, 4, 5), np.float32))
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_one_zero_band_does_not_invalidate_remaining_bands(self):
        ms = np.ones((8, 4, 5), dtype=np.float32) * 80
        ms[0] = 0
        result = brovey8(np.full((16, 20), 70, np.float32), ms)
        np.testing.assert_array_equal(result[0], np.zeros((16, 20), np.float32))
        np.testing.assert_allclose(result[1:], 80, rtol=1e-6, atol=1e-5)

    def test_actual_cubic_resampling_matches_independent_gdal_resample_formula(self):
        y, x = np.indices((7, 9), dtype=np.float32)
        ms = np.stack([(b + 1) * 30 + x * (b + 2) + y * 3 + 0.3 * x * y for b in range(8)])
        py, px = np.indices((28, 36), dtype=np.float32)
        pan = 100 + 2 * px + py
        dataset = gdal.GetDriverByName("MEM").Create("", 9, 7, 8, gdal.GDT_Float32)
        for b in range(8):
            dataset.GetRasterBand(b + 1).WriteArray(ms[b])
        up = dataset.ReadAsArray(buf_xsize=36, buf_ysize=28, resample_alg=gdal.GRIORA_Cubic).astype(np.float64)
        expected = up * (pan / up.mean(axis=0))[None]
        np.testing.assert_allclose(brovey8(pan, ms), expected, rtol=2e-6, atol=2e-4)

    def test_source_arrays_and_global_gdal_thread_setting_unchanged(self):
        pan = np.full((16, 20), 100, dtype=np.float64)
        ms = np.full((8, 4, 5), 200, dtype=np.uint16)
        originals = (pan.copy(), ms.copy())
        prior = gdal.GetConfigOption("GDAL_NUM_THREADS")
        try:
            gdal.SetConfigOption("GDAL_NUM_THREADS", "2")
            first, second = brovey8(pan, ms), brovey8(pan, ms)
            self.assertEqual(gdal.GetConfigOption("GDAL_NUM_THREADS"), "2")
            np.testing.assert_array_equal(first, second)
        finally:
            gdal.SetConfigOption("GDAL_NUM_THREADS", prior)
        np.testing.assert_array_equal(pan, originals[0])
        np.testing.assert_array_equal(ms, originals[1])

    def test_bad_geometry_negative_nonfinite_and_masks_rejected(self):
        pan, ms = np.ones((16, 20)), np.ones((8, 4, 5))
        for bad_pan, bad_ms in [(pan[:-1], ms), (pan[None], ms), (pan, ms[:3]),
                                (-pan, ms), (pan * np.nan, ms), (pan, ms * np.inf),
                                (pan, np.ma.array(ms, mask=False))]:
            with self.subTest(shape=np.shape(bad_pan)):
                with self.assertRaises(ValueError):
                    brovey8(bad_pan, bad_ms)

    def test_explicit_configuration_no_nbits_and_all_bands(self):
        tree = ET.fromstring(_xml(16, 20))
        options = tree.find("PansharpeningOptions")
        self.assertEqual(options.findtext("NumThreads"), "1")
        self.assertEqual(options.findtext("NoData"), "None")
        self.assertEqual(options.findtext("SpatialExtentAdjustment"), "None")
        self.assertIsNone(options.find("BitDepth"))
        self.assertEqual(len(options.findall("SpectralBand")), 8)
        metadata = config_metadata()
        self.assertEqual(metadata["software"]["gdal_release"], gdal.VersionInfo("RELEASE_NAME"))
        self.assertEqual(len(metadata["implementation_sha256"]), 64)
        self.assertEqual(metadata["config"]["output_clipping"], "none")


if __name__ == "__main__":
    unittest.main()
