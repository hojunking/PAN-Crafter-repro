"""Fast synthetic checks; no inference, dataset or campaign side effects."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.ours_qualitative_render import render_scene, shared_rgb


class RenderTests(unittest.TestCase):
    def raw(self, bands=4):
        ref = np.arange(bands * 17 * 19, dtype=np.float32).reshape(bands, 17, 19) % 2048
        return {"gt": ref.copy(), "lms": ref.copy() * 0.9, "pan": ref[:1].copy()}

    def test_rr_native_dimensions_shared_bounds_no_input_mutation(self):
        raw = self.raw()
        pred = raw["gt"] + 10
        before = pred.copy()
        with tempfile.TemporaryDirectory() as tmp:
            result = render_scene(pred, raw, "QB", tmp, 0, ["B", "G", "R", "NIR"], 2047, "rr")
            cfg = result["configuration"]
            self.assertEqual(cfg["rgb_indices_zero_based"], [2, 1, 0])
            self.assertEqual(cfg["reference"], "RR GT")
            self.assertEqual(cfg["error_summary_dn"]["mean"], 10.0)
            for folder in ("output", "reference", "input_pan", "input_lms", "errormap"):
                with Image.open(Path(tmp) / folder / "testimg0.png") as image:
                    self.assertEqual(image.size, (19, 17))
                    self.assertIn("PREVIEW_ONLY", image.info["Status"])
            with Image.open(Path(tmp) / "output/testimg0.png") as image:
                np.testing.assert_array_equal(np.asarray(image), shared_rgb(pred, [2, 1, 0], cfg["rgb_bounds_dn"]))
            np.testing.assert_array_equal(pred, before)
            self.assertEqual(len(result["files"]), 6)
            self.assertFalse(cfg["paper_comparison_ready"])
            repeated = render_scene(pred, raw, "QB", tmp, 0, ["B", "G", "R", "NIR"], 2047, "rr")
            self.assertEqual(result, repeated)
            with self.assertRaises(FileExistsError):
                render_scene(pred + 1, raw, "QB", tmp, 0, ["B", "G", "R", "NIR"], 2047, "rr")

    def test_fr_uses_lms_not_gt_and_never_creates_error(self):
        raw = self.raw(8)
        del raw["gt"]
        with tempfile.TemporaryDirectory() as tmp:
            result = render_scene(raw["lms"], raw, "WV3", tmp, 19, None, 2047, "full")
            cfg = result["configuration"]
            self.assertEqual(cfg["rgb_indices_zero_based"], [4, 2, 1])
            self.assertEqual(cfg["reference"], "FR native LMS (not ground truth)")
            self.assertIsNone(cfg["error_formula"])
            self.assertFalse((Path(tmp) / "errormap").exists())
            self.assertEqual(len(result["files"]), 5)

    def test_band_order_is_explicit_and_respected(self):
        raw = self.raw()
        with tempfile.TemporaryDirectory() as tmp:
            result = render_scene(raw["gt"], raw, "QB", tmp, 0, ["R", "G", "B", "NIR"], 2047, "rr")
            self.assertEqual(result["configuration"]["rgb_indices_zero_based"], [0, 1, 2])
            with self.assertRaises(ValueError):
                render_scene(raw["gt"], raw, "QB", tmp, 1, None, 2047, "rr")

    def test_constant_bands_and_reject_invalid_shape_nonfinite(self):
        raw = {"gt": np.zeros((4, 10, 11)), "lms": np.zeros((4, 10, 11)), "pan": np.zeros((1, 10, 11))}
        with tempfile.TemporaryDirectory() as tmp:
            result = render_scene(raw["gt"], raw, "QB", tmp, 1, ["B", "G", "R", "NIR"], 2047, "rr")
            self.assertEqual(result["configuration"]["error_summary_dn"]["mean"], 0)
            with self.assertRaises(ValueError):
                render_scene(raw["gt"][:, :-1], raw, "QB", tmp, 2, ["B", "G", "R", "NIR"], 2047, "rr")
            raw["gt"][0, 0, 0] = float("nan")
            with self.assertRaises(ValueError):
                render_scene(raw["gt"], raw, "QB", tmp, 2, ["B", "G", "R", "NIR"], 2047, "rr")


if __name__ == "__main__":
    unittest.main()
