"""Numerical shift gradients, boundary support and executed Stage1 warp paths."""
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from panda_sensor5 import plan
from panda_sensor5.selection import PROTOCOL, validation_inventory
from panda_sensor5.stage1 import build_stage1
from panda_sensor5.warp import boundary_manifest, warp_pan, warp_support_mask
from ta2.registration import register_band


class UnifiedWarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_fractional_shift_gradcheck_and_zero_shift_ramp_gradient(self):
        generator = torch.Generator().manual_seed(37)
        image = torch.randn(1, 1, 12, 14, dtype=torch.float64, generator=generator)
        shift = torch.tensor([[.31, -.67]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda d: warp_pan(image, d),
            (shift,), eps=1.e-6, atol=1.e-5, rtol=1.e-4))
        y, x = torch.meshgrid(torch.arange(24, dtype=torch.float64),
                              torch.arange(28, dtype=torch.float64), indexing="ij")
        ramp = (2 * y + x)[None, None]
        zero = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        objective = lambda d: warp_pan(ramp, d)[..., 8:-8, 8:-8].mean()
        gradient, = torch.autograd.grad(objective(zero), zero)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(bool((gradient.abs() > .5).all()))
        for axis in range(2):
            delta = torch.zeros_like(zero); delta[0, axis] = 1.e-6
            finite_difference = (objective(delta) - objective(-delta)) / 2.e-6
            torch.testing.assert_close(gradient[0, axis], finite_difference,
                                       atol=1.e-5, rtol=1.e-5)

    def test_integer_sampling_sign_and_whole_sample_reflection(self):
        image = torch.arange(12 * 14, dtype=torch.float64).reshape(1, 1, 12, 14)
        actual = warp_pan(image, torch.tensor([[2., -3.]], dtype=torch.float64))
        def reflect(coordinate, length):
            position = coordinate % (2 * (length - 1))
            return position if position < length else 2 * (length - 1) - position
        expected = torch.tensor([[image[0, 0, reflect(y + 2, 12), reflect(x - 3, 14)]
                                  for x in range(14)] for y in range(12)], dtype=torch.float64)
        torch.testing.assert_close(actual[0, 0], expected, atol=1.e-12, rtol=0.)

    def test_r3_epsilon_observed_support_fits_fixed_view8(self):
        angles = torch.arange(360, dtype=torch.float32) * math.pi / 180
        shifts = 3 * torch.stack((angles.sin(), angles.cos()), dim=1)
        support = warp_support_mask(64, 64, shifts)
        self.assertTrue(bool(support[..., 8:-8, 8:-8].all()))
        self.assertTrue(bool(support[..., 7:-7, 7:-7].all()))
        self.assertFalse(bool(support[..., 6:-6, 6:-6].all()))
        self.assertEqual(boundary_manifest()["minimum_clean_input_margin"], 7)
        self.assertEqual(boundary_manifest()["aligner_view_margin"], 8)

    def test_stage1_native_and_epsilon_paths_execute_same_lanczos(self):
        from panda_sensor5.tests.test_core import sources
        with tempfile.TemporaryDirectory() as tmp:
            registered = sources(tmp, "QB", count=1)
            cfg = plan.resolve_case("s1", plan.DONORS[0])
            stage = build_stage1(cfg, registered.sources).stage
            with torch.no_grad():
                stage.aligner.fc2.bias.copy_(torch.tensor([.375, -.625]))
            rr = registered.batch([0])
            epsilon = torch.tensor([[.5, -1.25]])
            observed = []
            hook = stage.aligner.register_forward_pre_hook(
                lambda module, args: observed.append(args[0].detach().clone()))
            try:
                with patch("panda_sensor5.stage1.sample_epsilon", return_value=epsilon), \
                     patch("panda_sensor5.aligner.warp_pan", wraps=warp_pan) as native_warp, \
                     patch("panda_sensor5.stage1.warp_pan", wraps=warp_pan) as injected_warp, \
                     torch.no_grad():
                    result = stage(rr)
                self.assertEqual(native_warp.call_count, 1)
                self.assertEqual(injected_warp.call_count, 1)
                self.assertEqual(len(observed), 2)
                torch.testing.assert_close(observed[0], rr.pan, atol=0., rtol=0.)
                torch.testing.assert_close(observed[1], warp_pan(rr.pan, epsilon), atol=0., rtol=0.)
                torch.testing.assert_close(result["native"]["pan_aligned"],
                    warp_pan(rr.pan, result["native"]["u"]), atol=0., rtol=0.)
                self.assertEqual(float(result["epsilon_rr_view_support_fraction"]), 1.)
            finally:
                hook.remove()

    def test_validation_registration_support_and_rgb_only_inventory(self):
        self.assertEqual(PROTOCOL["margin"] - PROTOCOL["bound"] -
            math.ceil(3 * max(PROTOCOL["sigma_p"], PROTOCOL["sigma_r"])) - 1,
            PROTOCOL["source_support_margin"])
        shifts = torch.tensor([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]])
        support = warp_support_mask(64, 64, shifts)
        self.assertTrue(bool(support[..., 8:-8, 8:-8].all()))
        self.assertFalse(bool(support[..., 4:-4, 4:-4].all()))
        rng = np.random.default_rng(7)
        reference = rng.uniform(.2, .8, (64, 64)).astype("float32")
        pan = torch.from_numpy(reference)[None, None].expand(len(shifts), 1, 64, 64)
        shifted = warp_pan(pan, shifts).numpy()[:, 0]
        for image, delta in zip(shifted, shifts):
            registered = register_band(image, reference, descriptor="gradient", sigma_p=.8,
                sigma_r=.8, bound=8, margin=20)
            self.assertTrue(registered["valid"], registered)
            np.testing.assert_allclose(registered["shift"], -delta.numpy(), atol=.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation.h5"
            gt = np.zeros((2, 4, 64, 64), dtype="float32")
            gt[0, 3] = 2047 * reference  # NIR-only texture must be ineligible.
            gt[1, :3] = 2047 * reference
            with h5py.File(path, "w") as handle:
                handle["gt"] = gt
                handle["pan"] = np.broadcast_to(2047 * reference, (2, 1, 64, 64))
            source = SimpleNamespace(sensor="QB", bands=4, peak=2047,
                _verify=lambda split: path, manifest_sha=lambda split: "a" * 64,
                entry={"splits": {"val": {"sha256": "b" * 64}}}, catalog_sha256="c" * 64)
            inventory = validation_inventory(source)
            self.assertEqual(inventory["eligible"], [False, True])


if __name__ == "__main__":
    unittest.main()
