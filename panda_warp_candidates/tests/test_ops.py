"""Geometry, autograd, and explicit non-lossless counterexamples."""
import math
import unittest

import torch

from panda_warp_candidates.ops import (
    METHODS, candidate_manifest, fft_shift, warp_global, windowed_sinc_support_mask,
)


class WarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_zero_identity_shape_dtype_and_noncontiguous_input(self):
        generator = torch.Generator().manual_seed(123)
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(2, 3, 23, 27, generator=generator, dtype=dtype).transpose(-1, -2)
            for method in METHODS:
                y = warp_global(x, torch.zeros(2, 2, dtype=dtype), method=method, fft_padding=8)
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y.dtype, dtype)
                torch.testing.assert_close(y, x, atol=2e-6 if dtype == torch.float32 else 1e-12, rtol=0)

    def test_integer_axes_have_expected_sign_and_batch_channel_behavior(self):
        x = torch.arange(2 * 3 * 24 * 28, dtype=torch.float64).reshape(2, 3, 24, 28)
        d = torch.tensor([[1., -2.], [-2., 3.]], dtype=x.dtype)
        expected = torch.stack((x[0, :, 7:19, 4:20], x[1, :, 4:16, 9:25]))
        for method in METHODS:
            y = warp_global(x, d, method=method, fft_padding=8)
            torch.testing.assert_close(y[..., 6:18, 6:22], expected, atol=1e-10, rtol=0)

    def test_dc_and_finite_boundaries(self):
        x = torch.tensor([.3, -2.], dtype=torch.float64)[:, None, None, None].expand(2, 3, 20, 22)
        d = torch.tensor([[.23, -.46], [-3.2, 2.4]], dtype=x.dtype)
        for method in METHODS:
            y = warp_global(x, d, method=method, fft_padding=8)
            self.assertTrue(bool(torch.isfinite(y).all()))
            torch.testing.assert_close(y, x, atol=1e-12, rtol=0)

    def test_gradcheck_image_and_shift(self):
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(2, 2, 9, 10, generator=generator, dtype=torch.float64, requires_grad=True)
        d = torch.tensor([[.23, -.46], [-.32, .41]], dtype=x.dtype, requires_grad=True)
        for method in METHODS:
            with self.subTest(method=method):
                self.assertTrue(torch.autograd.gradcheck(
                    lambda image, delta: warp_global(image, delta, method=method, fft_padding=3),
                    (x, d), fast_mode=True, atol=1e-6, rtol=1e-4))

    def test_integer_shift_gradients_are_finite(self):
        generator = torch.Generator().manual_seed(512)
        x = torch.randn(1, 1, 12, 13, generator=generator, dtype=torch.float64)
        for method in METHODS:
            d = torch.zeros(1, 2, dtype=x.dtype, requires_grad=True)
            y = warp_global(x, d, method=method, fft_padding=4)
            gradient, = torch.autograd.grad(y[..., 3:8, 4:9].square().sum(), d)
            self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_fft_half_pixel_nyquist_is_not_lossless(self):
        x = ((-1.) ** torch.arange(32, dtype=torch.float64))[None, None, None].expand(1, 1, 32, 32)
        d = torch.tensor([[0., .5]], dtype=x.dtype)
        y = fft_shift(x, d, padding=8)
        z = fft_shift(y, -d, padding=8)
        self.assertLess(float(y.abs().max()), 1e-12)
        self.assertGreater(float((z - x).square().mean().sqrt()), .99)
        manifest = candidate_manifest("fft")
        self.assertFalse(manifest["pad_crop_roundtrip_guaranteed_lossless"])
        self.assertFalse(manifest["finite_observed_support_mask"])

    def test_fft_oneway_sine_has_small_but_nonzero_boundary_effect(self):
        xs = torch.arange(256, dtype=torch.float64)
        x = torch.cos(2 * math.pi * .125 * xs + .3)[None, None, None].expand(1, 1, 64, 256)
        ideal = torch.cos(2 * math.pi * .125 * (xs + .5) + .3)
        y = fft_shift(x, torch.tensor([[0., .5]], dtype=x.dtype), padding=32)
        interior = (y[0, 0, 32, 16:-16] - ideal[16:-16]).square().mean().sqrt()
        full = (y[0, 0, 32] - ideal).square().mean().sqrt()
        self.assertGreater(float(interior), 1e-6)
        self.assertLess(float(interior), .002)
        self.assertGreater(float(full), float(interior))

    def test_joint_nyquist_corner_has_rotation_counterexample(self):
        coordinate = torch.arange(32, dtype=torch.float64)
        x = ((-1.) ** (coordinate[:, None] + coordinate[None, :]))[None, None]
        d = torch.tensor([[.25, .25]], dtype=x.dtype)
        rotated_delta = torch.stack((-d[:, 1], d[:, 0]), dim=1)
        expected = torch.rot90(fft_shift(x, d, padding=8), 1, (-2, -1))
        actual = fft_shift(torch.rot90(x, 1, (-2, -1)), rotated_delta, padding=8)
        self.assertGreater(float((actual - expected).square().mean().sqrt()), .99)
        self.assertFalse(candidate_manifest("fft")["c4_equivariance_guaranteed"])

    def test_finite_tap_mask_does_not_reuse_bicubic_support(self):
        d = torch.tensor([[.2, -.4]])
        m4 = windowed_sinc_support_mask(24, 28, d, radius=4)
        m6 = windowed_sinc_support_mask(24, 28, d, radius=6)
        self.assertEqual(m6.shape, (1, 1, 24, 28))
        self.assertTrue(bool((~m6 | m4).all()))
        self.assertLess(int(m6.sum()), int(m4.sum()))
        self.assertFalse(bool(m6[..., 0, :].any()))
        self.assertTrue(bool(m6[..., 8:16, 8:20].all()))

    def test_invalid_inputs_fail_closed(self):
        x = torch.ones(1, 1, 12, 12)
        d = torch.zeros(1, 2)
        with self.assertRaises(ValueError): fft_shift(x, d, padding=12)
        with self.assertRaises(ValueError): fft_shift(x, d + 4, padding=4)
        with self.assertRaises(ValueError): fft_shift(x, d, padding=0)
        with self.assertRaises(ValueError): warp_global(x, d, method="unknown")
        with self.assertRaises(ValueError): warp_global(x.half(), d, method="lanczos4")
        with self.assertRaises(ValueError): warp_global(x, d + float("nan"), method="lanczos4")
        with self.assertRaises(ValueError): warp_global(x, d.expand(2, 2), method="lanczos4")


if __name__ == "__main__":
    unittest.main()
