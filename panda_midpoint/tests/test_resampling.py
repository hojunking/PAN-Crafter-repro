"""Numerical/phase checks for the post-augmentation LRMS upsampler."""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from align.resample import interp23tap
from panda_midpoint.resampling import upsample_lrms_after_augmentation


def _augment(x, augmentation):
    hflip, vflip, rotation = augmentation
    if hflip:
        x = x.flip(-1)
    if vflip:
        x = x.flip(-2)
    return torch.rot90(x, rotation, (-2, -1))


class ResamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def random_input(self, shape=(2, 3, 12, 16), dtype=torch.float64):
        return torch.randn(shape, dtype=dtype, generator=torch.Generator().manual_seed(19))

    def test_bundled_dlpan_reference_equivalence(self):
        path = (Path(__file__).resolve().parents[2] / "deploy/pandep_shared/project/src"
                / "pan_shared/metrics/interp23.py")
        spec = importlib.util.spec_from_file_location("midpoint_test_interp23_reference", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 2e-6)):
            x = self.random_input(dtype=dtype)
            actual = upsample_lrms_after_augmentation(x, ((False, False, 0),) * len(x))
            expected = torch.stack([
                torch.from_numpy(reference.interp23tap(
                    sample.permute(1, 2, 0).numpy(), 4)).permute(2, 0, 1)
                for sample in x
            ])
            torch.testing.assert_close(actual.double(), expected, rtol=0, atol=tolerance)

    def test_all_sixteen_transforms_match_paired_hr_augmentation(self):
        transformations = tuple(itertools.product((False, True), (False, True), range(4)))
        for dtype, tolerance in ((torch.float64, 3e-14), (torch.float32, 3e-6)):
            x = self.random_input(shape=(1, 2, 16, 16), dtype=dtype)
            augmented = torch.cat([_augment(x, a) for a in transformations])
            expected = torch.cat([_augment(interp23tap(x, 4), a) for a in transformations])
            actual = upsample_lrms_after_augmentation(augmented, transformations)
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(actual.device, x.device)
            torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)

    def test_rectangular_inputs_and_noncontiguous_rotations(self):
        x = self.random_input()
        for augmentation in itertools.product((False, True), (False, True), range(4)):
            transformed = _augment(x, augmentation)
            actual = upsample_lrms_after_augmentation(transformed, (augmentation,) * len(x))
            expected = _augment(interp23tap(x, 4), augmentation)
            self.assertEqual(actual.shape, (*transformed.shape[:2],
                                            4 * transformed.shape[2], 4 * transformed.shape[3]))
            torch.testing.assert_close(actual, expected, rtol=0, atol=3e-14)

    def test_impulse_native_phase_and_flipped_phase(self):
        x = torch.zeros(1, 1, 16, 16, dtype=torch.float64)
        x[0, 0, 3, 5] = 1
        for augmentation in itertools.product((False, True), (False, True), range(4)):
            actual = upsample_lrms_after_augmentation(_augment(x, augmentation), (augmentation,))
            expected_peak = torch.zeros(1, 1, 64, 64, dtype=torch.float64)
            expected_peak[0, 0, 4 * 3 + 2, 4 * 5 + 2] = 1
            expected_peak = _augment(expected_peak, augmentation)
            self.assertEqual(actual.argmax().item(), expected_peak.argmax().item())
        raw_flipped = interp23tap(x.flip(-1), 4)
        corrected = upsample_lrms_after_augmentation(x.flip(-1), ((True, False, 0),))
        self.assertEqual(raw_flipped.argmax().item() - corrected.argmax().item(), 1)

    def test_no_bicubic_sampler_and_no_extra_lr_augmentation(self):
        x = self.random_input()
        expected = torch.roll(interp23tap(x, 4), shifts=(-1, 0), dims=(-2, -1))
        with patch("torch.nn.functional.interpolate", side_effect=AssertionError("interpolate called")), \
             patch("torch.nn.functional.grid_sample", side_effect=AssertionError("grid_sample called")):
            actual = upsample_lrms_after_augmentation(x, ((False, True, 0),) * len(x))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_mixed_batch_gradients_match_explicit_reference(self):
        augmentations = ((False, False, 0), (True, False, 0), (False, True, 1))
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 3e-5)):
            x = self.random_input((3, 2, 8, 10), dtype).requires_grad_()
            independent = x.detach().clone().requires_grad_()
            actual = upsample_lrms_after_augmentation(x, augmentations)
            expected = torch.cat([
                torch.roll(interp23tap(independent[i:i + 1], 4), shifts=roll, dims=(-2, -1))
                for i, roll in enumerate(((0, 0), (0, -1), (-1, -1)))
            ])
            coefficients = self.random_input(tuple(actual.shape), dtype)
            (actual * coefficients).sum().backward()
            (expected * coefficients).sum().backward()
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
            self.assertGreater(float(x.grad.abs().sum()), 0)
            torch.testing.assert_close(x.grad, independent.grad, rtol=0, atol=tolerance)

    def test_cpu_autocast_cannot_reduce_dtype(self):
        x = self.random_input(dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = upsample_lrms_after_augmentation(x, ((False, False, 0),) * len(x))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, interp23tap(x, 4), rtol=0, atol=0)

    def test_invalid_metadata_rejected(self):
        x = self.random_input(shape=(1, 1, 8, 8))
        for metadata in (None, (), "bad", ((False, False),), ((False, False, 0, 0),),
                         ((0, False, 0),), ((False, 1, 0),), ((False, False, True),),
                         ((False, False, -1),), ((False, False, 4),),
                         ((False, False, 1.0),), ((False, False, np.int64(1)),),
                         ((False, False, 0),) * 2):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                upsample_lrms_after_augmentation(x, metadata)

    def test_invalid_shapes_dtype_and_nonfinite_inputs_rejected(self):
        for x in (torch.zeros(1, 8, 8), torch.zeros(0, 1, 8, 8),
                  torch.zeros(1, 0, 8, 8), torch.zeros(1, 1, 5, 8),
                  torch.zeros(1, 1, 8, 5), torch.zeros(1, 1, 8, 8, dtype=torch.float16),
                  torch.zeros(1, 1, 8, 8, dtype=torch.int64),
                  torch.full((1, 1, 8, 8), float("nan")),
                  torch.full((1, 1, 8, 8), float("inf"))):
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                upsample_lrms_after_augmentation(x, ((False, False, 0),))
        with self.assertRaises(TypeError):
            upsample_lrms_after_augmentation(np.zeros((1, 1, 8, 8)), ((False, False, 0),))


if __name__ == "__main__":
    unittest.main()
