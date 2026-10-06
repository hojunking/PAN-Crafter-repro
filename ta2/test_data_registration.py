"""Synthetic evidence for coordinates, native LMS identity, and GT isolation."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import torch

from align.resample import augment_hr, interp23tap
from pa.warp import warp_pan
from ta2.data import (DatasetBundle, NativeDataset, NativeStream, audit_source,
                      operator_contract, sha256, transform_shift)
from ta2.registration import estimate_shift, fixed_band_weights, quadratic_peak, register_band


def fixture(path, *, bands=4, count=2, size=64, sentinel_gt=False):
    rng = np.random.default_rng(45)
    ms = rng.uniform(10, 50, (count, bands, size//4, size//4))
    lms = interp23tap(torch.from_numpy(ms)).numpy()
    with h5py.File(path, "w") as f:
        f["ms"], f["lms"] = ms, lms
        f["pan"] = lms.mean(1, keepdims=True)
        f["gt"] = np.full_like(lms, np.nan) if sentinel_gt else np.clip(lms, 0, 100)
    return ms, lms


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_parity_and_corrupted_lms_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"tiny.h5"
            fixture(path)
            result = audit_source(path, "train", bands=4, max_dn=100, phase_samples=1)
            self.assertEqual(result["lms_parity"]["status"], "PASS")
            self.assertLess(result["lms_parity"]["max_abs_dn"], 1e-9)
            self.assertIn("bicubic_vs_gt", result["phase_diagnostics"]["rows"][0])
            with h5py.File(path, "r+") as f:
                f["lms"][1, 0, 20, 20] += 0.01
            with self.assertRaisesRegex(ValueError, "BLOCKED_LMS_LINEAGE"):
                audit_source(path, "train", bands=4, max_dn=100, phase_samples=0)

    def test_fr_never_accesses_gt_sentinel(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"fr.h5"
            fixture(path, sentinel_gt=True)
            original = h5py.File.__getitem__
            def guarded(handle, key):
                if key in ("gt", "/gt"):
                    raise AssertionError("FR GT sentinel was accessed")
                return original(handle, key)
            with patch.object(h5py.File, "__getitem__", guarded):
                result = audit_source(path, "fr", bands=4, max_dn=100, phase_samples=1)
                dataset = NativeDataset(path, "fr", bands=4, max_dn=100)
                self.assertNotIn("gt", dataset.arrays)
                self.assertNotIn("gt", dataset.base(0))
            self.assertFalse(result["source_gt_accessed"])
            self.assertNotIn("lms_vs_gt", result["phase_diagnostics"]["rows"][0])

    def test_missing_lms_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"tiny.h5"
            fixture(path)
            with h5py.File(path, "r+") as f:
                del f["lms"]
            with self.assertRaisesRegex(ValueError, "BLOCKED_LMS_LINEAGE"):
                audit_source(path, "train", bands=4, max_dn=100, phase_samples=0)

    def test_lms_read_before_hr_augment(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"tiny.h5"
            _, lms = fixture(path)
            dataset = NativeDataset(path, "train", bands=4, max_dn=100)
            for rot in range(4):
                actual = dataset.get_view(0, rot)["lms"]
                expected = torch.from_numpy(np.rot90(lms[0, :, ::-1, ::-1], rot, axes=(1, 2)).copy().astype(np.float32)).mul_(2/100).sub_(1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(dataset.arrays["lms"].flags.writeable)
            self.assertEqual(dataset.get_view(0, 0)["hflip"], True)
            self.assertEqual(dataset.base(0)["hflip"], False)

    def test_source_binding_rejects_change(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"tiny.h5"
            fixture(path)
            identity = sha256(path)
            with h5py.File(path, "r+") as f:
                f["ms"][0, 0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                NativeDataset(path, "train", bands=4, max_dn=100, expected_sha=identity)

    def test_rng_stream_resume_and_global_isolation(self):
        random.seed(876)
        before = random.getstate()
        a, b = NativeStream(7, 991), NativeStream(7, 991)
        for _ in range(5):
            random.random()
            self.assertEqual(a.next(9), b.next(9))
        saved = json.loads(json.dumps(a.state_dict()))
        expected = a.next(48)
        c = NativeStream(7, 991)
        c.load_state_dict(saved)
        self.assertEqual(expected, c.next(48))
        prior = random.getstate()
        c.next(100)
        self.assertEqual(prior, random.getstate())
        self.assertNotEqual(before, prior)

    def test_lms_operator_uses_ms_only_and_phase(self):
        x = torch.zeros(1, 4, 16, 16, dtype=torch.float64)
        x[..., 5, 6] = 1
        y = interp23tap(x)
        self.assertEqual(np.unravel_index(y[0, 0].argmax().item(), y.shape[-2:]), (22, 26))
        self.assertFalse(operator_contract()["pan_or_gt_dependency"])
        self.assertFalse(operator_contract()["bicubic_fallback"])

    def test_case_shift_policies_and_gt_isolation(self):
        bundle = object.__new__(DatasetBundle)
        bundle.datasets = {s: [0]*4 for s in ("train", "val", "rr", "fr")}
        calls = []
        shifts = [[1., 3.], [3., 5.], [5., 7.], [7., 9.]]
        def offline(split, reference="NATIVE_LMS", descriptor="gradient", **kw):
            calls.append((split, reference, descriptor, kw))
            return {"shifts": shifts}
        bundle.offline_shifts = offline
        self.assertTrue(torch.equal(bundle.case_shifts("TA2-B08", "rr"), torch.tensor(shifts).roll(-1, 0)))
        self.assertTrue(torch.equal(bundle.case_shifts("TA2-B02", "fr"), torch.tensor([[4., 6.]]).repeat(4, 1)))
        self.assertEqual(calls[-1][0], "train")
        bundle.case_shifts("TA2-B06", "train")
        self.assertEqual(calls[-1][1], "GT")
        bundle.case_shifts("TA2-B06", "rr")
        self.assertEqual(calls[-1][1], "NATIVE_LMS")
        with self.assertRaises(ValueError):
            bundle.case_shifts("TA2-S07", "fr")
        bundle.case_shifts("TA2-U01", "val")
        self.assertEqual(calls[-1][1], "NATIVE_LMS")
        calls_before = len(calls)
        canonical = bundle.case_shifts("TA2-B03", "val")
        self.assertEqual(calls_before, len(calls))
        self.assertEqual(canonical.data_ptr(), bundle.case_shifts("TA2-D01", "val").data_ptr())
        sample_copy = canonical[[0, 1]]
        sample_copy.zero_()
        torch.testing.assert_close(canonical, torch.tensor(shifts))
        transformed = transform_shift(canonical[[0, 1]], [0, 1])
        transformed.add_(100)
        torch.testing.assert_close(canonical, torch.tensor(shifts))


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(10)
        self.reference = cv2.GaussianBlur(rng.uniform(0, 1, (96, 96)).astype(np.float32), (0, 0), 1.)

    def test_sampling_shift_sign_fractional_both_descriptors(self):
        for descriptor in ("gradient", "intensity"):
            for d in ([1., -2.], [.5, -.25], [-2., 1.]):
                p = warp_pan(torch.tensor(self.reference)[None, None], torch.tensor([d]))[0, 0].numpy()
                result = register_band(p, self.reference, descriptor=descriptor, sigma_p=.8, sigma_r=.8)
                self.assertTrue(result["valid"], result)
                self.assertLess(np.linalg.norm(np.asarray(result["shift"])+d), .2)

    def test_warp_augmentation_equivariance_and_gradient(self):
        p = torch.tensor(self.reference)[None, None].repeat(4, 1, 1, 1)
        c = torch.tensor([[.4, -.7]], requires_grad=True).repeat(4, 1)
        rot = torch.arange(4)
        yes = torch.ones(4)
        transformed = transform_shift(c, rot)
        lhs = augment_hr(warp_pan(p, c), yes, yes, rot)
        rhs = warp_pan(augment_hr(p, yes, yes, rot), transformed)
        torch.testing.assert_close(lhs[..., 8:-8, 8:-8], rhs[..., 8:-8, 8:-8], rtol=1e-5, atol=1e-5)
        self.assertIsNotNone(torch.autograd.grad(transformed.square().sum(), c)[0])

    def test_low_texture_and_boundary_zero_fallback(self):
        flat = np.ones((4, 96, 96), np.float32)
        result = estimate_shift(flat[0], flat, max_dn=1)
        self.assertFalse(result["valid"])
        self.assertEqual(result["shift"], [0., 0.])
        moved = warp_pan(torch.tensor(self.reference)[None, None], torch.tensor([[8., 0.]]))[0, 0].numpy()
        item = register_band(moved, self.reference, sigma_p=.8, sigma_r=.8)
        self.assertEqual(item["fallback_reason"], "GRID_BOUNDARY")
        self.assertEqual(item["shift"], [0., 0.])

    def test_invalid_quadratic_never_integer_fallback(self):
        surface = np.ones((17, 17), np.float64)
        self.assertEqual(quadratic_peak(surface, 8, 8)[1], "NOT_LOCAL_MAXIMUM")
        self.assertEqual(quadratic_peak(surface, 0, 8)[1], "GRID_BOUNDARY")

    def test_top_three_and_inference_gt_forbidden(self):
        ref = np.stack([self.reference]*4)
        result = estimate_shift(self.reference, ref, target="GT", split="train")
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["selected_bands"]), 3)
        for split in ("val", "rr", "fr"):
            with self.assertRaisesRegex(ValueError, "GT_REGISTRATION_FORBIDDEN"):
                estimate_shift(self.reference, ref, target="GT", split=split)
        oracle = estimate_shift(self.reference, ref, target="GT", split="rr", oracle=True)
        self.assertEqual(oracle["record_type"], "ORACLE_RR_ONLY")
        val = estimate_shift(self.reference, ref, target="GT", split="val", oracle=True)
        self.assertEqual(val["record_type"], "ORACLE_VALIDATION_ONLY")
        with self.assertRaisesRegex(ValueError, "GT_REGISTRATION_FORBIDDEN"):
            estimate_shift(self.reference, ref, target="GT", split="fr", oracle=True)

    def test_band_weights_train_lms_without_gt(self):
        p = self.reference[None, None]
        lms = np.repeat(p, 4, axis=1)
        result = fixed_band_weights(p, lms)
        np.testing.assert_allclose(result["weights"], np.ones((2, 4))/4)
        self.assertFalse(result["uses_gt"])


if __name__ == "__main__":
    unittest.main()
